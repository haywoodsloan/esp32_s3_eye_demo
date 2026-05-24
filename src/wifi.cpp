// Wi-Fi station bring-up.
//
// Standard ESP-IDF "WIFI_MODE_STA" pattern: init NVS (Wi-Fi needs it
// to cache calibration data), default netif + event loop, then
// esp_wifi_init / set_mode / set_config / start. Connection state is
// tracked via a FreeRTOS EventGroup so wifi_init() can block until we
// either have an IP or have run out of retries.
//
// Everything in this file is private to the module; the public API is
// in wifi.h. We avoid heap-allocating credentials -- the SSID and
// password are baked in at compile time via wifi_credentials.h, which
// keeps the password out of any debug log unless someone goes out of
// their way to print it.

#include "wifi.h"
#include "wifi_credentials.h"

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"

#include "esp_err.h"
#include "esp_check.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include <cstdio>
#include <cstring>

static const char *TAG = "wifi";

// EventGroup bits used to gate wifi_init() until association completes.
//   _CONNECTED bit: set on IP_EVENT_STA_GOT_IP.
//   _FAIL bit    : set after MAX_RETRY consecutive disconnects without a
//                  successful association in between.
static constexpr int WIFI_CONNECTED_BIT = BIT0;
static constexpr int WIFI_FAIL_BIT      = BIT1;

// How many times we'll re-try associate before giving up and returning
// an error from wifi_init(). After a successful association the
// counter resets, so a long-running session that briefly loses its AP
// keeps trying forever.
static constexpr int MAX_RETRY          = 6;

static EventGroupHandle_t s_event_group = nullptr;
static esp_netif_t       *s_sta_netif   = nullptr;
static int                s_retry       = 0;
static bool               s_connected   = false;
static bool               s_initialised = false;

static void on_wifi_event(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data)
{
    (void)arg;
    (void)event_data;
    if (event_base != WIFI_EVENT) {
        return;
    }
    switch (event_id) {
    case WIFI_EVENT_STA_START:
        // Kick off the first connection attempt now that the WiFi
        // driver has finished booting. STA_DISCONNECTED handles
        // every subsequent retry.
        esp_wifi_connect();
        break;
    case WIFI_EVENT_STA_DISCONNECTED:
        s_connected = false;
        if (s_retry < MAX_RETRY) {
            ++s_retry;
            ESP_LOGW(TAG, "disconnected; retry %d/%d", s_retry, MAX_RETRY);
            esp_wifi_connect();
        } else {
            ESP_LOGE(TAG, "exhausted %d retries -- giving up", MAX_RETRY);
            if (s_event_group) {
                xEventGroupSetBits(s_event_group, WIFI_FAIL_BIT);
            }
        }
        break;
    default:
        break;
    }
}

static void on_ip_event(void *arg, esp_event_base_t event_base,
                        int32_t event_id, void *event_data)
{
    (void)arg;
    if (event_base != IP_EVENT || event_id != IP_EVENT_STA_GOT_IP) {
        return;
    }
    auto *event = static_cast<ip_event_got_ip_t *>(event_data);
    ESP_LOGI(TAG, "got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    s_retry     = 0;
    s_connected = true;
    if (s_event_group) {
        xEventGroupSetBits(s_event_group, WIFI_CONNECTED_BIT);
    }
}

esp_err_t wifi_init(void)
{
    if (s_initialised && s_connected) {
        return ESP_OK;
    }
    if (s_initialised) {
        // Already brought up, but currently disconnected. The event
        // handler will keep retrying in the background; just report
        // status to the caller without re-initialising the stack.
        return s_connected ? ESP_OK : ESP_ERR_TIMEOUT;
    }

    // 1. NVS -- required by the WiFi driver for calibration / cred cache.
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES ||
        err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // The NVS partition is unusable -- erase and reformat. This is
        // the standard recipe out of the IDF examples; happens after
        // an OTA that grew the NVS schema.
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    // 2. Default netif + event loop + STA interface.
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_sta_netif = esp_netif_create_default_wifi_sta();

    // 3. WiFi driver.
    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));

    s_event_group = xEventGroupCreate();
    ESP_RETURN_ON_FALSE(s_event_group, ESP_ERR_NO_MEM, TAG,
                        "alloc wifi event group");

    esp_event_handler_instance_t any_id_inst, got_ip_inst;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, on_wifi_event,
        nullptr, &any_id_inst));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, on_ip_event,
        nullptr, &got_ip_inst));

    // Field-by-field init: ESP-IDF flips wifi_config_t members
    // between releases and the C++ designated-init rule about
    // declaration order would make the zero-init pattern fragile.
    wifi_config_t wifi_cfg = {};
    std::strncpy(reinterpret_cast<char *>(wifi_cfg.sta.ssid),
                 WIFI_SSID, sizeof(wifi_cfg.sta.ssid));
    std::strncpy(reinterpret_cast<char *>(wifi_cfg.sta.password),
                 WIFI_PASSWORD, sizeof(wifi_cfg.sta.password));
    // Accept anything from open to WPA2-PSK. Most home APs are WPA2;
    // WPA3-only APs would require explicit pmf_cfg.required = true.
    wifi_cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    wifi_cfg.sta.pmf_cfg.capable    = true;
    wifi_cfg.sta.pmf_cfg.required   = false;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    s_initialised = true;
    ESP_LOGI(TAG, "STA started, connecting to \"%s\"", WIFI_SSID);

    // Block until either we associate (and pick up an IP) or the
    // event handler reports a hard failure after MAX_RETRY tries.
    EventBits_t bits = xEventGroupWaitBits(
        s_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE, portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        return ESP_OK;
    }
    ESP_LOGE(TAG, "failed to connect to \"%s\"", WIFI_SSID);
    return ESP_ERR_TIMEOUT;
}

bool wifi_is_connected(void)
{
    return s_connected;
}

esp_err_t wifi_get_ipv4(char *out_ipv4_str, size_t buf_len)
{
    if (!out_ipv4_str || buf_len < 16) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!s_connected || !s_sta_netif) {
        return ESP_ERR_INVALID_STATE;
    }
    esp_netif_ip_info_t info = {};
    ESP_RETURN_ON_ERROR(esp_netif_get_ip_info(s_sta_netif, &info),
                        TAG, "get_ip_info");
    std::snprintf(out_ipv4_str, buf_len, IPSTR, IP2STR(&info.ip));
    return ESP_OK;
}
