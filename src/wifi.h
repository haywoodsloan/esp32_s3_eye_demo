#pragma once

#include "esp_err.h"

// Initialise NVS, the default WiFi/netif/event-loop scaffolding, and
// connect to the SSID/password supplied via wifi_credentials.h in
// station mode. Blocks (with retries + backoff) until either the
// device is associated and has an IP address, or all retries have
// been exhausted.
//
// Returns ESP_OK once a DHCP lease has been obtained. On failure the
// device is left in STA disconnected state -- the caller can still
// proceed with the rest of the app; a later module can call this
// again to retry. Idempotent: subsequent calls after a successful
// connect short-circuit immediately.
esp_err_t wifi_init(void);

// True iff the station is currently associated and has an IP.
bool wifi_is_connected(void);

// Copy the current IPv4 address into `out_ipv4_str` (in printable
// "a.b.c.d" form). Returns ESP_OK on success, ESP_ERR_INVALID_STATE
// if not connected, or ESP_ERR_INVALID_ARG if `out_ipv4_str` is null
// or `buf_len` is too small for a full address.
esp_err_t wifi_get_ipv4(char *out_ipv4_str, size_t buf_len);
