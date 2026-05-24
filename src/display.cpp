#include "display.h"
#include "board_pins.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "esp_check.h"
#include "esp_log.h"
#include "esp_cache.h"
#include "esp_heap_caps.h"

#include "driver/gpio.h"
#include "driver/spi_master.h"

#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"

static const char *TAG = "display";

static esp_lcd_panel_io_handle_t s_io     = NULL;
static esp_lcd_panel_handle_t    s_panel  = NULL;
static SemaphoreHandle_t         s_idle   = NULL;

static bool IRAM_ATTR on_color_trans_done(esp_lcd_panel_io_handle_t io,
                                          esp_lcd_panel_io_event_data_t *edata,
                                          void *user_ctx)
{
    BaseType_t hp = pdFALSE;
    xSemaphoreGiveFromISR(s_idle, &hp);
    return hp == pdTRUE;
}

static esp_err_t backlight_init(void)
{
    // C++ designated initialisers require declaration order and full
    // member coverage; ESP-IDF struct layouts shift between releases, so
    // we zero-init and assign by name to stay version-resilient.
    gpio_config_t cfg = {};
    cfg.mode         = GPIO_MODE_OUTPUT;
    cfg.pin_bit_mask = 1ULL << BOARD_LCD_PIN_BL;
    ESP_RETURN_ON_ERROR(gpio_config(&cfg), TAG, "backlight gpio_config");
    // Hold the backlight off while the panel is being initialised.
    return gpio_set_level((gpio_num_t)BOARD_LCD_PIN_BL, !BOARD_LCD_BL_ON_LEVEL);
}

esp_err_t display_init(void)
{
    if (s_panel) {
        return ESP_OK;
    }

    s_idle = xSemaphoreCreateBinary();
    ESP_RETURN_ON_FALSE(s_idle, ESP_ERR_NO_MEM, TAG, "alloc idle semaphore");
    // Pre-give so the very first draw doesn't wait for a non-existent flush.
    xSemaphoreGive(s_idle);

    ESP_RETURN_ON_ERROR(backlight_init(), TAG, "backlight init");

    ESP_LOGI(TAG, "init SPI%d for LCD", BOARD_LCD_HOST + 1);
    spi_bus_config_t bus_cfg = {};
    bus_cfg.sclk_io_num     = BOARD_LCD_PIN_SCLK;
    bus_cfg.mosi_io_num     = BOARD_LCD_PIN_MOSI;
    bus_cfg.miso_io_num     = BOARD_LCD_PIN_MISO;
    bus_cfg.quadwp_io_num   = -1;
    bus_cfg.quadhd_io_num   = -1;
    bus_cfg.max_transfer_sz = BOARD_LCD_H_RES * BOARD_LCD_V_RES * sizeof(uint16_t);
    ESP_RETURN_ON_ERROR(spi_bus_initialize(BOARD_LCD_HOST, &bus_cfg,
                                           SPI_DMA_CH_AUTO),
                        TAG, "spi_bus_initialize");

    esp_lcd_panel_io_spi_config_t io_cfg = {};
    io_cfg.cs_gpio_num         = (gpio_num_t)BOARD_LCD_PIN_CS;
    io_cfg.dc_gpio_num         = (gpio_num_t)BOARD_LCD_PIN_DC;
    io_cfg.spi_mode            = 0;
    io_cfg.pclk_hz             = BOARD_LCD_PIXEL_CLOCK_HZ;
    io_cfg.trans_queue_depth   = 10;
    io_cfg.on_color_trans_done = on_color_trans_done;
    io_cfg.lcd_cmd_bits        = 8;
    io_cfg.lcd_param_bits      = 8;
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_io_spi(
                            (esp_lcd_spi_bus_handle_t)BOARD_LCD_HOST,
                            &io_cfg, &s_io),
                        TAG, "new_panel_io_spi");

    ESP_LOGI(TAG, "install ST7789 driver (%dx%d)",
             BOARD_LCD_H_RES, BOARD_LCD_V_RES);
    esp_lcd_panel_dev_config_t panel_cfg = {};
    panel_cfg.reset_gpio_num = (gpio_num_t)BOARD_LCD_PIN_RST;
    panel_cfg.rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_RGB;
    panel_cfg.bits_per_pixel = BOARD_LCD_BITS_PER_PIXEL;
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_st7789(s_io, &panel_cfg, &s_panel),
                        TAG, "new_panel_st7789");

    ESP_RETURN_ON_ERROR(esp_lcd_panel_reset(s_panel), TAG, "panel_reset");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_init(s_panel), TAG, "panel_init");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_invert_color(s_panel, true),
                        TAG, "invert_color");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_swap_xy(s_panel, false),
                        TAG, "swap_xy");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_mirror(s_panel, false, false),
                        TAG, "mirror");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_disp_on_off(s_panel, true),
                        TAG, "disp_on");

    // Panel is up, light it.
    return gpio_set_level((gpio_num_t)BOARD_LCD_PIN_BL, BOARD_LCD_BL_ON_LEVEL);
}

esp_err_t display_draw_rgb565(int x_start, int y_start,
                              int x_end,   int y_end,
                              const void *pixels)
{
    if (!s_panel) {
        return ESP_ERR_INVALID_STATE;
    }

    // Camera HAL on this board has "PSRAM DMA mode disabled" — frames are
    // DMA'd to internal SRAM then CPU-memcpy'd to PSRAM, which leaves
    // dirty cache lines over the framebuffer. The LCD SPI DMA reads
    // straight from physical PSRAM, so without a writeback we'd see stale
    // pixel data (visible as ghosting / partial-frame artifacts).
    //
    // The banner buffer is also in PSRAM; the cost of msync'ing it every
    // draw is negligible (the buffer is small and already clean after the
    // first call). UNALIGNED is set because nothing here guarantees a
    // cache-line aligned length.
    if (esp_ptr_external_ram(pixels)) {
        const int    w     = x_end - x_start;
        const int    h     = y_end - y_start;
        const size_t bytes = (size_t)w * (size_t)h * sizeof(uint16_t);
        esp_cache_msync((void *)pixels, bytes,
                        ESP_CACHE_MSYNC_FLAG_DIR_C2M |
                        ESP_CACHE_MSYNC_FLAG_UNALIGNED);
    }

    // Wait for the previously queued transfer to finish so the caller's
    // buffer is safe to mutate or recycle.
    xSemaphoreTake(s_idle, portMAX_DELAY);
    return esp_lcd_panel_draw_bitmap(s_panel, x_start, y_start,
                                     x_end, y_end, pixels);
}

void display_wait_idle(void)
{
    if (!s_idle) {
        return;
    }
    xSemaphoreTake(s_idle, portMAX_DELAY);
    xSemaphoreGive(s_idle);
}
