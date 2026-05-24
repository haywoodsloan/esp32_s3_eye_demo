#pragma once

#include <stdint.h>
#include "esp_err.h"

// Initialise the on-board ST7789 LCD: SPI bus, panel IO, panel driver,
// orientation, colour inversion, and backlight. Safe to call once at boot.
esp_err_t display_init(void);

// Push a tile of RGB565 pixels to the panel. Blocks until any previously
// queued transfer has completed before queueing the new one, so the caller
// may reuse / free `pixels` immediately after this function returns.
//
// `x_end` and `y_end` are exclusive (i.e. width = x_end - x_start).
esp_err_t display_draw_rgb565(int x_start, int y_start,
                              int x_end,   int y_end,
                              const void *pixels);

// Block until the most recent display_draw_rgb565() has fully flushed to
// the panel. Useful before tearing down a buffer that the LCD DMA might
// still be reading.
void display_wait_idle(void);
