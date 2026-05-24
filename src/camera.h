#pragma once

#include "esp_err.h"
#include "esp_camera.h"

// Initialise the on-board OV2640 camera in RGB565 / 240x240 mode, with
// double-buffered frames in PSRAM. Applies the BSP's default orientation
// (vflip=1, hmirror=0).
esp_err_t camera_init(void);
