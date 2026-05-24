#pragma once

// =============================================================================
// Board pin map for the Espressif ESP32-S3-EYE (rev 2.x)
// =============================================================================

// ---- ST7789 1.3" 240x240 SPI LCD --------------------------------------------
// The Espressif BSP wires this panel to SPI3. Backlight on GPIO 48 is
// ACTIVE LOW (drive 0 to turn the backlight on).
#define BOARD_LCD_HOST            SPI3_HOST
#define BOARD_LCD_PIN_SCLK        21
#define BOARD_LCD_PIN_MOSI        47
#define BOARD_LCD_PIN_MISO        -1
#define BOARD_LCD_PIN_DC          43
#define BOARD_LCD_PIN_CS          44
#define BOARD_LCD_PIN_RST         -1
#define BOARD_LCD_PIN_BL          48
#define BOARD_LCD_BL_ON_LEVEL     0

#define BOARD_LCD_H_RES           240
#define BOARD_LCD_V_RES           240
#define BOARD_LCD_BITS_PER_PIXEL  16
#define BOARD_LCD_PIXEL_CLOCK_HZ  (40 * 1000 * 1000)

// ---- OV2640 camera ----------------------------------------------------------
#define BOARD_CAM_PIN_PWDN        -1
#define BOARD_CAM_PIN_RESET       -1
#define BOARD_CAM_PIN_XCLK        15
#define BOARD_CAM_PIN_SIOD         4
#define BOARD_CAM_PIN_SIOC         5
#define BOARD_CAM_PIN_D7          16
#define BOARD_CAM_PIN_D6          17
#define BOARD_CAM_PIN_D5          18
#define BOARD_CAM_PIN_D4          12
#define BOARD_CAM_PIN_D3          10
#define BOARD_CAM_PIN_D2           8
#define BOARD_CAM_PIN_D1           9
#define BOARD_CAM_PIN_D0          11
#define BOARD_CAM_PIN_VSYNC        6
#define BOARD_CAM_PIN_HREF         7
#define BOARD_CAM_PIN_PCLK        13

#define BOARD_CAM_XCLK_HZ         (20 * 1000 * 1000)
