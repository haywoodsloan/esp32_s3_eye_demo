#pragma once

#include "esp_err.h"
#include "esp_camera.h"

// Initialise the on-board OV2640 camera in RGB565 / QVGA (320x240) mode
// with double-buffered frames in PSRAM. Applies the BSP's default
// orientation (vflip=1, hmirror=0). Despite the OV2640 advertising a
// FRAMESIZE_240X240 "1:1 crop" mode, that mode in practice produces
// anamorphic output (x-axis horizontally compressed relative to y on
// every OV2640 board we've tested, including the S3-EYE): a real face
// shows up taller than wide in the natural orientation and gets
// vertically squished in 90/270-rotated holds, which pushes the
// rotated detector inputs out of distribution and tanks accuracy on
// those orients. QVGA (320x240) is a true-aspect 4:3 capture, which
// we then center-crop to 240x240 in software via
// `camera_fb_get_square()`; that crop is guaranteed isotropic so the
// detector sees naturally-proportioned faces regardless of how the
// device is held.
esp_err_t camera_init(void);

// Fetch a camera frame and center-crop it in place to a true-isotropic
// 240x240 RGB565 view. Patches `fb->width`, `fb->height`, and
// `fb->len` in place so downstream code can keep treating the buffer
// as a contiguous 240x240 frame. Must be paired with
// `esp_camera_fb_return()` exactly like a raw `esp_camera_fb_get()`.
camera_fb_t *camera_fb_get_square(void);
