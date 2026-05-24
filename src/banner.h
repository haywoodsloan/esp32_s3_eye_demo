#pragma once

#include "esp_err.h"

#include <stdint.h>

// Pre-allocate the PSRAM-backed alpha mask used to overlay the
// "NEW FACE DETECTED" banner on top of the live camera feed.
// Idempotent. After this returns the mask contains an upright
// (angle 0) render of the banner text positioned at the bottom of
// the face's view.
esp_err_t banner_init(void);

// Re-rasterise the banner text into the alpha mask, rotated by
// `angle_rad` radians around the buffer centre (positive = clockwise
// in screen coordinates) so it reads upright relative to the detected
// face. Cheap enough to call on every banner-trigger. No-op if
// banner_init() has not been called.
void banner_render(float angle_rad);

// Composite the banner text into a board-sized RGB565 frame buffer
// in-place. Pixels where the alpha mask is zero are left untouched;
// pixels under the text are blended toward dark green using the mask
// value as alpha. Safe to call every frame; the cost is dominated by
// the (sparse) non-zero pixels of the text mask. `frame` must hold
// `width * height` RGB565 big-endian pixels matching the board's LCD
// resolution.
void banner_compose_overlay(uint16_t *frame, int width, int height);
