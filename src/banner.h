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

// Re-rasterise the name banner: one line of UPPERCASE text placed at
// the bottom edge of the buffer, rotated by `angle_rad` so it reads
// upright relative to the detected face. Uses an independent alpha /
// outline buffer pair so it can coexist with the "NEW FACE DETECTED"
// banner during the enrollment window. `name` is uppercased and
// filtered to characters present in the bundled font; everything else
// (lowercase, accented, digits beyond 0-9, etc.) is mapped to the
// closest match or dropped. Callers should keep the visible name
// short -- the renderer truncates anything longer than fits on a
// single 240-px line. No-op if banner_init() has not been called.
void banner_render_name(const char *name, float angle_rad);

// Composite the name banner mask into a board-sized RGB565 frame
// in-place. Same semantics as banner_compose_overlay, but reads from
// the name banner's independent buffers and tints toward navy blue.
void banner_compose_name_overlay(uint16_t *frame, int width, int height);
