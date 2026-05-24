#pragma once

#include "esp_err.h"

#include <stdbool.h>
#include <stdint.h>

// Spin up the face-recognition task. Requires that the camera has been
// initialised (esp_camera_init). Also requires banner_init() to have
// been called, since this task drives banner_render() on enrollment.
// The task pulls its own frames from the camera; it does not interfere
// with the render task. The known-face list is held in memory and is
// reset on every boot. Returns ESP_OK on success.
esp_err_t face_ai_init(void);

// True while the "NEW FACE DETECTED" banner should be composited onto
// the camera stream on the LCD. Cheap (single atomic load) — safe to
// poll from the render loop on every frame.
bool face_ai_banner_active(void);

// Snapshot of the most recent successful face detection, transformed
// back into camera (display) frame coordinates so the render loop can
// draw the bbox + 5 keypoints directly on top of the live preview.
typedef struct {
    bool     valid;          // true if the last detection was successful
    uint32_t stamp_ms;       // millis-since-boot when the snapshot was written
    int16_t  box[4];         // x0, y0, x1, y1 axis-aligned in the display frame
    int16_t  keypoints[10];  // (x, y) * 5: eyes, nose, mouth corners
} face_overlay_t;

// Atomically copy the latest detection snapshot into *out. The caller
// should treat the data as stale if `(now_ms - out->stamp_ms)` exceeds
// the detector's expected per-frame budget. Safe to call from any task.
void face_ai_get_overlay(face_overlay_t *out);
