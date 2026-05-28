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
[[nodiscard]] esp_err_t face_ai_init(void);

// True while the "NEW FACE DETECTED" banner should be composited onto
// the camera stream on the LCD. Cheap (single atomic load) — safe to
// poll from the render loop on every frame.
bool face_ai_banner_active(void);

// True while a recognised face's name should be drawn along the bottom
// edge of the live preview in navy blue. Fires whenever the matcher
// (or its IoU recognition cache) returns a known face whose user-set
// name is non-empty. Auto-clears a short window after the matcher
// stops returning that face, so the name disappears soon after the
// person leaves the frame. Cheap atomic load -- safe to poll every
// render frame.
bool face_ai_name_banner_active(void);

// True while the face_ai task is currently inside a heavy ESP-DL
// inference (detector or feature model). Render task should throttle
// itself when this returns true so the LCD-SPI -> octal-PSRAM read
// traffic doesn't contend with the inference's weight-streaming bus
// pressure. Cheap atomic load -- safe to poll every render frame.
bool face_ai_inference_busy(void);

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

// ---------------------------------------------------------------
// Known-face database (in-memory, resets on every boot)
// ---------------------------------------------------------------
//
// Each enrolment stores not just the embedding but also a small
// thumbnail crop of the face and an optional human-readable name.
// The web UI consumes the accessors below to render the gallery and
// to PATCH the names.

// Thumbnail dimensions and name buffer length are part of the public
// ABI (web UI + REST API both read FACE_THUMB_DIM*FACE_THUMB_DIM
// pixels). Declared as `static constexpr int` (not `#define`) so
// header consumers get type-safety and so the values appear in the
// debug symbol table.
static constexpr int FACE_THUMB_DIM = 64;
static constexpr int FACE_NAME_MAX  = 32;

typedef struct {
    int      idx;                 // 0-based position in the face DB
    uint32_t enrolled_ms;         // millis-since-boot when first enrolled
    int      thumb_w;             // thumbnail width in pixels (== FACE_THUMB_DIM)
    int      thumb_h;             // thumbnail height in pixels (== FACE_THUMB_DIM)
    char     name[FACE_NAME_MAX]; // user-set name; empty string if unset
} face_db_entry_t;

// Number of known faces currently in the database.
int  face_db_count(void);

// Copy metadata for face `idx` into *out. Returns false if `idx` is
// out of range. Thread-safe.
bool face_db_get_entry(int idx, face_db_entry_t *out);

// Copy thumbnail pixels (RGB565 big-endian, FACE_THUMB_DIM ** 2 pixels)
// for face `idx` into `dst`. `dst_capacity_pixels` must be at least
// FACE_THUMB_DIM*FACE_THUMB_DIM. Returns false if `idx` is out of
// range or `dst` is too small. Thread-safe.
bool face_db_copy_thumb(int idx, uint16_t *dst,
                        size_t dst_capacity_pixels);

// Set the human-readable name for face `idx`. The string is copied;
// at most FACE_NAME_MAX-1 chars are stored (truncated, NUL-terminated).
// Returns false if `idx` is out of range. Thread-safe.
bool face_db_set_name(int idx, const char *name);

// Remove face `idx` from the in-memory database. Subsequent entries
// shift down by one position, so any externally-held index past this
// one is now stale. Returns false if `idx` is out of range. Also
// invalidates the AI task's internal recognition cache so a freshly-
// deleted face can't continue to "match" via stale bbox-IoU. Thread-
// safe.
bool face_db_delete(int idx);
