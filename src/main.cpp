// ESP32-S3-EYE: continuously stream the on-board OV2640 camera to the
// 240x240 ST7789 LCD, while a background task runs ESP-WHO face
// detection + recognition on the same frames. When an unrecognised face
// passes the sharpness gate it gets enrolled in-memory and a dark-green
// "NEW FACE DETECTED" overlay is composited onto the live preview for
// a couple of seconds, oriented to read upright relative to the face.

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_camera.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_timer.h"

#include "banner.h"
#include "board_pins.h"
#include "camera.h"
#include "display.h"
#include "face_ai.h"
#include "wifi.h"

#include <algorithm>
#include <math.h>
#include <stdint.h>

static const char *TAG = "app";

// Detection-overlay constants. The bbox + keypoint dots are drawn in
// the same big-endian RGB565 byte order that the camera frame and the
// LCD both use, so we hardcode the BE-swapped 16-bit values.
//   BOX_COLOR:  pure green   (RGB565 0x07E0 -> BE 0xE007)
// Each keypoint index gets its own colour so the user can tell at a
// glance which feature the detector latched onto. Order matches ESP-WHO's
// MNP postprocessor output:
//   0: left eye           red       (0xF800 -> BE 0x00F8)
//   1: right eye          yellow    (0xFFE0 -> BE 0xE0FF)
//   2: nose tip           lime      (0x07E0 -> BE 0xE007)
//   3: left mouth corner  magenta   (0xF81F -> BE 0x1FF8)
//   4: right mouth corner cyan      (0x07FF -> BE 0xFF07)
static constexpr uint16_t OVERLAY_BOX_COLOR = 0xE007;
static constexpr uint16_t OVERLAY_DOT_COLORS[5] = {
    0x00F8,  // left eye           - red
    0xE0FF,  // right eye          - yellow
    0xE007,  // nose tip           - lime
    0x1FF8,  // left mouth corner  - magenta
    0xFF07,  // right mouth corner - cyan
};
// Edge thickness of the bbox in pixels, and radius of each keypoint
// dot. Chosen to match the look of the stock ESP32-S3-EYE demo.
static constexpr int      OVERLAY_BOX_THICKNESS = 2;
static constexpr int      OVERLAY_DOT_RADIUS    = 3;
// Detection results older than this are treated as stale and not
// drawn. face_ai now explicitly invalidates the overlay after a
// sustained miss streak, so this is purely a safety net for the
// pathological case where the AI task wedges; sized well over the
// AI-side OVERLAY_CLEAR_MISSES window so the two never disagree
// during normal operation.
static constexpr uint32_t OVERLAY_FRESH_MS = 2000;

static inline void overlay_fill_hline(uint16_t *fb, int w, int h,
                                      int x0, int x1, int y, uint16_t c)
{
    if (y < 0 || y >= h) return;
    if (x0 < 0) x0 = 0;
    if (x1 > w) x1 = w;
    uint16_t *row = fb + (size_t)y * w;
    for (int x = x0; x < x1; ++x) row[x] = c;
}

static inline void overlay_fill_vline(uint16_t *fb, int w, int h,
                                      int x, int y0, int y1, uint16_t c)
{
    if (x < 0 || x >= w) return;
    if (y0 < 0) y0 = 0;
    if (y1 > h) y1 = h;
    for (int y = y0; y < y1; ++y) fb[(size_t)y * w + x] = c;
}

static void overlay_draw_rect(uint16_t *fb, int w, int h,
                              int x0, int y0, int x1, int y1, uint16_t c)
{
    for (int t = 0; t < OVERLAY_BOX_THICKNESS; ++t) {
        overlay_fill_hline(fb, w, h, x0, x1, y0 + t,     c);
        overlay_fill_hline(fb, w, h, x0, x1, y1 - 1 - t, c);
        overlay_fill_vline(fb, w, h, x0 + t,     y0, y1, c);
        overlay_fill_vline(fb, w, h, x1 - 1 - t, y0, y1, c);
    }
}

static void overlay_draw_dot(uint16_t *fb, int w, int h,
                             int cx, int cy, int r, uint16_t c)
{
    const int r2 = r * r;
    for (int dy = -r; dy <= r; ++dy) {
        const int y = cy + dy;
        if (y < 0 || y >= h) continue;
        uint16_t *row = fb + (size_t)y * w;
        for (int dx = -r; dx <= r; ++dx) {
            if (dx * dx + dy * dy > r2) continue;
            const int x = cx + dx;
            if (x < 0 || x >= w) continue;
            row[x] = c;
        }
    }
}

static void overlay_draw_face(uint16_t *fb, int w, int h,
                              const face_overlay_t &ov)
{
    overlay_draw_rect(fb, w, h,
                      ov.box[0], ov.box[1], ov.box[2], ov.box[3],
                      OVERLAY_BOX_COLOR);
    for (int i = 0; i < 5; ++i) {
        const int kx = ov.keypoints[2 * i];
        const int ky = ov.keypoints[2 * i + 1];
        if (kx == 0 && ky == 0) continue;   // unset slot
        overlay_draw_dot(fb, w, h, kx, ky,
                         OVERLAY_DOT_RADIUS, OVERLAY_DOT_COLORS[i]);
    }
}

// Render task lives on core 0 alongside the camera HAL (cam_hal is
// pinned to core 0 via CONFIG_CAMERA_CORE0 in sdkconfig.defaults).
// Both are light, bursty consumers — the camera HAL services its DMA
// interrupts and the render task just queues one SPI transfer per
// frame — so they share core 0 comfortably. The face AI task is
// pinned to core 1 by itself so its ~50-100 ms inference runs in
// parallel with the camera/render pipeline instead of fighting it for
// CPU.
#define RENDER_TASK_CORE        0
#define RENDER_TASK_PRIORITY    5
#define RENDER_TASK_STACK       4096

static esp_err_t mount_face_db_partition(void)
{
    // SPIFFS partition for the face-recognition feature DB. The face_ai
    // task wipes it on every boot, so we don't care about its contents
    // persisting — but we do need the partition mounted and formatted.
    //
    // Zero-init + field assignment instead of a designated-init literal
    // so we don't have to track every IDF field addition or the C++
    // declaration-order rule.
    esp_vfs_spiffs_conf_t conf = {};
    conf.base_path              = "/spiffs";
    conf.partition_label        = "storage";
    conf.max_files              = 4;
    conf.format_if_mount_failed = true;
    esp_err_t err = esp_vfs_spiffs_register(&conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "spiffs mount failed: %s", esp_err_to_name(err));
        return err;
    }

    size_t total = 0, used = 0;
    if (esp_spiffs_info(conf.partition_label, &total, &used) == ESP_OK) {
        ESP_LOGI(TAG, "spiffs mounted: %u / %u bytes used",
                 (unsigned)used, (unsigned)total);
    }
    return ESP_OK;
}

static void render_task(void *arg)
{
    ESP_LOGI(TAG, "render task running on core %d", xPortGetCoreID());

    // We hold the frame buffer that the LCD DMA is currently reading and
    // only return it to the camera pool AFTER the next display_draw_rgb565()
    // has waited for that DMA to drain. Returning it any sooner would let
    // the camera overwrite it mid-transfer, producing tearing / ghosting.
    camera_fb_t *in_flight = NULL;

    uint32_t frames    = 0;
    int64_t  window_us = esp_timer_get_time();

    for (;;) {
        // Stream live frames to the panel every iteration; when the
        // banner is active we composite "NEW FACE DETECTED" on top of
        // the frame in-place before pushing it to the LCD. This replaces
        // the old behaviour where the entire screen was hijacked by a
        // static banner buffer.
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "frame capture failed");
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        if (face_ai_banner_active()) [[unlikely]] {
            // Mutating the camera buffer in-place is safe: the HAL
            // re-fills it via DMA on the next capture, fully overwriting
            // our changes, and display_draw_rgb565 issues a cache
            // writeback so the SPI DMA sees the composited pixels.
            banner_compose_overlay((uint16_t *)fb->buf,
                                   fb->width, fb->height);
        }

        // Draw the live face bbox + keypoints on top of the frame
        // (and on top of the banner, if any) so the user always sees
        // what the detector is tracking, like the stock ESP-WHO demo.
        // Snapshots are dropped if they're older than OVERLAY_FRESH_MS
        // so the overlay disappears promptly when the face leaves.
        face_overlay_t ov;
        face_ai_get_overlay(&ov);
        if (ov.valid) {
            const uint32_t now_ms_u =
                (uint32_t)(esp_timer_get_time() / 1000);
            if ((uint32_t)(now_ms_u - ov.stamp_ms) < OVERLAY_FRESH_MS) {
                overlay_draw_face((uint16_t *)fb->buf,
                                  fb->width, fb->height, ov);
            }
        }

        // Blocks until the previously queued LCD transfer has fully drained,
        // then queues this frame. Once this returns:
        //   - `in_flight` (from the previous iteration) is safe to release.
        //   - `fb->buf` is now the buffer the LCD DMA is reading from.
        display_draw_rgb565(0, 0, fb->width, fb->height, fb->buf);

        if (in_flight) {
            esp_camera_fb_return(in_flight);
        }
        in_flight = fb;

        if (++frames >= 60) {
            int64_t now_us  = esp_timer_get_time();
            int64_t dt_us   = now_us - window_us;
            if (dt_us > 0) {
                float fps = (float)frames * 1000000.0f / (float)dt_us;
                ESP_LOGI(TAG, "%.1f fps", fps);
            }
            frames    = 0;
            window_us = now_us;
        }
    }
}

// ESP-IDF's startup code looks up app_main by an unmangled symbol name,
// so it must keep C linkage even though this file is C++.
extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-S3-EYE face-recognition demo");

    ESP_ERROR_CHECK(display_init());
    ESP_ERROR_CHECK(camera_init());
    ESP_ERROR_CHECK(banner_init());
    ESP_ERROR_CHECK(mount_face_db_partition());
    ESP_ERROR_CHECK(face_ai_init());

    // Bring Wi-Fi up after the video / AI pipeline is already running so
    // the live preview starts immediately and a slow / failed Wi-Fi
    // associate doesn't gate the user-facing experience. wifi_init()
    // logs the IP on success; failure just leaves us offline.
    if (wifi_init() != ESP_OK) {
        ESP_LOGW(TAG, "continuing without Wi-Fi");
    }

    BaseType_t ok = xTaskCreatePinnedToCore(render_task, "render",
                                            RENDER_TASK_STACK, NULL,
                                            RENDER_TASK_PRIORITY, NULL,
                                            RENDER_TASK_CORE);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "failed to start render task");
    }
}
