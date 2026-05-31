#include "camera.h"
#include "board_pins.h"

#include "esp_check.h"
#include "esp_log.h"

#include <cstring>

static const char *TAG = "camera";

// Capture dimensions. The sensor is configured for 4:3 QVGA so the
// DSP downscaler operates at equal x/y ratios (true isotropic
// pixels); we then center-crop to a 240x240 square in software via
// camera_fb_get_square(). FRAMESIZE_240X240's claimed "1:1" output
// is in practice anamorphic on OV2640, so we don't use it. See the
// header for the full story.
static constexpr int CAM_CAPTURE_W = 320;
static constexpr int CAM_CAPTURE_H = 240;
static constexpr int CAM_SQUARE    = 240;
static constexpr int CAM_CROP_X    = (CAM_CAPTURE_W - CAM_SQUARE) / 2; // 40

esp_err_t camera_init(void)
{
    // C++ designated initialisers require declaration order and full
    // member coverage; ESP-IDF's camera_config_t shifts between releases,
    // so we zero-init and assign by name to stay version-resilient.
    camera_config_t cfg = {};
    cfg.pin_pwdn       = BOARD_CAM_PIN_PWDN;
    cfg.pin_reset      = BOARD_CAM_PIN_RESET;
    cfg.pin_xclk       = BOARD_CAM_PIN_XCLK;
    cfg.pin_sccb_sda   = BOARD_CAM_PIN_SIOD;
    cfg.pin_sccb_scl   = BOARD_CAM_PIN_SIOC;

    cfg.pin_d7         = BOARD_CAM_PIN_D7;
    cfg.pin_d6         = BOARD_CAM_PIN_D6;
    cfg.pin_d5         = BOARD_CAM_PIN_D5;
    cfg.pin_d4         = BOARD_CAM_PIN_D4;
    cfg.pin_d3         = BOARD_CAM_PIN_D3;
    cfg.pin_d2         = BOARD_CAM_PIN_D2;
    cfg.pin_d1         = BOARD_CAM_PIN_D1;
    cfg.pin_d0         = BOARD_CAM_PIN_D0;
    cfg.pin_vsync      = BOARD_CAM_PIN_VSYNC;
    cfg.pin_href       = BOARD_CAM_PIN_HREF;
    cfg.pin_pclk       = BOARD_CAM_PIN_PCLK;

    cfg.xclk_freq_hz   = BOARD_CAM_XCLK_HZ;
    cfg.ledc_timer     = LEDC_TIMER_0;
    cfg.ledc_channel   = LEDC_CHANNEL_0;

    cfg.pixel_format   = PIXFORMAT_RGB565;
    cfg.frame_size     = FRAMESIZE_QVGA;        // true-aspect 4:3, cropped to square in software
    cfg.jpeg_quality   = 12;                  // ignored for RGB565
    // Four buffers: one being filled by the camera, one being read by
    // the LCD DMA, one held by the AI task while it runs inference,
    // and one queued as the "latest" so neither consumer ever stalls
    // the producer. Costs an extra 115 KB of PSRAM.
    //
    // Tried fb_count=3 (saves ~115 KB PSRAM). Result: idle preview FPS
    // dropped 16 -> 10 (-38%), active FPS 12 -> 7.5 (-37%). The LCD
    // path waits on a free buffer during the ~700 ms feat inference.
    // Detect-only runs ~25 ms faster (PSRAM bus contention), but that
    // doesn't come close to making up for the preview regression. Do
    // not retry without first shortening feat inference.
    cfg.fb_count       = 4;
    cfg.fb_location    = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode      = CAMERA_GRAB_LATEST;

    ESP_RETURN_ON_ERROR(esp_camera_init(&cfg), TAG, "esp_camera_init");

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_hmirror(s, 0);
        s->set_vflip(s, 1);

        // Initial tuning is the MID preset of the adaptive AE bias
        // that face_ai.cpp drives at runtime. We start conservative
        // so the first second after boot can't over-expose a bright
        // scene before adaptation has any data; the face_ai task
        // then moves us toward DIM or BRIGHT once it has measured
        // the actual scene luma over a few frames. See the comment
        // block over apply_ae_preset() in face_ai.cpp for which
        // sensor knobs each preset controls.
        s->set_gainceiling(s, GAINCEILING_32X);
        s->set_aec2(s, 0);
        s->set_ae_level(s, 1);
        s->set_brightness(s, 1);
        s->set_contrast(s, 1);
        s->set_lenc(s, 1);
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
    }

    ESP_LOGI(TAG, "OV2640 ready: RGB565 %dx%d (capture) -> %dx%d (square) %d fb in PSRAM",
             CAM_CAPTURE_W, CAM_CAPTURE_H,
             CAM_SQUARE, CAM_SQUARE, cfg.fb_count);
    return ESP_OK;
}

camera_fb_t *camera_fb_get_square(void)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        return nullptr;
    }
    if (fb->width == CAM_SQUARE && fb->height == CAM_SQUARE) {
        // Already a square frame (e.g. a unit test or someone reconfigured
        // the sensor at runtime) -- nothing to crop.
        return fb;
    }
    if (fb->width != CAM_CAPTURE_W || fb->height != CAM_CAPTURE_H) {
        // Unexpected capture size -- return as-is and let the consumer
        // log/reject. We don't want to silently shred a buffer whose
        // layout we don't recognise.
        ESP_LOGW(TAG, "unexpected capture size %dx%d (expected %dx%d)",
                 (int)fb->width, (int)fb->height,
                 CAM_CAPTURE_W, CAM_CAPTURE_H);
        return fb;
    }

    // Center-crop in place: every output row y in [0, 240) takes the
    // 240-pixel slice [CAM_CROP_X, CAM_CROP_X + 240) of input row y
    // and packs it at offset y*240 of the same buffer. Dst always
    // starts at a lower byte offset than src for every row, so a
    // forward memmove is safe; we still use memmove (not memcpy) for
    // the row that straddles -- rows 0..2 -- where dst+rowlen actually
    // overlaps src.
    uint16_t *p = reinterpret_cast<uint16_t *>(fb->buf);
    for (int y = 0; y < CAM_SQUARE; ++y) {
        std::memmove(p + y * CAM_SQUARE,
                     p + y * CAM_CAPTURE_W + CAM_CROP_X,
                     CAM_SQUARE * sizeof(uint16_t));
    }
    fb->width  = CAM_SQUARE;
    fb->height = CAM_SQUARE;
    fb->len    = static_cast<size_t>(CAM_SQUARE) * CAM_SQUARE * sizeof(uint16_t);
    return fb;
}
