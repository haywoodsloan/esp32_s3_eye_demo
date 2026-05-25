#include "camera.h"
#include "board_pins.h"

#include "esp_check.h"
#include "esp_log.h"

static const char *TAG = "camera";

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
    cfg.frame_size     = FRAMESIZE_240X240;
    cfg.jpeg_quality   = 12;                  // ignored for RGB565
    // Four buffers: one being filled by the camera, one being read by
    // the LCD DMA, one held by the AI task while it runs inference,
    // and one queued as the "latest" so neither consumer ever stalls
    // the producer. Costs an extra 115 KB of PSRAM.
    cfg.fb_count       = 4;
    cfg.fb_location    = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode      = CAMERA_GRAB_LATEST;

    ESP_RETURN_ON_ERROR(esp_camera_init(&cfg), TAG, "esp_camera_init");

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_hmirror(s, 0);
        s->set_vflip(s, 1);

        // Low-light tuning for the OV2640. The defaults are tuned for
        // outdoor / well-lit scenes and produce dim, low-contrast
        // frames indoors that defeat the face detector. Every knob
        // below either widens the AE / AGC range the sensor can use,
        // or biases what's left for the detector's benefit. We
        // deliberately do not touch any of the app-level "is this a
        // real face" gates (MIN_DETECT_SCORE, keypoints_look_upright,
        // MIN_SHARPNESS); the goal here is better INPUT, not a more
        // permissive classifier, so the false-positive surface
        // doesn't grow.
        //
        //   gainceiling 128x ceiling-the-stack analogue gain. CLAHE +
        //                    the keypoint-geometry check downstream
        //                    will clean up the noise; we'd rather
        //                    have a noisy face the detector can find
        //                    than a quiet black frame it can't.
        //   aec2 off         the OV2640's alternate AE algorithm
        //                    biases for highlight retention, which
        //                    leaves faces under-exposed in dim
        //                    rooms. The default AEC1 exposes a stop
        //                    or so brighter in those conditions,
        //                    which is what we actually want for
        //                    face detection.
        //   ae_level +2      maximum AE-target bias. Daytime scenes
        //                    clip a bit; indoor / dim scenes get
        //                    the boost they need.
        //   brightness +2    maximum post-AE additive offset; opens
        //                    the shadows that AE alone leaves crushed.
        //   contrast +1      slight S-curve so CLAHE downstream has
        //                    wider material to work with on truly
        //                    flat low-light frames.
        //   lenc on          lens shading correction. The OV2640 +
        //                    S3-EYE lens darkens noticeably near
        //                    the corners, exactly where the user's
        //                    face often ends up at close range.
        //   whitebal on +    explicit AWB-on. Defensive: a recent
        //   awb_gain on      sensor firmware bug had AWB off after
        //                    boot on some boards, which gave faces
        //                    a strong colour cast that hurt the
        //                    detector.
        s->set_gainceiling(s, GAINCEILING_128X);
        s->set_aec2(s, 0);
        s->set_ae_level(s, 2);
        s->set_brightness(s, 2);
        s->set_contrast(s, 1);
        s->set_lenc(s, 1);
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
    }

    ESP_LOGI(TAG, "OV2640 ready: RGB565 %dx%d, %d fb in PSRAM",
             BOARD_LCD_H_RES, BOARD_LCD_V_RES, cfg.fb_count);
    return ESP_OK;
}
