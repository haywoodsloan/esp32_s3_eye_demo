// Face-recognition task. Owns the ESP-WHO HumanFaceDetect detector plus
// a HumanFaceFeat embedder, and maintains its own in-memory list of
// known-face embeddings. Runs on core 0 so it shares CPU with the
// camera HAL but never touches core 1 (where the LCD render loop lives);
// the render task remains free to push camera frames at full FPS.
//
// Why we bypass HumanFaceRecognizer:
//   * Its match threshold is a private constant set to 0.5 in the
//     library; the public API exposes no setter. With the OV2640 +
//     MobileFaceNet pairing on a 240x240 RGB565 frame we see same-face
//     similarities in the 0.55-0.75 range and a non-trivial fraction of
//     frames slip below 0.5, which is enough to spawn a brand-new id
//     for an already-known face.
//   * The spec calls for an "internal list" that resets on board reset.
//     HumanFaceFeat + a std::vector here matches that wording exactly
//     and skips a SPIFFS round-trip we don't otherwise need.
// HumanFaceFeat::run returns an L2-normalised float embedding, so cosine
// similarity collapses to a plain dot product.

#include "face_ai.h"
#include "banner.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "human_face_detect.hpp"
#include "human_face_recognition.hpp" // HumanFaceFeat lives here
#include "dl_tensor_base.hpp"

#include "esp_dsp.h"          // dsps_dotprod_f32: SIMD dot product
#include "dsps_mem.h"         // dsps_memset_aes3 / dsps_memcpy_aes3: S3-optimised memory primitives

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <vector>

namespace
{

    constexpr const char *TAG = "face_ai";

    // Task placement. Pinned to core 1 because core 0 is already busy
    // with the camera HAL (cam_hal, prio 23) and the render task, both
    // of which are short bursts that finish quickly but happen at the
    // sensor's full frame rate. Inference on this task takes ~50-100 ms
    // per detection-frame and would otherwise starve the camera HAL of
    // CPU. Splitting them across cores lets the camera/render pipeline
    // run at full FPS while the AI loop chews through frames in
    // parallel.
    //
    // Priority is still kept above the render task's priority so that
    // when both happen to be waiting on the camera fb queue at the same
    // instant (rare — they're on different cores now — but possible)
    // FreeRTOS hands the next fb to the AI task. Render is the
    // background consumer that picks up every fb the AI loop hasn't
    // grabbed.
    constexpr UBaseType_t TASK_PRIO = 6;
    constexpr BaseType_t TASK_CORE = 1;
    constexpr uint32_t TASK_STACK = 16 * 1024;

    // Banner hold duration, in milliseconds (matches g_banner_until_ms).
    // The banner is now a non-intrusive overlay on top of the live
    // preview rather than a full-screen takeover, and the message is
    // split across the top and bottom edges of the screen — giving the
    // user a comfortable 5 s to read both halves while keeping the
    // centre of the frame clear of text.
    constexpr uint32_t BANNER_HOLD_MS = 5000;

    // Minimum face bounding-box edge in pixels. Tiny detections are usually
    // false positives or too small for the embedder to be reliable on.
    // On the 240x240 sensor frame, 83 px is ~35% of the frame edge (~12%
    // of the area); 15% larger than the previous 72 px gate so the user
    // has to be a bit closer before recognition / enrolment kicks in.
    constexpr int MIN_FACE_PX = 83;

    // Minimum detector confidence for a result to be acted on at all.
    // MSR+MNP's internal threshold is low enough that the detector
    // occasionally fires on rough face-like blobs (upside-down faces,
    // textured fabric, hand silhouettes), and once the orient-cycle
    // locks onto one of those false positives the HUD and the banner
    // start chasing the wrong target. Anything below this score is
    // treated as a miss and the orient cycle resumes.
    //
    // 0.60 was picked empirically: lower lets noisy faces through and
    // re-introduces the upside-down false-positive class; higher
    // starts costing real detections on far-from-camera or partially
    // occluded faces.
    // Minimum detector confidence for a result to be acted on at all.
    // The detector occasionally fires on rough face-like blobs and on
    // upside-down faces; the upside-down case is already caught much
    // more reliably by keypoints_look_upright(), so we can afford to
    // keep this score floor permissive. At close range the detector's
    // confidence drops (faces fill the frame, out of training
    // distribution), and in low-light scenes the model's confidence
    // drops even further -- we lean on the keypoint-geometry triple
    // check and the IoU recognition cache to suppress false positives
    // rather than this floor.
    constexpr float MIN_DETECT_SCORE = 0.40f;

    // Recognition cache. The embedder is the most expensive single
    // stage in the pipeline (~50 ms / call on the S8 model), and when
    // the user is sitting still the same face's bbox barely moves
    // between consecutive detection frames. We cache the last
    // recognition outcome (matched id or just-enrolled), and if the
    // next detection's bbox overlaps it strongly we skip the
    // embedder + cosine sweep entirely and treat it as a continuation
    // of the same identity.
    //
    // RECOG_CACHE_MS bounds the staleness so a swapped-in different
    // face that happens to land at the same place gets re-identified
    // within half a second. RECOG_IOU_PCT picks a strict-but-not-
    // paranoid overlap; head jitter still passes it, but someone
    // stepping into a different pose does not.
    constexpr uint32_t RECOG_CACHE_MS = 1500;
    constexpr int      RECOG_IOU_PCT  = 55;

    // Sharpness gate — average per-sample (|dx| + |dy|) on the green channel
    // inside the face bbox. Kept fairly permissive because the OV2640
    // is fixed-focus around ~30 cm: faces inside that range are genuinely
    // softer than the gate the original far-face tuning assumed. The
    // embedder copes well with mild blur; we mostly want this gate to
    // catch motion-smeared frames, not pose-related softness.
    constexpr int MIN_SHARPNESS = 15;

    // Cosine-similarity threshold for "this is an already-known face".
    // No longer a single constant -- the value lives in g_tuning and is
    // driven by the adaptive AE preset selected from frame mean luma.
    // See Tuning / apply_ae_preset() / tuning_for() below for the
    // per-preset values and the rationale.

    // Frames-in-a-row required before a fresh unknown face commits to
    // an enrolment. Also adaptive (per-preset) -- bright scenes can
    // safely enrol on a single frame, dim scenes want a second frame
    // of confirmation to avoid spuriously enrolling a known person
    // whose noisy embedding briefly dipped below match_thr.

    // Hard cap on the in-memory known-face list. The list resets on every
    // boot, but within a single power-on session this bounds how much
    // PSRAM the embedder vector can chew through if recognition keeps
    // mis-firing on the same person. ~32 * feat_len(=512) * 4 B ~= 64 KB,
    // well within budget on this board.
    constexpr size_t MAX_KNOWN_FACES = 32;

    // Throttle the AI loop when no face is in frame. Without this, the
    // detection-only path takes ~50-100 ms and starves the render task of
    // camera frames (both tasks block on the same fb queue, and AI's prio 6
    // outranks render's prio 5, so AI wins the wakeup every time the
    // camera HAL produces a new fb). Tighter than before (was 80 ms) to
    // minimise the wait between orientation-cycle tries; render still gets
    // the camera HAL's full output rate on the other core.
    constexpr uint32_t NO_FACE_DELAY_MS = 40;

    // Throttled diagnostics: one summary log per second so the serial
    // output is useful but never flooded.
    constexpr int64_t STATS_PERIOD_US = 1000LL * 1000LL;

    // Banner-end deadline as a millisecond count since boot, accessed
    // cross-task by the AI task (writer) and the render task + AI loop
    // (readers). Stored as uint32_t -- and NOT int64_t -- because the
    // Xtensa LX7 has lock-free 32-bit atomics (`s32c1i`) but no native
    // 8-byte CAS: `std::atomic<int64_t>` lowers to libatomic calls that
    // take a global mutex, which we'd be paying on every render frame
    // and every AI iteration. uint32 milliseconds gives ~49 days of
    // headroom from boot, well past any plausible session lifetime.
    std::atomic<uint32_t> g_banner_until_ms{0};

    // Latest face detection in DISPLAY-frame coordinates, published
    // each successful detection for the render task to draw on top of
    // the live preview. Guarded by a portMUX so the multi-field write
    // is atomic w.r.t. the render task on the other core.
    portMUX_TYPE          g_overlay_mux = portMUX_INITIALIZER_UNLOCKED;
    face_overlay_t        g_overlay     = {};

    // Known-face database. Each entry packages the L2-normalised
    // embedding (used by the matcher), a downsampled RGB565 thumbnail
    // cropped from the enrolment frame (served by the web UI), and a
    // user-set name (likewise mutated through the web UI). The mutex
    // gates every reader/writer; contention is rare (writers fire on
    // enrolment, the matcher reads inside one critical section per
    // detection, the HTTP handlers read on user demand) so a plain
    // FreeRTOS mutex is plenty.
    struct KnownFace
    {
        std::vector<float>    feat;
        std::vector<uint16_t> thumb;        // FACE_THUMB_DIM^2 RGB565BE pixels
        char                  name[FACE_NAME_MAX] = {0};
        uint32_t              enrolled_ms = 0;
    };
    std::vector<KnownFace> g_known_faces;
    SemaphoreHandle_t      g_db_mux = nullptr;

    inline uint32_t now_ms() noexcept
    {
        return static_cast<uint32_t>(esp_timer_get_time() / 1000);
    }

    // Cheap focus / motion-blur metric: average of |g(x,y) - g(x+1,y)| +
    // |g(x,y) - g(x,y+1)| over a strided sample of the face bbox, where g
    // is taken from the high byte of the BE-stored uint16_t pixel (green
    // dominates that byte in RGB565). A sharp face yields ~30-80, a blurry
    // or motion-smeared face ~5-15.
    int focus_metric(const uint16_t *__restrict__ rgb565, int img_w, int img_h,
                     int x0, int y0, int x1, int y1) noexcept
    {
        if (x0 < 1)
            x0 = 1;
        if (y0 < 1)
            y0 = 1;
        if (x1 > img_w - 2)
            x1 = img_w - 2;
        if (y1 > img_h - 2)
            y1 = img_h - 2;
        if (x1 - x0 < 8 || y1 - y0 < 8)
            return 0;

        int n = 0;
        int sum = 0;
        for (int y = y0; y < y1; y += 3)
        {
            const uint16_t *row = rgb565 + y * img_w;
            for (int x = x0; x < x1 - 1; x += 3)
            {
                const int g = (row[x] >> 8) & 0xFF;
                const int gx = (row[x + 1] >> 8) & 0xFF;
                const int gy = (row[x + img_w] >> 8) & 0xFF;
                sum += std::abs(g - gx) + std::abs(g - gy);
                ++n;
            }
        }
        return n ? sum / n : 0;
    }

    // Embeddings are L2-normalised by FeatPostprocessor::l2_norm() before
    // they leave the model, so cosine similarity collapses to a plain dot
    // product. ESP-DSP ships an Xtensa LX7 SIMD (PIE) implementation of
    // dsps_dotprod_f32 that hits roughly 6x the throughput of a plain C
    // loop on this chip, which matters as the known-face list grows.
    float cosine_sim(const float *__restrict__ a,
                     const float *__restrict__ b, int n) noexcept
    {
        float s = 0.f;
        dsps_dotprod_f32(a, b, &s, n);
        return s;
    }

    // --- Orientation handling -------------------------------------------------
    //
    // ESP-WHO's face detector is only trained on upright faces; rotate the
    // board 90 deg and detection stops working. We work around that by
    // pre-rotating each camera frame into a PSRAM scratch buffer (0/90/180/
    // /270 deg) and feeding the rotated frame to the detector. One
    // orientation is tried per frame; if it fails we advance to the next on
    // the next frame, so unsuccessful discovery costs at most 4 frames of
    // latency. Once an orientation succeeds we lock onto it (subsequent
    // frames re-use that orientation and detection is single-shot again)
    // until we lose the face, then the cycle resumes.

    enum class Orient : uint8_t
    {
        ROT_0 = 0,
        ROT_90_CW = 1,
        ROT_180 = 2,
        ROT_270_CW = 3
    };

    constexpr int FRAME_DIM = 240; // OV2640 is configured for 240x240 RGB565.

    // ---------------------------------------------------------------
    // Detector-input pipeline
    // ---------------------------------------------------------------
    //
    // For every frame we have to (potentially) (1) sample the frame to
    // decide whether to contrast-stretch, (2) rotate the frame to make
    // the face upright, and (3) apply the stretch. The naive version
    // did each step as a separate full pass over 240x240 RGB565 in
    // PSRAM, which on the S3 burns ~3-4 ms per pass on memory
    // bandwidth alone. The fused version below does all three in at
    // most one full pass plus one subsampled pass:
    //
    //   * analyze_luma_subsampled() walks every PREP_STRIDE-th pixel
    //     in both axes (=> 1/16 of the frame) to build a coarse luma
    //     histogram and pick the 2/98 percentile bounds. This is
    //     plenty of statistical resolution for a frame-level stretch
    //     decision and keeps the analysis pass effectively free.
    //
    //   * If the frame is already well-exposed *and* the requested
    //     orientation is ROT_0, we skip every subsequent step and
    //     point the detector straight at the camera fb (zero copy).
    //
    //   * Otherwise prep_detect_input() runs exactly one pass that
    //     fuses the rotation index math with an optional per-channel
    //     LUT apply (the contrast stretch). The LUT branch is hoisted
    //     out of the hot loop via a non-type template parameter so
    //     the inner loop is straight-line on each instantiation.
    //
    // Net effect on the worst case (need stretch + rotate) drops from
    // 3 full PSRAM passes down to ~1.06. The common-case live-preview
    // path (well exposed, ROT_0 locked) drops to a single subsampled
    // pass.

    // Per-pixel kernel: read one RGB565BE word from `sp`, optionally
    // run its three channels through `lut`, and write the result to
    // `dp`. With WithLut=false this collapses to a single uint16_t
    // copy; with WithLut=true it's load-byte / unpack / 3x LUT /
    // repack / store-byte.
    template <bool WithLut>
    inline void copy_pixel(const uint16_t *__restrict__ sp,
                           uint16_t *__restrict__ dp,
                           const uint8_t *lut) noexcept
    {
        if constexpr (!WithLut) {
            *dp = *sp;
        } else {
            const uint8_t *sb = (const uint8_t *)sp;
            uint8_t       *db = (uint8_t *)dp;
            const uint8_t hi = sb[0];
            const uint8_t lo = sb[1];
            const uint8_t r5 = (uint8_t)(hi >> 3);
            const uint8_t g6 = (uint8_t)(((hi & 0x07) << 3) | (lo >> 5));
            const uint8_t b5 = (uint8_t)(lo & 0x1F);
            const uint8_t r8 = (uint8_t)((r5 << 3) | (r5 >> 2));
            const uint8_t g8 = (uint8_t)((g6 << 2) | (g6 >> 4));
            const uint8_t b8 = (uint8_t)((b5 << 3) | (b5 >> 2));
            const uint8_t nr5 = (uint8_t)(lut[r8] >> 3);
            const uint8_t ng6 = (uint8_t)(lut[g8] >> 2);
            const uint8_t nb5 = (uint8_t)(lut[b8] >> 3);
            db[0] = (uint8_t)((nr5 << 3) | (ng6 >> 3));
            db[1] = (uint8_t)((ng6 << 5) | nb5);
        }
    }

    // Fused rotate + optional stretch. Each orient gets a dedicated
    // double-loop with branch-free indexing; WithLut is a template
    // parameter so the inner kernel is straight-line in either mode.
    template <bool WithLut>
    void rotate_to_scratch(Orient o,
                           const uint16_t *__restrict__ src,
                           uint16_t *__restrict__ dst,
                           int n, const uint8_t *lut) noexcept
    {
        switch (o) {
        case Orient::ROT_0: {
            if constexpr (!WithLut) {
                // dsps_memcpy_aes3 is the S3 SIMD memcpy. On a
                // 16-byte-aligned buffer the SCRATCH<-fb copy of
                // 115 KB drops to roughly 2/3 the wall time of
                // libc memcpy because the inner loop pumps 16
                // bytes per iteration via the AES3 store-pair
                // extension instead of 4.
                dsps_memcpy_aes3(dst, src,
                                 (size_t)n * n * sizeof(uint16_t));
            } else {
                const int total = n * n;
                for (int i = 0; i < total; ++i) {
                    copy_pixel<true>(src + i, dst + i, lut);
                }
            }
            break;
        }
        case Orient::ROT_90_CW: {
            for (int y = 0; y < n; ++y) {
                for (int x = 0; x < n; ++x) {
                    copy_pixel<WithLut>(src + y * n + x,
                                        dst + x * n + (n - 1 - y),
                                        lut);
                }
            }
            break;
        }
        case Orient::ROT_180: {
            const int total = n * n;
            for (int i = 0; i < total; ++i) {
                copy_pixel<WithLut>(src + total - 1 - i, dst + i, lut);
            }
            break;
        }
        case Orient::ROT_270_CW: {
            for (int y = 0; y < n; ++y) {
                for (int x = 0; x < n; ++x) {
                    copy_pixel<WithLut>(src + y * n + x,
                                        dst + (n - 1 - x) * n + y,
                                        lut);
                }
            }
            break;
        }
        }
    }

    // ---------------------------------------------------------------
    // CLAHE (Contrast Limited Adaptive Histogram Equalization)
    // ---------------------------------------------------------------
    //
    // The global percentile stretch we used to apply only fixes the
    // whole frame's luma distribution; it can't help a face that's
    // in shadow inside an otherwise bright scene (the global
    // histogram still looks healthy). CLAHE solves that: the image
    // is split into a grid of tiles, each tile gets its own
    // equalisation LUT built from its own histogram, and pixels in
    // between tiles get bilinearly-interpolated LUT outputs so
    // there are no visible tile boundaries. A per-tile clip-limit
    // caps how much any single histogram bin can contribute to its
    // CDF, which keeps the equalisation from amplifying sensor
    // noise in flat regions -- the exact failure mode that vanilla
    // HE has in low light.
    //
    // Layout / numbers tuned for the 240x240 detector input:
    //
    //   - 4 x 4 tile grid  -> 60 x 60 pixels per tile. Coarser than
    //     OpenCV's default 8x8 but cheap (12 KB of static state
    //     instead of 48 KB) and still adapts well to top-vs-bottom
    //     and left-vs-right lighting differences that matter to
    //     face detection.
    //   - clip limit       -> 3 % of tile pixel count (108 of 3600).
    //     Standard CLAHE value; tighter clipping starves contrast,
    //     looser clipping starts looking like vanilla HE again.
    //   - luma proxy       -> G6 expanded to 8 bits. Same proxy the
    //     percentile path used; cheap to extract from BE-packed
    //     RGB565.
    //   - apply            -> per-channel (R / G / B) using the
    //     SAME luma-derived LUT. Skipping the YCbCr round-trip
    //     keeps the inner loop tight; minor chroma shift is
    //     irrelevant to the face detector, and the live preview
    //     does not see this buffer.

    constexpr int CLAHE_N        = 4;                          // tile grid (N x N)
    constexpr int CLAHE_TILE_SZ  = FRAME_DIM / CLAHE_N;        // 60
    constexpr int CLAHE_TILE_PX  = CLAHE_TILE_SZ * CLAHE_TILE_SZ;  // 3600

    // ---------------------------------------------------------------
    // Motion pre-screen
    // ---------------------------------------------------------------
    //
    // Once the detector lock has fully expired (consecutive_misses
    // beyond OVERLAY_CLEAR_MISSES) the AI loop is just polling a
    // mostly-empty room. Running CLAHE + the detector + the padded
    // fallback on every frame in that state is ~80-160 ms of dead
    // work per cycle for zero benefit.
    //
    // Standard embedded-vision trick: take a coarse SAD between
    // consecutive frames and skip the heavy work when the scene
    // hasn't changed. We use an 8x8 sample grid (1/900 of pixels
    // touched, < 50 us per frame) and a generous SAD threshold so
    // sensor noise at high gain doesn't false-trigger.
    //
    // FORCE_INTERVAL guarantees we still run a real detect every N
    // quiet frames so a slowly-arriving face that stays under the
    // motion threshold doesn't go invisible forever.

    constexpr int MOTION_GRID                  = 8;
    constexpr int MOTION_STRIDE                = FRAME_DIM / MOTION_GRID;
    constexpr int MOTION_SAD_THRESHOLD         = 200;
    constexpr int MOTION_FORCE_INTERVAL_FRAMES = 50;

    struct MotionState {
        uint8_t samples[MOTION_GRID * MOTION_GRID] = {};
        bool    valid       = false;
        int     quiet_frames = 0;
    };

    __attribute__((hot))
    bool scene_is_static(const uint16_t *fb, MotionState &m) noexcept
    {
        uint8_t cur[MOTION_GRID * MOTION_GRID];
        const uint8_t *p8 = (const uint8_t *)fb;
        for (int j = 0; j < MOTION_GRID; ++j) {
            const int y = MOTION_STRIDE / 2 + j * MOTION_STRIDE;
            for (int i = 0; i < MOTION_GRID; ++i) {
                const int x = MOTION_STRIDE / 2 + i * MOTION_STRIDE;
                const int idx = (y * FRAME_DIM + x) * 2;
                const uint8_t hi = p8[idx];
                const uint8_t lo = p8[idx + 1];
                const uint8_t g6 =
                    (uint8_t)(((hi & 0x07) << 3) | (lo >> 5));
                cur[j * MOTION_GRID + i] =
                    (uint8_t)((g6 << 2) | (g6 >> 4));
            }
        }
        if (!m.valid) {
            std::memcpy(m.samples, cur, sizeof(cur));
            m.valid = true;
            m.quiet_frames = 0;
            return false;
        }
        int sad = 0;
        for (int i = 0; i < MOTION_GRID * MOTION_GRID; ++i) {
            sad += std::abs((int)cur[i] - (int)m.samples[i]);
        }
        std::memcpy(m.samples, cur, sizeof(cur));
        if (sad < MOTION_SAD_THRESHOLD) {
            ++m.quiet_frames;
            if (m.quiet_frames >= MOTION_FORCE_INTERVAL_FRAMES) {
                m.quiet_frames = 0;
                return false;
            }
            return true;
        }
        m.quiet_frames = 0;
        return false;
    }
    // CLAHE clip limit is driven by the adaptive tuning (see Tuning
    // / g_tuning further down). The compile-time CLAHE_TILE_PX is
    // still useful for the % math when building presets.

    // Latest 0..255 mean luma of the detector input, written by
    // apply_clahe() (the histogram pass it does covers every pixel,
    // so the mean is essentially free to compute). Read by the AI
    // loop's adaptive-AE check; relaxed ordering is fine because
    // both writer and reader are on the same task, and the only
    // cross-task reader (web UI heartbeat in the future) tolerates
    // a stale value.
    std::atomic<int> g_last_mean_luma{128};

    // ---------------------------------------------------------------
    // Adaptive AE bias
    // ---------------------------------------------------------------
    //
    // The sensor settings we apply at boot are tuned MID -- moderate
    // bias toward brightness, suitable for typical indoor light.
    // From there the AI loop measures the scene's mean luma once
    // every AE_CHECK_INTERVAL_MS and shifts the sensor toward DIM
    // (more boost) or BRIGHT (less boost) as needed. Three discrete
    // presets instead of a continuous mapping because each preset
    // change costs a few SCCB writes plus 100-200 ms of AE settle,
    // so we'd rather pick one of three good answers than chase a
    // moving target.
    //
    // Hysteresis bands are intentionally asymmetric: the threshold
    // to MOVE INTO a preset is further from the centre than the
    // threshold to LEAVE it, so a scene hovering at a boundary
    // doesn't flap between presets every AE check.

    enum class AEPreset { DIM, MID, BRIGHT };

    constexpr int      AE_CHECK_INTERVAL_MS = 3000;   // min seconds between transitions
    // Up-transitions (scene got brighter than current preset assumes):
    constexpr int      AE_DIM_TO_MID_LUMA   = 120;
    constexpr int      AE_MID_TO_BRIGHT_LUMA = 185;
    // Down-transitions (scene got dimmer than current preset assumes):
    constexpr int      AE_BRIGHT_TO_MID_LUMA = 150;
    constexpr int      AE_MID_TO_DIM_LUMA    = 85;

    // Matching / preprocessing parameters that move with the AE
    // preset. Same idea as apply_ae_preset() but for the AI side of
    // the pipeline rather than the sensor: each scene-luma band has
    // a different signal/noise profile, so the thresholds the
    // matcher uses should track that profile.
    //
    //   match_thr        cosine similarity floor for "this is a known
    //                    face". DIM: noise drags same-person sims
    //                    toward 0.30-0.45 so we have to bend lower
    //                    to keep recognising. BRIGHT: same-person
    //                    sims comfortably hit 0.6+, so a tighter
    //                    floor suppresses cross-IDs.
    //   enroll_debounce  frames-in-a-row of "unknown" required
    //                    before we commit a new enrolment. DIM
    //                    needs 2 because noisy single frames can
    //                    fake an unknown for a person we already
    //                    know; MID/BRIGHT can fire on a single
    //                    frame for snappy enrolment latency.
    //   clahe_clip_lim   per-tile histogram clip-limit for CLAHE.
    //                    DIM needs aggressive clipping so a flat
    //                    histogram still produces big output range;
    //                    BRIGHT needs gentle clipping so an
    //                    already-balanced scene doesn't get over-
    //                    contrasted into a posterised look.

    struct Tuning {
        float match_thr;
        int   enroll_debounce;
        int   clahe_clip_lim;   // absolute pixel count, not %
    };

    // Re-derive CLAHE clip absolutes from the tile size for clarity
    // (rather than baking constants). Tile pixel count is fixed at
    // compile time, so the % math collapses too.
    static constexpr int TILE_PX_HELPER = (FRAME_DIM / 4) * (FRAME_DIM / 4); // 3600

    constexpr Tuning TUNING_DIM = {
        /*match_thr*/       0.32f,
        /*enroll_debounce*/ 2,
        /*clahe_clip_lim*/  TILE_PX_HELPER * 7 / 100,   // 252 / 3600 = 7%
    };
    constexpr Tuning TUNING_MID = {
        0.38f,
        1,
        TILE_PX_HELPER * 5 / 100,                       // 180 / 3600 = 5%
    };
    constexpr Tuning TUNING_BRIGHT = {
        0.45f,
        1,
        TILE_PX_HELPER * 3 / 100,                       // 108 / 3600 = 3%
    };

    // Live tuning that the rest of the loop reads. Updated alongside
    // every AE preset transition. Initialised to MID because that's
    // what camera.cpp's boot defaults apply.
    Tuning g_tuning = TUNING_MID;

    constexpr const Tuning &tuning_for(AEPreset p) noexcept
    {
        switch (p) {
        case AEPreset::DIM:    return TUNING_DIM;
        case AEPreset::BRIGHT: return TUNING_BRIGHT;
        case AEPreset::MID:    break;
        }
        return TUNING_MID;
    }

    // Push the named preset to the OV2640. Idempotent at the SCCB
    // level: re-applying the same preset is a no-op as far as the
    // sensor is concerned, so the caller doesn't need to track
    // whether anything actually changed.
    void apply_ae_preset(AEPreset p)
    {
        // Update the app-side tuning first -- the sensor SCCB writes
        // below can take 100+ ms to settle but the matcher /
        // preprocessor parameters take effect immediately on the
        // next frame, which is what we want.
        g_tuning = tuning_for(p);

        sensor_t *s = esp_camera_sensor_get();
        if (!s) {
            return;
        }
        switch (p) {
        case AEPreset::DIM:
            // Maximum bias toward brightness for genuinely dim
            // indoor scenes / night-time. Highlights may clip; we
            // pay that cost for the face being detectable at all.
            s->set_gainceiling(s, GAINCEILING_128X);
            s->set_ae_level(s, 2);
            s->set_brightness(s, 2);
            break;
        case AEPreset::MID:
            // Moderate bias -- the boot default. Typical indoor
            // lighting with room ambient ~200-500 lux.
            s->set_gainceiling(s, GAINCEILING_32X);
            s->set_ae_level(s, 1);
            s->set_brightness(s, 1);
            break;
        case AEPreset::BRIGHT:
            // Neutral exposure with low gain headroom. Daylight /
            // strongly-lit indoor. Highlight retention matters more
            // here than shadow lift.
            s->set_gainceiling(s, GAINCEILING_4X);
            s->set_ae_level(s, 0);
            s->set_brightness(s, 0);
            break;
        }
    }

    // The CLAHE apply pass is the hottest 57,600-pixel inner loop
    // in the project. Marking it `hot` plus per-function `O3` lets
    // the compiler unroll the inner kernel more aggressively and
    // -- on this GCC version with the Xtensa LX7 backend -- emit
    // tighter pipelined sequences for the bilinear + clip math.
    __attribute__((hot, optimize("O3")))
    void apply_clahe(uint16_t *pixels) noexcept
    {
        // hist[t] : per-tile 256-bin luma histogram (uint16 is plenty
        //           since one tile holds CLAHE_TILE_PX = 3600 pixels).
        // lut[t]  : per-tile 256-entry 8 -> 8-bit equalisation LUT.
        // Both are `static` so they stay resident in internal SRAM
        // (random-access in the histogram pass benefits massively
        // from L1/IRAM speed vs PSRAM) without paying a
        // malloc/free every detect call.
        static uint16_t hist[CLAHE_N * CLAHE_N][256];
        static uint8_t  lut[CLAHE_N * CLAHE_N][256];

        // Precompute the bilinear interpolation tables along each
        // axis once -- `fx`/`fy` are constant given x/y, and the
        // tile-index pair (t0, t1) only changes at tile boundaries.
        // We also cache the simple `x / CLAHE_TILE_SZ` map used by
        // the histogram pass; Xtensa LX7 has no integer-divide
        // instruction so the compiler lowers /60 to a multiply-by-
        // reciprocal sequence, and a byte-wide table replaces ~3-5
        // cycles with a single load.
        // 240 * 4 * sizeof(int16_t) ~ 1.9 KB total -- trivial.
        static int16_t fx_tab[FRAME_DIM];   // 0..256
        static int16_t tx0_tab[FRAME_DIM];
        static int16_t tx1_tab[FRAME_DIM];
        static int8_t  tile_idx_tab[FRAME_DIM];

        auto build_axis_table = [](int16_t f_out[FRAME_DIM],
                                   int16_t t0_out[FRAME_DIM],
                                   int16_t t1_out[FRAME_DIM]) {
            for (int x = 0; x < FRAME_DIM; ++x) {
                const int xc = x - CLAHE_TILE_SZ / 2;
                int t0 = xc / CLAHE_TILE_SZ;
                int frac = xc - t0 * CLAHE_TILE_SZ;
                if (xc < 0) { t0 = 0; frac = 0; }
                int t1 = t0 + 1;
                if (t1 >= CLAHE_N) { t1 = CLAHE_N - 1; frac = 0; }
                f_out[x]  = (int16_t)((frac * 256) / CLAHE_TILE_SZ);
                t0_out[x] = (int16_t)t0;
                t1_out[x] = (int16_t)t1;
            }
        };
        static bool axis_tables_built = false;
        if (!axis_tables_built) {
            build_axis_table(fx_tab, tx0_tab, tx1_tab);
            for (int x = 0; x < FRAME_DIM; ++x) {
                tile_idx_tab[x] = (int8_t)(x / CLAHE_TILE_SZ);
            }
            axis_tables_built = true;
        }

        // ----- Pass 1: per-tile histograms -------------------------
        // dsps_memset_aes3 is the S3-tuned SIMD memset from ESP-DSP;
        // it clears 16 bytes per inner-loop iteration via the AES3
        // store extension, ~3-4x faster than libc memset on a 32 KB
        // buffer in internal SRAM.
        dsps_memset_aes3(hist, 0, sizeof(hist));
        const uint8_t *p8 = (const uint8_t *)pixels;
        for (int y = 0; y < FRAME_DIM; ++y) {
            const int ty = tile_idx_tab[y];           // y / CLAHE_TILE_SZ via table
            const int row_tile_base = ty * CLAHE_N;
            const uint8_t *row = p8 + (size_t)y * FRAME_DIM * 2;
            for (int x = 0; x < FRAME_DIM; ++x) {
                const int idx = x * 2;
                const uint8_t hi = row[idx];
                const uint8_t lo = row[idx + 1];
                const uint8_t g6 = (uint8_t)(((hi & 0x07) << 3) | (lo >> 5));
                const uint8_t y8 = (uint8_t)((g6 << 2) | (g6 >> 4));
                ++hist[row_tile_base + tile_idx_tab[x]][y8];
            }
        }

        // Frame-mean luma, derived for free from the histograms we
        // just built. The face_ai loop reads it to drive the
        // adaptive AE preset switch -- see apply_ae_preset() below.
        // Computed BEFORE the clip-limit pass scrambles the bin
        // counts; the redistributed histogram is no longer the
        // scene's actual luma distribution.
        {
            uint32_t total_sum = 0;
            for (int t = 0; t < CLAHE_N * CLAHE_N; ++t) {
                for (int v = 0; v < 256; ++v) {
                    total_sum += (uint32_t)hist[t][v] * (uint32_t)v;
                }
            }
            g_last_mean_luma.store(
                (int)(total_sum / ((uint32_t)FRAME_DIM * FRAME_DIM)),
                std::memory_order_relaxed);
        }

        // ----- Pass 2: per-tile clip-limit -> CDF -> LUT -----------
        const int clip_lim = g_tuning.clahe_clip_lim;
        for (int t = 0; t < CLAHE_N * CLAHE_N; ++t) {
            // Clip the histogram at the active clip-limit and tally
            // the excess that we lopped off. The clip-limit moves
            // with the AE preset so a dim scene gets aggressive
            // boost while a bright scene gets a gentle touch-up.
            uint32_t excess = 0;
            for (int v = 0; v < 256; ++v) {
                if (hist[t][v] > clip_lim) {
                    excess += hist[t][v] - clip_lim;
                    hist[t][v] = clip_lim;
                }
            }
            // Redistribute the excess uniformly across all 256 bins
            // (with the remainder spread across the lowest bins).
            // This keeps the total pixel count == CLAHE_TILE_PX so
            // the CDF -> LUT scaling stays normalised.
            const uint32_t bonus    = excess >> 8;
            const uint32_t leftover = excess & 0xFF;
            for (int v = 0; v < 256; ++v) {
                hist[t][v] += bonus + (v < (int)leftover ? 1 : 0);
            }
            // CDF, scaled into the LUT's 0..255 output range.
            uint32_t cdf = 0;
            for (int v = 0; v < 256; ++v) {
                cdf += hist[t][v];
                lut[t][v] = (uint8_t)((cdf * 255) / CLAHE_TILE_PX);
            }
        }

        // ----- Pass 3: apply with bilinear interpolation ----------
        //
        // OpenMV's CLAHE (which is Zuiderveld's textbook
        // implementation) equalizes the LUMA channel only and
        // reconstructs RGB from the equalized Y + the original U/V.
        // We do the same idea more cheaply: compute Y for each pixel,
        // bilinear-interp the LUT to get Y', then add `dY = Y' - Y`
        // to each of R/G/B. Same luma shift on every channel keeps
        // the chroma differences (R-G, G-B, R-B) identical to the
        // input, so face skin tones don't drift the way they did
        // under the previous per-channel CLAHE. Bonus: one bilinear
        // interpolation per pixel instead of three.
        //
        // Hot-loop micro-optimisations vs the textbook layout:
        //  * The four neighbouring LUT base pointers (l00..l11) only
        //    change at tile-column boundaries inside a row, which is
        //    just 4 times per 240-pixel row. The compiler can't see
        //    that on its own because they depend on tx0_tab[x] /
        //    tx1_tab[x] -- so we cache them and refresh only when
        //    tx0 actually changes.
        //  * Luma is `(r + 2*g + b) >> 2` instead of `(r+g+b+1)/3`.
        //    Avoids the multiply-by-reciprocal sequence the compiler
        //    lowers /3 to; the 0.25R + 0.5G + 0.25B weighting is
        //    close enough to BT.601 for our use that the detector
        //    can't tell the difference.
        uint8_t *w8 = (uint8_t *)pixels;
        for (int y = 0; y < FRAME_DIM; ++y) {
            // Per-row vertical tile indices and Q8 fraction.
            const int yc = y - CLAHE_TILE_SZ / 2;
            int ty0 = yc / CLAHE_TILE_SZ;
            int ty_frac = yc - ty0 * CLAHE_TILE_SZ;
            if (yc < 0) { ty0 = 0; ty_frac = 0; }
            int ty1 = ty0 + 1;
            if (ty1 >= CLAHE_N) { ty1 = CLAHE_N - 1; ty_frac = 0; }
            const int fy     = (ty_frac * 256) / CLAHE_TILE_SZ;
            const int fy_inv = 256 - fy;
            const int row_base = y * FRAME_DIM;
            const int row_ty0_base = ty0 * CLAHE_N;
            const int row_ty1_base = ty1 * CLAHE_N;

            // Cached LUT base pointers for the current tile column.
            // Initialised to "no column matched yet" so the first
            // iteration unconditionally fetches them.
            int last_tx0 = -1;
            const uint8_t *l00 = nullptr;
            const uint8_t *l01 = nullptr;
            const uint8_t *l10 = nullptr;
            const uint8_t *l11 = nullptr;

            for (int x = 0; x < FRAME_DIM; ++x) {
                const int tx0 = tx0_tab[x];
                if (tx0 != last_tx0) {
                    const int tx1 = tx1_tab[x];
                    l00 = lut[row_ty0_base + tx0];
                    l01 = lut[row_ty0_base + tx1];
                    l10 = lut[row_ty1_base + tx0];
                    l11 = lut[row_ty1_base + tx1];
                    last_tx0 = tx0;
                }
                const int fx  = fx_tab[x];
                const int fx_inv = 256 - fx;

                const int idx = (row_base + x) * 2;
                const uint8_t hi = w8[idx];
                const uint8_t lo = w8[idx + 1];
                const uint8_t r5 = (uint8_t)(hi >> 3);
                const uint8_t g6 = (uint8_t)(((hi & 0x07) << 3) | (lo >> 5));
                const uint8_t b5 = (uint8_t)(lo & 0x1F);
                const int r8e = (int)((r5 << 3) | (r5 >> 2));
                const int g8e = (int)((g6 << 2) | (g6 >> 4));
                const int b8e = (int)((b5 << 3) | (b5 >> 2));

                // Luma proxy: 0.25R + 0.5G + 0.25B via shifts.
                const int y_in = (r8e + g8e * 2 + b8e) >> 2;

                // Bilinear-interp the LUT for Y across the four
                // surrounding tiles.
                const int v00 = l00[y_in];
                const int v01 = l01[y_in];
                const int v10 = l10[y_in];
                const int v11 = l11[y_in];
                const int top = v00 * fx_inv + v01 * fx;
                const int bot = v10 * fx_inv + v11 * fx;
                const int y_out = (top * fy_inv + bot * fy) >> 16;

                // Apply the luma delta to each channel; clip to
                // 0..255. Same delta on every channel preserves
                // chroma.
                const int dy = y_out - y_in;
                int nr = r8e + dy;
                int ng = g8e + dy;
                int nb = b8e + dy;
                if (nr < 0) nr = 0; else if (nr > 255) nr = 255;
                if (ng < 0) ng = 0; else if (ng > 255) ng = 255;
                if (nb < 0) nb = 0; else if (nb > 255) nb = 255;

                const uint8_t nr5 = (uint8_t)(nr >> 3);
                const uint8_t ng6 = (uint8_t)(ng >> 2);
                const uint8_t nb5 = (uint8_t)(nb >> 3);
                w8[idx]     = (uint8_t)((nr5 << 3) | (ng6 >> 3));
                w8[idx + 1] = (uint8_t)((ng6 << 5) | nb5);
            }
        }
    }

    // Prepare the buffer the detector will read this frame. Returns
    // the pointer to feed into HumanFaceDetect::run(). May return
    // `src` (zero-copy fast path) or `scratch` (after a rotate /
    // CLAHE / both).
    const uint16_t *prep_detect_input(Orient o,
                                      const uint16_t *src,
                                      uint16_t *scratch,
                                      int n) noexcept
    {
        // Always apply CLAHE to the detector input. The previous
        // "only when the histogram looks narrow" heuristic was wrong
        // for the low-light case it was supposed to help: high
        // sensor gain produces frames whose noise floor alone gives
        // a wide histogram range, the heuristic concludes the frame
        // is fine, CLAHE is skipped, and the real (low-contrast)
        // signal under the noise stays unrecoverable. Paying ~5 ms
        // of CLAHE on every detection frame is a fair price for
        // detection working at all in dim scenes.
        rotate_to_scratch<false>(o, src, scratch, n, nullptr);
        apply_clahe(scratch);
        return scratch;
    }

    // --- Close-range padded fallback -------------------------------
    //
    // ESP-WHO's MSR+MNP face detector is trained on faces that occupy
    // roughly 5-60 % of the image. When the user is very close to the
    // OV2640 the face fills almost the whole 240x240 frame, which
    // pushes the detector well outside its training distribution and
    // causes it to silently fail. The primary detection pass catches
    // every realistic mid/far-range case; this fallback exists for
    // when that pass returns nothing despite the user clearly being
    // there.
    //
    // The recipe is dead simple: resample the prepared detector
    // buffer down into a centred INNER x INNER box of an otherwise
    // mid-grey 240x240 buffer. A close-up face that was filling ~95 %
    // of the frame now occupies INNER/FRAME_DIM of it, which lands
    // back in the model's happy range. Bounding boxes / keypoints
    // come out in padded coords and have to be remapped before the
    // rest of the pipeline can use them.
    //
    // Coordinate remap: padded coord  xp in [BORDER, BORDER+INNER) maps
    // to detection-frame  x = (xp - BORDER) * FRAME_DIM / INNER.
    // Clamped to [0, FRAME_DIM-1] to keep downstream code safe.

    constexpr int PADDED_INNER  = 160;
    constexpr int PADDED_BORDER = (FRAME_DIM - PADDED_INNER) / 2;

    void shrink_into_padded(const uint16_t *src, uint16_t *dst) noexcept
    {
        // RGB565BE for mid-grey: R5=15, G6=31, B5=15 -> 0x7BEF native,
        // bytes swap to 0xEF7B. Fill 32 bits at a time (two pixels per
        // write) -- on a word-aligned PSRAM buffer this is roughly
        // 2x the throughput of the per-pixel store. FRAME_DIM*FRAME_DIM
        // is 57600 which is even, so we never write past the end.
        const uint16_t grey_be = (uint16_t)(((0x7BEFu >> 8) & 0xFF) |
                                            ((0x7BEFu & 0xFF) << 8));
        const uint32_t grey_be_pair =
            ((uint32_t)grey_be << 16) | (uint32_t)grey_be;
        const size_t total_pairs = (size_t)FRAME_DIM * FRAME_DIM / 2;
        uint32_t *dst32 = reinterpret_cast<uint32_t *>(dst);
        for (size_t i = 0; i < total_pairs; ++i) {
            dst32[i] = grey_be_pair;
        }
        // Nearest-neighbour resample src (FRAME_DIM x FRAME_DIM) into
        // the centred INNER x INNER region.
        for (int y = 0; y < PADDED_INNER; ++y) {
            const int sy = (y * FRAME_DIM) / PADDED_INNER;
            const uint16_t *srow = src + (size_t)sy * FRAME_DIM;
            uint16_t       *drow = dst +
                (size_t)(PADDED_BORDER + y) * FRAME_DIM + PADDED_BORDER;
            for (int x = 0; x < PADDED_INNER; ++x) {
                const int sx = (x * FRAME_DIM) / PADDED_INNER;
                drow[x] = srow[sx];
            }
        }
    }

    // Apply the padded -> detection-frame coord remap in place over a
    // bbox / keypoint list. Operates on raw ints to match the
    // dl::detect::result_t storage layout.
    int remap_padded_axis(int v) noexcept
    {
        // Clip to inner box (gray border can't legitimately host
        // detections, but the postprocessor's NMS sometimes spits out
        // boxes that slightly overhang the grey).
        if (v < PADDED_BORDER) v = PADDED_BORDER;
        if (v > PADDED_BORDER + PADDED_INNER - 1) {
            v = PADDED_BORDER + PADDED_INNER - 1;
        }
        return ((v - PADDED_BORDER) * FRAME_DIM) / PADDED_INNER;
    }

    void remap_padded_result(dl::detect::result_t *r) noexcept
    {
        r->box[0] = remap_padded_axis(r->box[0]);
        r->box[1] = remap_padded_axis(r->box[1]);
        r->box[2] = remap_padded_axis(r->box[2]);
        r->box[3] = remap_padded_axis(r->box[3]);
        for (size_t i = 0; i + 1 < r->keypoint.size(); i += 2) {
            r->keypoint[i]     = remap_padded_axis(r->keypoint[i]);
            r->keypoint[i + 1] = remap_padded_axis(r->keypoint[i + 1]);
        }
    }

    // Banner rotation = how the on-screen face is tilted relative to upright
    // in the *un-rotated* camera frame (which is what the user sees on the
    // LCD). If we rotated the image by N*90 deg CW to make the face upright
    // for the detector, then in the original frame the face was tilted by
    // the SAME amount CCW, so the banner rotates that much CCW. The sign
    // below was chosen empirically against the previous keypoint-based
    // calibration; if upright text comes out rotated the wrong way, flip
    // the sign.
    float banner_angle_for(Orient o)
    {
        constexpr float quarter = 3.14159265358979323846f * 0.5f;
        return -(float)((int)o) * quarter;
    }

    // Refines banner_angle_for() using the actual head tilt measured from
    // the detector's eye + nose keypoints. The MNP postprocessor emits 5
    // keypoints in detection-frame coords as a flat [x0,y0, x1,y1, ...]
    // vector; indices 0..1 and 2..3 are the two eyes, 4..5 is the nose.
    //
    // We unrotate the keypoints back into the *display* camera frame (the
    // un-rotated frame the LCD shows), compute the head's local "down"
    // direction (eye-midpoint -> nose), derive the banner's text-right
    // vector from it, and snap to the nearest quarter turn. Using the
    // eye+nose triangle removes the image-left vs subject-left eye-labelling
    // ambiguity entirely: "down" always points from forehead to chin, so
    // the orientation we recover is unambiguous up to noise. The discrete
    // orientation cycling that gates the detector is no longer used for
    // banner placement; the banner snaps to whichever cardinal the head
    // actually points to.
    //
    // Returns the angle in radians that banner_render() expects (positive =
    // CW in y-down image coords).
    void unrotate_point(Orient o, int n, int &x, int &y)
    {
        const int xd = x, yd = y;
        switch (o)
        {
        case Orient::ROT_0:
            x = xd;
            y = yd;
            break;
        case Orient::ROT_90_CW:
            // rotate_90cw: dst[x*n + (n-1-y)] = src[y*n + x]
            // Source (xs, ys) lands at (n-1-ys, xs) in the detector frame.
            x = yd;
            y = n - 1 - xd;
            break;
        case Orient::ROT_180:
            x = n - 1 - xd;
            y = n - 1 - yd;
            break;
        case Orient::ROT_270_CW:
            // rotate_270cw: dst[(n-1-x)*n + y] = src[y*n + x]
            // Source (xs, ys) lands at (ys, n-1-xs) in the detector frame.
            x = n - 1 - yd;
            y = xd;
            break;
        }
    }

    // Reject detections whose keypoint geometry doesn't match an
    // upright face in the DETECTOR INPUT frame. The primary failure
    // mode this guards against is the detector firing on an upside-
    // down face (faces have rough top/bottom symmetry: forehead vs
    // chin, eye sockets vs nostrils). The previous "nose below
    // eye-midline" check wasn't enough because when the detector
    // does fire on a flipped face it labels keypoints CORRECTLY for
    // what it sees, which means the nose lands above the eye
    // midline in detector coords and the check is supposed to fail
    // -- but a near-coincident keypoint set can still pass it by a
    // single pixel.
    //
    // Three independent properties must all hold for an upright
    // face. An upside-down or sideways detection can't fake all
    // three simultaneously, even with worst-case keypoint noise.
    //
    //   1. Eyes must be more horizontal than vertical, with a
    //      non-trivial separation. Rejects sideways faces and
    //      degenerate "all keypoints at the centre" results.
    //   2. Nose y must be at least eye_dx / 8 below the eye
    //      midline. A flipped face has the nose-keypoint above the
    //      eye midline (nose_drop < 0) and an ambiguous near-zero
    //      drop is also rejected by the explicit margin.
    //   3. When mouth keypoints are present (always, for MNP) the
    //      mouth midline must be below the nose. Catches the rare
    //      cases where the detector mis-labels enough points to
    //      slip past (2) but the mouth-vs-nose ordering still
    //      reveals the flip.
    bool keypoints_look_upright(const dl::detect::result_t *det)
    {
        if (!det || det->keypoint.size() < 6) {
            // No keypoints to validate — trust the bbox.
            return true;
        }

        const int e0x = det->keypoint[0];
        const int e0y = det->keypoint[1];
        const int e1x = det->keypoint[2];
        const int e1y = det->keypoint[3];
        const int ny  = det->keypoint[5];

        const int eye_mid_y = (e0y + e1y) / 2;
        const int eye_dx    = std::abs(e1x - e0x);

        // Reject degenerate keypoint sets (eyes coincident or
        // implausibly close). This catches the original false-
        // positive class where the detector lit up on a near-
        // symmetric blob and emitted near-identical keypoints --
        // including the upside-down-faces-look-face-like case --
        // because in those failures eye_dx collapses toward zero.
        if (eye_dx < 4) {
            return false;
        }

        // The one check that an upside-down real face cannot fake:
        // the detector's predicted nose must sit below the predicted
        // eye midline. Real upside-down faces have nose ABOVE the
        // eye midline in detector input (the predicted keypoints
        // come out CORRECT for what the model sees, which is flipped).
        // The margin >= eye_dx / 8 protects against near-coincident
        // keypoints sneaking past on a single-pixel noise.
        //
        // Previous iterations of this function also enforced a
        // "eyes more horizontal than vertical" check and a "mouth
        // below nose" check; both rejected too many real upright
        // detections in low light because high-gain keypoint noise
        // and the natural ~45 deg tilt the orient cycle leaves on
        // the table consistently tripped them. Letting those go is
        // safe -- the nose-below-eyes margin alone is sufficient
        // for the upside-down case, which was the original concern.
        const int nose_drop = ny - eye_mid_y;
        if (nose_drop * 8 < eye_dx) {
            return false;
        }

        return true;
    }

    // IoU between two integer axis-aligned bboxes, returned as a
    // 0..100 percentage. Used by the recognition cache to decide
    // whether the latest detection "continues" the previously
    // identified face.
    int iou_pct(const int *a, const int *b) noexcept
    {
        const int ix0 = std::max(a[0], b[0]);
        const int iy0 = std::max(a[1], b[1]);
        const int ix1 = std::min(a[2], b[2]);
        const int iy1 = std::min(a[3], b[3]);
        if (ix1 <= ix0 || iy1 <= iy0) {
            return 0;
        }
        const int inter = (ix1 - ix0) * (iy1 - iy0);
        const int aa    = (a[2] - a[0]) * (a[3] - a[1]);
        const int bb    = (b[2] - b[0]) * (b[3] - b[1]);
        const int uni   = aa + bb - inter;
        return uni > 0 ? (inter * 100) / uni : 0;
    }

    // Crop a square region centred on the detected face from the raw
    // camera fb (display-frame coords) and downsample it to the fixed
    // thumbnail dimensions consumed by the web UI. The crop is widened
    // ~20 % over the bbox so the saved thumb actually looks like the
    // person (forehead + chin context) rather than a tight portrait
    // mask. Nearest-neighbour resample is fine here — the thumb is
    // 64x64 from a 240x240 source, and the image is consumed by
    // human glance recognition, not pixel-precision tooling.
    void crop_face_thumb(const uint16_t *fb_pixels,
                         const dl::detect::result_t *det,
                         Orient locked_orient,
                         std::vector<uint16_t> *out_thumb) noexcept
    {
        int x0 = det->box[0], y0 = det->box[1];
        int x1 = det->box[2], y1 = det->box[3];
        unrotate_point(locked_orient, FRAME_DIM, x0, y0);
        unrotate_point(locked_orient, FRAME_DIM, x1, y1);
        const int xmin = std::min(x0, x1);
        const int xmax = std::max(x0, x1);
        const int ymin = std::min(y0, y1);
        const int ymax = std::max(y0, y1);

        const int bw = xmax - xmin;
        const int bh = ymax - ymin;
        const int cx = (xmin + xmax) / 2;
        const int cy = (ymin + ymax) / 2;
        int       side = std::max(bw, bh);
        side = side + side / 5;        // +20 % forehead / chin context
        const int half = side / 2;

        int cx0 = std::max(0, cx - half);
        int cy0 = std::max(0, cy - half);
        int cx1 = std::min(FRAME_DIM, cx + half);
        int cy1 = std::min(FRAME_DIM, cy + half);
        // Re-square after edge-clipping so the resample stays
        // isotropic; bias toward keeping the bbox visible by anchoring
        // on (cx0, cy0).
        const int s = std::min(cx1 - cx0, cy1 - cy0);
        cx1 = cx0 + s;
        cy1 = cy0 + s;
        if (s <= 0) {
            out_thumb->clear();
            return;
        }

        out_thumb->resize((size_t)FACE_THUMB_DIM * FACE_THUMB_DIM);
        uint16_t *dst = out_thumb->data();
        for (int ty = 0; ty < FACE_THUMB_DIM; ++ty) {
            const int sy = cy0 + (ty * s) / FACE_THUMB_DIM;
            const uint16_t *srow = fb_pixels + (size_t)sy * FRAME_DIM;
            for (int tx = 0; tx < FACE_THUMB_DIM; ++tx) {
                const int sx = cx0 + (tx * s) / FACE_THUMB_DIM;
                dst[ty * FACE_THUMB_DIM + tx] = srow[sx];
            }
        }
    }

    float head_roll_for(const dl::detect::result_t *det, Orient o)
    {
        if (!det || det->keypoint.size() < 6)
        {
            return banner_angle_for(o); // missing keypoints -> fall back
        }

        // Eye midpoint + nose in detection frame, then unrotated into the
        // display frame (= the LCD's view of the world).
        int e0x = det->keypoint[0], e0y = det->keypoint[1];
        int e1x = det->keypoint[2], e1y = det->keypoint[3];
        int nx = det->keypoint[4], ny = det->keypoint[5];
        unrotate_point(o, FRAME_DIM, e0x, e0y);
        unrotate_point(o, FRAME_DIM, e1x, e1y);
        unrotate_point(o, FRAME_DIM, nx, ny);

        // Head-local "down" direction in display-frame coords. For an upright
        // forward-facing face this is (0, +1) (nose below eye midpoint).
        const float dx = (float)(nx - (e0x + e1x) / 2);
        const float dy = (float)(ny - (e0y + e1y) / 2);
        if (dx == 0.0f && dy == 0.0f)
        {
            return banner_angle_for(o);
        }

        // banner_render rotates the text's local +x axis to (cos a, sin a) in
        // image coords. The banner's "text-right" should be perpendicular to
        // and 90 deg CCW (visually) from "down": text_right = (down_y, -down_x).
        // So a = atan2(text_right.y, text_right.x) = atan2(-down_x, down_y).
        constexpr float pi = 3.14159265358979323846f;
        constexpr float quarter = pi * 0.5f;
        // The +quarter offset is an empirically-tuned correction: the
        // un-rotation + atan2 derivation alone leaves the banner 90 deg
        // CCW of the head, so we rotate it one quarter-turn CW.
        const float a = atan2f(-dx, dy) + quarter;

        // Snap to nearest 90 deg bucket. Head tilts between buckets round
        // to whichever cardinal is closest.
        return roundf(a / quarter) * quarter;
    }

    void face_ai_task(void *)
    {
        ESP_LOGI(TAG, "task running on core %d, prio %d",
                 xPortGetCoreID(), (int)TASK_PRIO);

        // Construction loads the .espdl files out of the model partitions.
        // Costs ~1 second up-front.
        auto *detect = new HumanFaceDetect();
        // We previously bumped the MSR (proposal) stage's score floor to
        // 0.65 to speed up MNP refinement. That helped throughput at
        // mid range but starved real close-range detections whose MSR
        // confidence is naturally lower (face out of training
        // distribution). The default 0.5 was the right tradeoff; we now
        // get the same false-positive resistance from MIN_DETECT_SCORE
        // + keypoints_look_upright at the app level.
        auto *feat = new HumanFaceFeat();
        const int feat_len = feat->get_feat_len();

        // Scratch buffer for orientation-rotated frames. Prefer
        // internal SRAM (~3-4x faster than octal PSRAM at 80 MHz) so
        // the detector's input read traffic doesn't hit the slow bus.
        // ~115 KB is a big chunk of the S3's 320 KB internal heap but
        // it fits comfortably alongside the existing IDF + camera
        // stacks; if for some reason it doesn't, we transparently
        // fall back to PSRAM and lose only the bandwidth bonus.
        const size_t scratch_bytes = (size_t)FRAME_DIM * FRAME_DIM * sizeof(uint16_t);
        auto *scratch = (uint16_t *)heap_caps_aligned_alloc(
            16, scratch_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (scratch) {
            ESP_LOGI(TAG, "orientation scratch in internal SRAM (%u B)",
                     (unsigned)scratch_bytes);
        } else {
            scratch = (uint16_t *)heap_caps_aligned_alloc(
                16, scratch_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (scratch) {
                ESP_LOGW(TAG,
                         "internal SRAM full; orientation scratch in PSRAM (%u B)",
                         (unsigned)scratch_bytes);
            }
        }
        if (!scratch)
        {
            ESP_LOGE(TAG, "failed to alloc %u-byte orientation scratch",
                     (unsigned)scratch_bytes);
            vTaskDelete(nullptr);
            return;
        }

        // Secondary scratch for the close-range padded fallback. Only
        // used when the primary detect misses inside the sticky
        // window, so PSRAM (slower than internal SRAM but plentiful)
        // is fine -- the bandwidth cost is only paid on the
        // exception path.
        auto *padded_scratch = (uint16_t *)heap_caps_aligned_alloc(
            16, scratch_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!padded_scratch) {
            ESP_LOGW(TAG,
                     "no PSRAM for padded fallback (%u B); close-range "
                     "detection may suffer",
                     (unsigned)scratch_bytes);
        }

        // The known-face DB lives at module scope (g_known_faces /
        // g_db_mux) so the web server can read it from another task.
        // The task acquires g_db_mux for every read or write below.
        // Pre-reserve so the matcher's pointer to the embedding stays
        // valid for the lifetime of the entry.
        {
            xSemaphoreTake(g_db_mux, portMAX_DELAY);
            if (g_known_faces.capacity() < MAX_KNOWN_FACES) {
                g_known_faces.reserve(MAX_KNOWN_FACES);
            }
            xSemaphoreGive(g_db_mux);
        }

        // Counters for the per-second heartbeat.
        int64_t stats_next_us = esp_timer_get_time() + STATS_PERIOD_US;

        // Adaptive AE state. Initialised to MID because camera.cpp
        // applied the MID preset at boot. The face_ai loop checks
        // mean luma every AE_CHECK_INTERVAL_MS and shifts presets
        // up or down with hysteresis (see apply_ae_preset()).
        AEPreset  ae_preset       = AEPreset::MID;
        int64_t   ae_next_check_us =
            esp_timer_get_time() + (int64_t)AE_CHECK_INTERVAL_MS * 1000;

        // Motion pre-screen state. Initialised invalid; first
        // qualifying frame seeds the comparison and from there the
        // SAD vs previous tells us if the scene is unchanged.
        MotionState motion;
        uint32_t s_frames = 0;        // frames pulled from camera
        uint32_t s_raw_dets = 0;      // detector returned >=1 result (any score, any orient)
        uint32_t s_pad_hits = 0;      // padded-fallback rescued a frame
        uint32_t s_motion_skip = 0;   // skipped detection due to no scene change
        uint32_t s_detections = 0;    // kept after every gate (size/focus/score/upright)
        uint32_t s_rej_score = 0;     // biggest rejected by MIN_DETECT_SCORE
        uint32_t s_rej_geom = 0;      // biggest rejected by keypoints_look_upright
        uint32_t s_too_small = 0;     // biggest rejected for bbox size
        uint32_t s_too_blurry = 0;    // biggest rejected for focus
        uint32_t s_recognised = 0;    // matched an existing enrollment
        uint32_t s_enrolled = 0;      // new face added
        // Running best-score totals so the heartbeat can show
        // average detector confidence even when our gates reject.
        float    s_score_sum = 0.f;
        uint32_t s_score_n   = 0;

        // Debounce: consecutive frames so far that look like a new face.
        int unknown_streak = 0;

        // Recognition cache. While the same face stays in roughly
        // the same place we treat consecutive detections as the same
        // identity and skip the embedder + cosine sweep. See
        // RECOG_CACHE_MS / RECOG_IOU_PCT.
        struct {
            bool     valid       = false;
            int      box[4]      = {0, 0, 0, 0};   // detection-frame coords
            Orient   orient      = Orient::ROT_0;  // bbox is only comparable in same orient
            uint32_t stamp_ms    = 0;
        } recog_cache;

        // Orientation to TRY on this frame. After a successful detection we
    // leave this alone so the next frame uses the same orientation; on a
    // failed detection we may stick on it for a couple more attempts
    // (orient-sticky retry, see ORIENT_STICKY_MISSES below) before
    // advancing to the next of the four.
    Orient try_orient = Orient::ROT_0;

    // Consecutive miss frames in the current detection-loss streak. Used
    // by the orient-sticky retry: a quick blink / occlusion / motion blur
    // is FAR more likely than the user actually rotating the device
    // between frames, so we retry the last-known-good orient before
    // burning frames on the other three buckets. Reset on each success.
    int consecutive_misses = 0;
    // Number of misses we'll stay on the known-good orient before
    // cycling. ORIENT_STICKY_MISSES=2 keeps us responsive to genuine
    // rotation (worst case ~3-4 detection frames to discover the new
    // orient) while making transient blinks effectively free.
    constexpr int ORIENT_STICKY_MISSES = 4;

    // Number of consecutive misses we'll tolerate before treating the
    // face as actually gone and hiding the on-screen overlay.
    // Decoupled from ORIENT_STICKY_MISSES on purpose: we want
    // orientation cycling to react quickly (after just a couple of
    // misses) but we DON'T want the overlay to blink off and back on
    // every time the detector skips a frame on a blink, motion blur,
    // or pose change. At ~10 detections/sec this gives the user about
    // 0.8 s of grace before the box disappears, which matches the
    // perceived "they're still here" window without leaving stale
    // boxes around after they actually walk away.
    constexpr int OVERLAY_CLEAR_MISSES = 12;

        ESP_LOGI(TAG, "ESP-WHO ready, feat_len=%d, entering scan loop", feat_len);

        while (true)
        {
            // Heartbeat first - placed before every early-continue so we get
            // a per-second log even when the loop is degenerate (fb_get
            // timing out, banner pinning us, model errors). Silence here
            // means the task is wedged, not just unlucky.
            const int64_t now_us = esp_timer_get_time();

            // Adaptive AE bias. Run on the same coarse cadence as the
            // stats heartbeat; the mean-luma value driving the decision
            // is refreshed every detection frame inside apply_clahe(),
            // so this check is cheap (one atomic load + a few
            // comparisons). The asymmetric hysteresis bands stop us
            // from oscillating between presets when a scene's luma
            // sits near a boundary.
            if (now_us >= ae_next_check_us)
            {
                const int luma = g_last_mean_luma.load(
                    std::memory_order_relaxed);
                AEPreset next = ae_preset;
                switch (ae_preset) {
                case AEPreset::DIM:
                    if (luma > AE_DIM_TO_MID_LUMA)    next = AEPreset::MID;
                    break;
                case AEPreset::MID:
                    if      (luma > AE_MID_TO_BRIGHT_LUMA) next = AEPreset::BRIGHT;
                    else if (luma < AE_MID_TO_DIM_LUMA)    next = AEPreset::DIM;
                    break;
                case AEPreset::BRIGHT:
                    if (luma < AE_BRIGHT_TO_MID_LUMA) next = AEPreset::MID;
                    break;
                }
                if (next != ae_preset) {
                    const char *names[] = { "DIM", "MID", "BRIGHT" };
                    ESP_LOGI(TAG,
                             "AE preset %s -> %s (mean luma %d)",
                             names[(int)ae_preset], names[(int)next],
                             luma);
                    apply_ae_preset(next);
                    ae_preset = next;
                }
                ae_next_check_us = now_us +
                    (int64_t)AE_CHECK_INTERVAL_MS * 1000;
            }

            if (now_us >= stats_next_us)
            {
                const float avg_score = s_score_n ?
                    (s_score_sum / (float)s_score_n) : 0.f;
                ESP_LOGI(TAG,
                         "1s: frame=%u skip=%u raw=%u pad=%u kept=%u score_avg=%.2f "
                         "| rej:score=%u geom=%u small=%u blurry=%u "
                         "| known=%u new=%u (db=%u o=%d)",
                         (unsigned)s_frames,
                         (unsigned)s_motion_skip,
                         (unsigned)s_raw_dets,
                         (unsigned)s_pad_hits,
                         (unsigned)s_detections,
                         (double)avg_score,
                         (unsigned)s_rej_score,
                         (unsigned)s_rej_geom,
                         (unsigned)s_too_small,
                         (unsigned)s_too_blurry,
                         (unsigned)s_recognised, (unsigned)s_enrolled,
                         (unsigned)g_known_faces.size(),
                         (int)try_orient);
                s_frames = s_motion_skip = s_raw_dets = s_pad_hits = s_detections =
                    s_rej_score = s_rej_geom = s_too_small = s_too_blurry =
                    s_recognised = s_enrolled = 0;
                s_score_sum = 0.f;
                s_score_n = 0;
                stats_next_us = now_us + STATS_PERIOD_US;
            }

            // The banner is now an overlay composited into the live
            // preview by the render task, so we don't pause inference
            // while it's up — the detector keeps running, and any new
            // enrolment will simply refresh the overlay's deadline /
            // angle.

            camera_fb_t *fb = esp_camera_fb_get();
            if (!fb)
            {
                vTaskDelay(pdMS_TO_TICKS(20));
                continue;
            }
            ++s_frames;

            // Motion pre-screen: once the face has clearly left the
            // frame (consecutive_misses past the overlay-clear
            // threshold) and the coarse-luma SAD says the scene
            // hasn't changed, skip the entire detection pipeline.
            // Saves ~80-160 ms of dead work per cycle while the
            // room is empty; cost is < 50 us per frame on the
            // motion check itself.
            if (consecutive_misses >= OVERLAY_CLEAR_MISSES &&
                fb->width == FRAME_DIM && fb->height == FRAME_DIM &&
                scene_is_static((const uint16_t *)fb->buf, motion))
            {
                ++s_motion_skip;
                esp_camera_fb_return(fb);
                vTaskDelay(pdMS_TO_TICKS(NO_FACE_DELAY_MS));
                continue;
            }

            // Pre-process the camera fb into something the detector
            // will be happy with: optional 90/180/270 rotation so the
            // face is upright in the input, and an optional luma
            // contrast stretch when the OV2640's AE has clipped the
            // frame into a narrow band. Both are folded into at most
            // one full pass over the buffer (see prep_detect_input);
            // the well-exposed ROT_0 fast path skips the copy entirely
            // and points the detector at fb->buf directly.
            //
            // We deliberately do NOT mutate fb->buf: the render task
            // is reading it on the other core for the live preview,
            // and the user expects the preview to look like the raw
            // sensor output, not the contrast-stretched detector view.
            const uint16_t *src = (const uint16_t *)fb->buf;
            const uint16_t *to_detect = src;
            if (fb->width == FRAME_DIM && fb->height == FRAME_DIM)
            {
                to_detect = prep_detect_input(try_orient, src, scratch,
                                              FRAME_DIM);
            }

            dl::image::img_t img = {};
            img.data = (void *)to_detect;
            img.width = fb->width;
            img.height = fb->height;
            // Camera outputs RGB565 high-byte-first in memory; ESP-DL's
            // preprocessor consumes that format directly.
            img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE;

            std::list<dl::detect::result_t> &detections = detect->run(img);
            if (!detections.empty())
            {
                ++s_raw_dets;
            }

            // Act only on the largest-area detection. Avoids cross-enrolling
            // two people in frame and biases toward whoever is closest to
            // the camera.
            const dl::detect::result_t *biggest = nullptr;
            int best_area = 0;
            for (const auto &d : detections)
            {
                const int a = d.box_area();
                if (a > best_area)
                {
                    best_area = a;
                    biggest = &d;
                }
            }
            if (biggest) {
                s_score_sum += biggest->score;
                ++s_score_n;
            }

            // Close-range padded fallback. When the primary pass
            // misses, AND we either still have an active orient lock
            // (sticky window) or we're locked to ROT_0 on cold start,
            // try resampling the detector input into a centred 160x160
            // box of an otherwise mid-grey 240x240 buffer. That brings
            // a "face fills the frame" close-up back into the model's
            // training distribution. If THIS pass returns a usable
            // detection we remap its coords back into the regular
            // detection frame and let the rest of the pipeline treat
            // it like a normal hit.
            dl::detect::result_t padded_remap;
            // Padded retry pays ~80 ms per call. We want that cost
            // when the user is in frame and the close-range model-
            // distribution mismatch is the only thing keeping the
            // primary pass from finding them, but NOT during cold-
            // start orient discovery where the face is genuinely
            // absent on three of the four orients. Heuristic: only
            // run the retry while we still have a "recent lock"
            // (consecutive_misses < OVERLAY_CLEAR_MISSES), which is
            // the same window the on-screen HUD treats as "the
            // face is still here, just glitching". Past that
            // window we accept that the face has actually left and
            // cycle orients at the cheap primary-pass rate until
            // someone re-appears.
            const bool padded_eligible =
                padded_scratch &&
                fb->width == FRAME_DIM && fb->height == FRAME_DIM &&
                consecutive_misses < OVERLAY_CLEAR_MISSES;
            if (!biggest && padded_eligible)
            {
                shrink_into_padded(to_detect, padded_scratch);

                dl::image::img_t pimg = {};
                pimg.data     = padded_scratch;
                pimg.width    = FRAME_DIM;
                pimg.height   = FRAME_DIM;
                pimg.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE;

                auto &pdetections = detect->run(pimg);
                const dl::detect::result_t *pbest = nullptr;
                int pbest_area = 0;
                for (const auto &d : pdetections) {
                    const int a = d.box_area();
                    if (a > pbest_area) {
                        pbest_area = a;
                        pbest = &d;
                    }
                }
                if (pbest && pbest->score >= MIN_DETECT_SCORE) {
                    // Deep-copy into a local result we own, since the
                    // detector reuses its internal list on the next
                    // call.
                    padded_remap.category = pbest->category;
                    padded_remap.score    = pbest->score;
                    padded_remap.box      = pbest->box;
                    padded_remap.keypoint = pbest->keypoint;
                    remap_padded_result(&padded_remap);
                    // The upright check is performed in detection-
                    // frame coords, which after remap are equivalent
                    // to the primary path's coords -- safe to re-run.
                    if (keypoints_look_upright(&padded_remap)) {
                        ESP_LOGI(TAG,
                                 "padded fallback hit: score=%.2f "
                                 "(close-range face)",
                                 (double)padded_remap.score);
                        biggest = &padded_remap;
                        s_score_sum += padded_remap.score;
                        ++s_score_n;
                        ++s_pad_hits;
                    }
                }
            }

            if (!biggest ||
                biggest->score < MIN_DETECT_SCORE ||
                !keypoints_look_upright(biggest))
            {
                if (biggest)
                {
                    if (biggest->score < MIN_DETECT_SCORE) {
                        // Low-confidence detection: don't bother
                        // running the embedder on it, don't update
                        // the HUD with a flaky bbox, and don't lock
                        // the orient cycle onto whatever it landed on.
                        ESP_LOGD(TAG,
                                 "drop low-score det: score=%.2f < %.2f (orient=%d)",
                                 (double)biggest->score,
                                 (double)MIN_DETECT_SCORE,
                                 (int)try_orient);
                        ++s_rej_score;
                    } else {
                        // Detector fired on a face whose keypoint
                        // geometry doesn't match an upright face --
                        // most often this is the upside-down /
                        // sideways false-positive class.
                        ++s_rej_geom;
                    }
                }
                unknown_streak = 0;
                // Orient-sticky retry: the first few misses after a
                // successful detection re-try the same orient (the most
                // likely cause of a single miss is a blink / motion blur,
                // not a device rotation). Only after consecutive_misses
                // crosses the threshold do we begin cycling through the
                // other three orients.
                ++consecutive_misses;
                if (consecutive_misses > ORIENT_STICKY_MISSES)
                {
                    try_orient = static_cast<Orient>(((int)try_orient + 1) & 0x3);
                }
                if (consecutive_misses > OVERLAY_CLEAR_MISSES)
                {
                    // Sustained miss streak -- the face has almost
                    // certainly left. Hide the bbox / dots so the
                    // user gets visual feedback that the tracker
                    // lost lock. Shorter streaks (single blinks,
                    // motion blur, a half-second of orient cycling
                    // after a rotation) leave the overlay alone so
                    // it doesn't flicker every time the detector
                    // skips one frame.
                    portENTER_CRITICAL(&g_overlay_mux);
                    g_overlay.valid = false;
                    portEXIT_CRITICAL(&g_overlay_mux);
                }
                esp_camera_fb_return(fb);
                // Yield camera frames to the render task while nothing
                // interesting is going on. See NO_FACE_DELAY_MS comment.
                vTaskDelay(pdMS_TO_TICKS(NO_FACE_DELAY_MS));
                continue;
            }

            // Detection succeeded in this orientation; lock it in (don't
            // touch try_orient) so the next frame stays single-shot.
            const Orient locked_orient = try_orient;
            consecutive_misses = 0;
            ++s_detections;

            // Publish a display-frame snapshot of the bbox + keypoints
            // for the render task to draw on top of the live preview.
            // Both the bbox corners and each keypoint are produced in
            // the detector's (possibly rotated) input frame, so we
            // un-rotate them back into the camera frame the LCD shows
            // before publishing.
            {
                face_overlay_t snap = {};
                snap.valid    = true;
                snap.stamp_ms = now_ms();

                int p0x = biggest->box[0], p0y = biggest->box[1];
                int p1x = biggest->box[2], p1y = biggest->box[3];
                unrotate_point(locked_orient, FRAME_DIM, p0x, p0y);
                unrotate_point(locked_orient, FRAME_DIM, p1x, p1y);
                snap.box[0] = (int16_t)std::min(p0x, p1x);
                snap.box[1] = (int16_t)std::min(p0y, p1y);
                snap.box[2] = (int16_t)std::max(p0x, p1x);
                snap.box[3] = (int16_t)std::max(p0y, p1y);

                const int kp_pairs =
                    std::min(5, (int)biggest->keypoint.size() / 2);
                for (int i = 0; i < kp_pairs; ++i) {
                    int kx = biggest->keypoint[2 * i];
                    int ky = biggest->keypoint[2 * i + 1];
                    unrotate_point(locked_orient, FRAME_DIM, kx, ky);
                    snap.keypoints[2 * i]     = (int16_t)kx;
                    snap.keypoints[2 * i + 1] = (int16_t)ky;
                }

                portENTER_CRITICAL(&g_overlay_mux);
                g_overlay = snap;
                portEXIT_CRITICAL(&g_overlay_mux);
            }

            const int bw = biggest->box[2] - biggest->box[0];
            const int bh = biggest->box[3] - biggest->box[1];
            if (bw < MIN_FACE_PX || bh < MIN_FACE_PX)
            {
                ++s_too_small;
                unknown_streak = 0; // small box ⇒ not evidence of "new face"
                esp_camera_fb_return(fb);
                continue;
            }

            const int focus = focus_metric(to_detect,
                                           fb->width, fb->height,
                                           biggest->box[0], biggest->box[1],
                                           biggest->box[2], biggest->box[3]);
            if (focus < MIN_SHARPNESS)
            {
                ++s_too_blurry;
                unknown_streak = 0; // blurry ⇒ not trustworthy evidence
                esp_camera_fb_return(fb);
                continue;
            }

            // Recognition cache: if the most recent identified face's
            // bbox overlaps this one strongly (and the cache hasn't
            // gone stale), treat this frame as a continuation of that
            // identity and skip the ~50 ms embedder + cosine pass
            // entirely. The cache is invalidated on orient change so
            // we don't accidentally cross-compare bboxes from
            // different coordinate systems.
            const bool cache_hot =
                recog_cache.valid &&
                recog_cache.orient == locked_orient &&
                (now_ms() - recog_cache.stamp_ms) < RECOG_CACHE_MS &&
                iou_pct(biggest->box.data(), recog_cache.box) >
                    RECOG_IOU_PCT;
            if (cache_hot)
            {
                ++s_recognised;
                unknown_streak = 0;
                ESP_LOGD(TAG,
                         "cached recog: iou>%d, %u ms since last embed",
                         RECOG_IOU_PCT,
                         (unsigned)(now_ms() - recog_cache.stamp_ms));
                esp_camera_fb_return(fb);
                continue;
            }

            // Run the embedder. The returned TensorBase is owned by the
            // postprocessor and reused on the NEXT call, so we must copy
            // its floats before doing anything else that might run the
            // model again (which currently we don't, but the copy is
            // ~512 floats so it's not worth the foot-gun).
            dl::TensorBase *t = feat->run(img, biggest->keypoint);
            if (!t || t->get_size() != feat_len)
            {
                ESP_LOGW(TAG, "feat model returned unexpected tensor");
                esp_camera_fb_return(fb);
                continue;
            }
            const float *raw_feat = t->get_element_ptr<float>();

            // Compare against every stored embedding. We hold the
            // DB mutex for the duration so a concurrent rename from
            // the web server can't tear the read; the loop is at most
            // MAX_KNOWN_FACES * feat_len floats (~16 K FLOPs after
            // SIMD), well under a millisecond of critical section.
            float best_sim = -2.f;
            int best_id = -1;
            xSemaphoreTake(g_db_mux, portMAX_DELAY);
            for (size_t i = 0; i < g_known_faces.size(); ++i)
            {
                const float s = cosine_sim(raw_feat,
                                           g_known_faces[i].feat.data(),
                                           feat_len);
                if (s > best_sim)
                {
                    best_sim = s;
                    best_id = (int)i + 1; // 1-based for logging niceness
                }
            }
            xSemaphoreGive(g_db_mux);

            if (best_id > 0 && best_sim >= g_tuning.match_thr)
            {
                // Recognised — no enroll, no banner. We removed the
                // "OH... IT'S YOU" path: recognition is now silent so
                // the only on-screen event is a brand-new face.
                ++s_recognised;
                unknown_streak = 0;
                ESP_LOGI(TAG, "known face: id=%d sim=%.3f focus=%d orient=%d",
                         best_id, (double)best_sim, focus, (int)locked_orient);

                // Cache this hit so the next few frames at roughly
                // the same bbox can skip the embedder pass.
                recog_cache.valid    = true;
                recog_cache.box[0]   = biggest->box[0];
                recog_cache.box[1]   = biggest->box[1];
                recog_cache.box[2]   = biggest->box[2];
                recog_cache.box[3]   = biggest->box[3];
                recog_cache.orient   = locked_orient;
                recog_cache.stamp_ms = now_ms();

                esp_camera_fb_return(fb);
                continue;
            }

            // Either no enrolled faces at all, or every enrolled face was
            // below the match threshold. Treat this as evidence of "new
            // face" but require g_tuning.enroll_debounce such frames in a row
            // before actually enrolling — a single low-confidence frame
            // for a person we already know shouldn't fork a new identity.
            ++unknown_streak;
            ESP_LOGD(TAG, "unknown frame: best_sim=%.3f streak=%d/%d focus=%d",
                     (double)best_sim, unknown_streak,
                     g_tuning.enroll_debounce, focus);

            if (unknown_streak < g_tuning.enroll_debounce)
            {
                esp_camera_fb_return(fb);
                continue;
            }

            // Commit the enrollment. Copy the float* into the known list
            // (the source buffer is recycled by the model on the next run).
            // Every successful enrollment fires the "NEW FACE DETECTED"
            // overlay, refreshing both the deadline and the rotation angle
            // so the text always reads upright relative to the current
            // face pose.
            bool hit_cap = false;
            size_t new_count = 0;
            xSemaphoreTake(g_db_mux, portMAX_DELAY);
            if (g_known_faces.size() >= MAX_KNOWN_FACES) {
                hit_cap = true;
            } else {
                KnownFace entry;
                entry.feat.assign(raw_feat, raw_feat + feat_len);
                crop_face_thumb(src, biggest, locked_orient, &entry.thumb);
                entry.name[0]    = '\0';
                entry.enrolled_ms = now_ms();
                g_known_faces.emplace_back(std::move(entry));
                new_count = g_known_faces.size();
            }
            xSemaphoreGive(g_db_mux);

            if (hit_cap)
            {
                ESP_LOGD(TAG,
                         "skip enroll: hit global cap (%u), best_sim=%.3f",
                         (unsigned)MAX_KNOWN_FACES, (double)best_sim);
            }
            else
            {
                ++s_enrolled;

                // Newly enrolled face: prime the cache so we don't
                // immediately re-embed and re-match it against
                // itself on the very next frame.
                recog_cache.valid    = true;
                recog_cache.box[0]   = biggest->box[0];
                recog_cache.box[1]   = biggest->box[1];
                recog_cache.box[2]   = biggest->box[2];
                recog_cache.box[3]   = biggest->box[3];
                recog_cache.orient   = locked_orient;
                recog_cache.stamp_ms = now_ms();

                // Only (re)render the banner if one isn't already on
                // screen: we want the text orientation to reflect the
                // face pose AT THE MOMENT we first decided this was a
                // new face, not whatever angle the user happens to be
                // at five seconds later. Subsequent enrolments that
                // land inside the same 5 s window add their embeddings
                // silently and leave both the angle and the deadline
                // alone.
                if (g_banner_until_ms.load(std::memory_order_relaxed) <= now_ms())
                {
                    // Banner angle: discrete orientation refined with the
                    // eye-line roll measured from the detector's keypoints.
                    // See head_roll_for().
                    const float roll = head_roll_for(biggest, locked_orient);
                    banner_render(roll);
                    g_banner_until_ms.store(now_ms() + BANNER_HOLD_MS,
                                            std::memory_order_relaxed);

                    ESP_LOGI(TAG,
                             "NEW face enrolled id=%u (total=%u, bbox=%dx%d, focus=%d, orient=%d, roll=%.0fdeg)",
                             (unsigned)new_count,
                             (unsigned)new_count,
                             bw, bh, focus, (int)locked_orient,
                             (double)(roll * 180.f / 3.14159265f));
                }
                else
                {
                    ESP_LOGI(TAG,
                             "supplementary enrol id=%u (banner already up; total=%u, bbox=%dx%d, focus=%d)",
                             (unsigned)new_count,
                             (unsigned)new_count,
                             bw, bh, focus);
                }
            }
            unknown_streak = 0;

            esp_camera_fb_return(fb);
        }
    }

} // namespace

esp_err_t face_ai_init(void)
{
    // The DB mutex must exist before face_ai_task starts touching the
    // global g_known_faces. Init is idempotent: a second call to
    // face_ai_init (shouldn't happen, but defensive) won't leak the
    // old mutex because we only create one on the first pass.
    if (!g_db_mux) {
        g_db_mux = xSemaphoreCreateMutex();
    }
    if (!g_db_mux) {
        return ESP_ERR_NO_MEM;
    }

    const BaseType_t ok = xTaskCreatePinnedToCore(
        face_ai_task, "face_ai", TASK_STACK, nullptr,
        TASK_PRIO, nullptr, TASK_CORE);
    return (ok == pdPASS) ? ESP_OK : ESP_ERR_NO_MEM;
}

bool face_ai_banner_active(void)
{
    return g_banner_until_ms.load(std::memory_order_relaxed) > now_ms();
}

void face_ai_get_overlay(face_overlay_t *out)
{
    if (!out) {
        return;
    }
    portENTER_CRITICAL(&g_overlay_mux);
    *out = g_overlay;
    portEXIT_CRITICAL(&g_overlay_mux);
}

int face_db_count(void)
{
    if (!g_db_mux) {
        return 0;
    }
    xSemaphoreTake(g_db_mux, portMAX_DELAY);
    const int n = (int)g_known_faces.size();
    xSemaphoreGive(g_db_mux);
    return n;
}

bool face_db_get_entry(int idx, face_db_entry_t *out)
{
    if (!out || !g_db_mux) {
        return false;
    }
    bool ok = false;
    xSemaphoreTake(g_db_mux, portMAX_DELAY);
    if (idx >= 0 && idx < (int)g_known_faces.size()) {
        const auto &kf  = g_known_faces[idx];
        out->idx         = idx;
        out->enrolled_ms = kf.enrolled_ms;
        out->thumb_w     = FACE_THUMB_DIM;
        out->thumb_h     = FACE_THUMB_DIM;
        std::memcpy(out->name, kf.name, FACE_NAME_MAX);
        ok = true;
    }
    xSemaphoreGive(g_db_mux);
    return ok;
}

bool face_db_copy_thumb(int idx, uint16_t *dst, size_t dst_capacity_pixels)
{
    if (!dst || !g_db_mux ||
        dst_capacity_pixels < (size_t)FACE_THUMB_DIM * FACE_THUMB_DIM) {
        return false;
    }
    bool ok = false;
    xSemaphoreTake(g_db_mux, portMAX_DELAY);
    if (idx >= 0 && idx < (int)g_known_faces.size()) {
        const auto &kf = g_known_faces[idx];
        if (kf.thumb.size() == (size_t)FACE_THUMB_DIM * FACE_THUMB_DIM) {
            std::memcpy(dst, kf.thumb.data(),
                        kf.thumb.size() * sizeof(uint16_t));
            ok = true;
        }
    }
    xSemaphoreGive(g_db_mux);
    return ok;
}

bool face_db_set_name(int idx, const char *name)
{
    if (!name || !g_db_mux) {
        return false;
    }
    bool ok = false;
    xSemaphoreTake(g_db_mux, portMAX_DELAY);
    if (idx >= 0 && idx < (int)g_known_faces.size()) {
        auto &kf = g_known_faces[idx];
        std::strncpy(kf.name, name, FACE_NAME_MAX - 1);
        kf.name[FACE_NAME_MAX - 1] = '\0';
        ok = true;
    }
    xSemaphoreGive(g_db_mux);
    return ok;
}
