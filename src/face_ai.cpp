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
#include "camera.h"

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
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <vector>

// Test #11 (low-level perf pass): kick this entire translation unit up
// to -O3 with loop unrolling. The IDF default for our build is -O2
// (CONFIG_COMPILER_OPTIMIZATION_PERF=y); applying -O3 only here keeps
// the heavyweight ESP-DL and esp32-camera components compiled with
// their own validated flags while still letting GCC aggressively
// unroll our tight per-pixel loops (normalize_face_bbox, the bbox
// luma sample, the hflip TTA copy, etc.).
#pragma GCC push_options
#pragma GCC optimize ("O3")
#pragma GCC optimize ("unroll-loops")

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
    // Two floors, gated by the orient the detector ran in -- see the
    // `Orient` enum + `min_detect_score_for()` further down.
    //
    // Both floors were re-tuned for the ESPDET PICO 224 detector by
    // sampling its actual per-frame score distribution. Real-face
    // detections cluster tightly at 0.70-0.90 (mean ~0.81 in test
    // runs); ESPDET's own postprocessor already discards anything
    // below 0.50 before we see it, so app-level floors below 0.5 are
    // no-ops. The previous values (0.35 / 0.55) were inherited from
    // the MSR+MNP cascade, which had a much wider raw-score
    // distribution (0.3-0.95) and needed the slack.
    //
    //   * `MIN_DETECT_SCORE_ROT0` admits the lowest-score real faces
    //     (close-range padded fallback, partial occlusion, harsh
    //     side-lighting can pull legitimate scores down to ~0.55).
    //     Set just below the lowest observed real-face score.
    //
    //   * `MIN_DETECT_SCORE_ROTATED` is stricter -- the cost of
    //     accepting a weak detection at a non-default orient is high
    //     (the cycle locks there and the on-screen banner reads
    //     wrong-side-up); the cost of demanding more evidence is
    //     just one or two more cycles around the orient wheel.
    //     Legitimate rotated detections (user holding the device
    //     sideways) score 0.70+ so a floor of 0.65 doesn't gate
    //     genuine use.
    //
    // The orient cycle's other gates -- the 2-frame non-ROT_0 lock
    // confirmation, the ROT_0 cross-check, and the IoU recognition
    // cache -- catch the remaining false-positive classes (single-
    // frame fluky lock at the wrong orient, blob misfires on
    // textured backgrounds).
    constexpr float MIN_DETECT_SCORE_ROT0    = 0.50f;
    constexpr float MIN_DETECT_SCORE_ROTATED = 0.65f;

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

    // Spatial-track continuation. When the matcher returns "unknown"
    // but we very recently recognised someone whose bbox is still
    // roughly where the current bbox is, treat the current frame as
    // a continuation of that same identity rather than enrolling a
    // new one. This is the embedded variant of the DeepSORT pattern
    // (Wojke et al. 2017): spatial proximity is a more reliable
    // same-track signal than embedding similarity across short
    // timescales -- particularly when the user rotates the device
    // (the embedding shifts substantially across orients before the
    // gallery has a template for the new pose).
    //
    // SPATIAL_TRACK_MS is wider than RECOG_CACHE_MS so the override
    // can rescue frames that already missed the cache's IoU gate;
    // SPATIAL_TRACK_IOU_PCT is loose enough that an orient swap or
    // a half-frame head turn still passes; SPATIAL_TRACK_SIM_FLOOR
    // is a sanity check against the edge case where a different
    // person happens to land at the same bbox -- random face
    // embeddings sit around cosine 0.0, real same-person across
    // poses is reliably > 0.20 even at extreme angles.
    constexpr uint32_t SPATIAL_TRACK_MS         = 3000;
    constexpr int      SPATIAL_TRACK_IOU_PCT    = 30;
    constexpr float    SPATIAL_TRACK_SIM_FLOOR  = 0.20f;

    // Quality floor that a candidate enrolment frame must clear
    // before being committed to the DB. Below this, we silently
    // discard the candidate rather than seeding the gallery with a
    // low-quality template that subsequent across-pose detections
    // can't match -- which is the failure mode the spatial-track
    // continuation above is the second line of defence against.
    // 0.55 corresponds to roughly: detection score >= 0.75, focus
    // >= 75, and bbox area >= 90 x 90, all factored together.
    constexpr float    FACE_ENROL_MIN_QUALITY   = 0.55f;

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

    // Test #17: render-throttle flag. The face_ai task sets this true
    // around each detect->run / feat->run (the two ESP-DL inferences),
    // and the render task polls it in its main loop and skips the
    // PSRAM-heavy display_draw_rgb565() call while set. The flag-vs-
    // suspend approach lets the render task stay scheduled (so it can
    // wake up to drop the in-flight frame buffer back to the camera
    // pool) without consuming PSRAM bus bandwidth for the LCD push.
    std::atomic<bool> g_inference_busy{false};

    // Name-banner deadline: navy-blue banner along the bottom of the
    // LCD showing the recognised person's name. Refreshed every frame
    // that the matcher (or its IoU cache) returns a known face whose
    // user-set name is non-empty; the banner stays up for
    // NAME_BANNER_LINGER_MS past the last refresh so it doesn't blink
    // off every time the detector skips a frame on a blink / blur.
    // Same uint32_t-ms-from-boot lock-free atomic shape as
    // g_banner_until_ms for the same reason.
    std::atomic<uint32_t> g_name_banner_until_ms{0};

    // How long after the last successful name-banner refresh the
    // banner stays up. ~1.5 s feels right: long enough to bridge the
    // typical recognition gap when the face briefly leaves frame,
    // short enough that the name doesn't linger awkwardly after the
    // person actually walks away.
    constexpr uint32_t NAME_BANNER_LINGER_MS = 1500;

    // Latest face detection in DISPLAY-frame coordinates, published
    // each successful detection for the render task to draw on top of
    // the live preview. Guarded by a portMUX so the multi-field write
    // is atomic w.r.t. the render task on the other core.
    portMUX_TYPE          g_overlay_mux = portMUX_INITIALIZER_UNLOCKED;
    face_overlay_t        g_overlay     = {};

    // Known-face database. Each entry packages a small *gallery* of
    // L2-normalised embeddings (used by the matcher), a downsampled
    // RGB565 thumbnail cropped from the enrolment frame (served by
    // the web UI), and a user-set name (likewise mutated through the
    // web UI). The mutex gates every reader/writer; contention is
    // rare (writers fire on enrolment, the matcher reads inside one
    // critical section per detection, the HTTP handlers read on user
    // demand) so a plain FreeRTOS mutex is plenty.
    //
    // The embedding gallery (`feats`) starts with a single template
    // captured at enrolment. Each subsequent confidently-matched
    // detection that's pose-different from the existing templates
    // gets appended (up to FACE_MAX_TEMPLATES), so a person enrolled
    // looking straight at the camera gradually picks up off-angle /
    // profile views without the user having to re-enrol. The auto-
    // augment gate is tighter than the match gate by FACE_AUGMENT_MARGIN,
    // so a marginal match never seeds the gallery -- the gallery only
    // grows from matches that are unambiguous on the front-on
    // template, which keeps false-match rate from creeping up.
    //
    // Each template carries a *quality* score in [0, 1] computed at
    // ingest time from (detection_score * focus * bbox_area).
    // Quality drives slot eviction once the gallery is full: a new
    // augment candidate displaces the lowest-quality existing slot
    // only if its own quality is higher. This is the embedded variant
    // of the InsightFace / MagFace "quality-aware gallery" pattern --
    // we get to keep the best representative templates for each
    // identity instead of locking in whatever the first four high-
    // confidence frames happened to capture.
    constexpr size_t FACE_MAX_TEMPLATES   = 4;
    constexpr float  FACE_AUGMENT_MARGIN  = 0.10f;  // above match_thr
    constexpr float  FACE_AUGMENT_NOVEL   = 0.90f;  // template counts as new if best-existing-sim < this
    // Reference values for the quality-score normalisations. Scores
    // beyond these saturate at 1.0. Chosen so a "good" enrolment
    // frame -- score ~0.9, focus ~120, 130 x 130 face -- lands at
    // quality ~0.9.
    constexpr int    FACE_QUALITY_FOCUS_REF = 100;
    constexpr int    FACE_QUALITY_AREA_REF  = 120 * 120;

    struct FaceTemplate
    {
        std::vector<float> feat;
        float              quality = 0.0f;
    };

    struct KnownFace
    {
        // Gallery of L2-normalised embeddings + quality scores.
        // feats[0] is the enrolment template; feats[1..] are auto-
        // augmented templates captured from high-confidence matches
        // at different head poses. feat_len is invariant across
        // the gallery.
        std::vector<FaceTemplate> feats;
        std::vector<uint16_t>     thumb;        // FACE_THUMB_DIM^2 RGB565BE pixels
        char                      name[FACE_NAME_MAX] = {0};
        uint32_t                  enrolled_ms = 0;
    };

    // Combined quality for a freshly-captured face. Each factor is
    // clamped to [0, 1] and they multiply, so any single weakness
    // (very low confidence, blurry, small) drags the overall score
    // down -- exactly the behaviour we want when ranking templates
    // for gallery retention.
    static inline float face_template_quality(float det_score,
                                              int focus,
                                              int bbox_w,
                                              int bbox_h) noexcept
    {
        const float s_n = std::clamp(det_score, 0.0f, 1.0f);
        const float f_n = std::clamp(
            static_cast<float>(focus) /
            static_cast<float>(FACE_QUALITY_FOCUS_REF), 0.0f, 1.0f);
        const float a_n = std::clamp(
            static_cast<float>(bbox_w * bbox_h) /
            static_cast<float>(FACE_QUALITY_AREA_REF), 0.0f, 1.0f);
        return s_n * f_n * a_n;
    }
    std::vector<KnownFace> g_known_faces;
    SemaphoreHandle_t      g_db_mux = nullptr;

    // Cross-task signal: when set, the face_ai task clears its local
    // recognition cache on the next loop iteration. Raised by
    // face_db_delete() so a freshly-deleted face can't keep
    // "matching" via the AI task's cached bbox/id pair (whose
    // matched_id is now either dangling or, after the vector erase,
    // pointing at a different face). Single-writer / single-reader
    // (web-server task writes, AI task reads-and-clears) so a plain
    // bool atomic is fine.
    std::atomic<bool> g_recog_cache_invalidate{false};

    // RAII scope guard for g_db_mux. Acquires on construction, releases
    // on destruction; non-copyable / non-movable so we can't accidentally
    // hand out a lock. Every critical section against the face DB --
    // both inside the AI task and inside the public face_db_* readers
    // called from the web-server task -- goes through this guard.
    class FaceDbLock
    {
    public:
        FaceDbLock() noexcept { xSemaphoreTake(g_db_mux, portMAX_DELAY); }
        ~FaceDbLock() noexcept { xSemaphoreGive(g_db_mux); }
        FaceDbLock(const FaceDbLock &)            = delete;
        FaceDbLock &operator=(const FaceDbLock &) = delete;
    };

    inline uint32_t now_ms() noexcept
    {
        return static_cast<uint32_t>(esp_timer_get_time() / 1000);
    }

    // Cheap focus / motion-blur metric: average of |g(x,y) - g(x+1,y)| +
    // |g(x,y) - g(x,y+1)| over a strided sample of the face bbox, where g
    // is taken from the high byte of the BE-stored uint16_t pixel (green
    // dominates that byte in RGB565). A sharp face yields ~30-80, a blurry
    // or motion-smeared face ~5-15.
    //
    // Stride scales with the bbox's longer side so work is roughly
    // constant (~1600 samples) regardless of how close the face is.
    // Close-range faces can fill the frame (240x240 bbox) which at the
    // old fixed stride of 3 cost ~6400 iterations per call; the adaptive
    // stride caps that at ~1600 without losing the metric's stability.
    int focus_metric(const uint16_t *__restrict__ rgb565, int img_w, int img_h,
                     int x0, int y0, int x1, int y1) noexcept
    {
        x0 = std::clamp(x0, 1, img_w - 2);
        y0 = std::clamp(y0, 1, img_h - 2);
        x1 = std::clamp(x1, 1, img_w - 2);
        y1 = std::clamp(y1, 1, img_h - 2);
        if (x1 - x0 < 8 || y1 - y0 < 8) {
            return 0;
        }

        const int side   = std::max(x1 - x0, y1 - y0);
        const int stride = std::max(3, side / 40);

        int n   = 0;
        int sum = 0;
        for (int y = y0; y < y1; y += stride) {
            const uint16_t *row = rgb565 + y * img_w;
            for (int x = x0; x < x1 - 1; x += stride) {
                const int g  = (row[x]         >> 8) & 0xFF;
                const int gx = (row[x + 1]     >> 8) & 0xFF;
                const int gy = (row[x + img_w] >> 8) & 0xFF;
                sum += std::abs(g - gx) + std::abs(g - gy);
                ++n;
            }
        }
        return n ? sum / n : 0;
    }

    // Embeddings are L2-normalised by FeatPostprocessor::l2_norm() before
    // they leave the model (verified in esp-dl's
    // vision/recognition/dl_feat_postprocessor.cpp: postprocess() always
    // invokes l2_norm() before returning), so cosine similarity collapses
    // to a plain dot product. ESP-DSP ships an Xtensa LX7 SIMD
    // implementation of dsps_dotprod_f32 written with the PIE EE.*
    // 128-bit Q-register instructions; per the official ESP-DSP
    // benchmark page (docs.espressif.com/projects/esp-dsp/.../esp-dsp-
    // benchmarks.html) the S3-optimised version at N=256 is roughly
    // 3-5x the throughput of the ANSI implementation, which is enough
    // to keep the match loop well under a millisecond even at the full
    // 32-face DB cap.
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

    // See `MIN_DETECT_SCORE_ROT0` / `MIN_DETECT_SCORE_ROTATED` above
    // for the rationale behind the two-tier floor. Selector lives
    // here rather than next to the constants because it needs the
    // `Orient` enum (which can't move higher without dragging the
    // pre-rotation pipeline up with it).
    constexpr float min_detect_score_for(Orient o) noexcept
    {
        return (o == Orient::ROT_0) ? MIN_DETECT_SCORE_ROT0
                                    : MIN_DETECT_SCORE_ROTATED;
    }

    constexpr int FRAME_DIM = 240; // OV2640 is configured for 240x240 RGB565.

    // ---------------------------------------------------------------
    // Detector-input pipeline
    // ---------------------------------------------------------------
    //
    // Every camera frame the AI task picks up runs through:
    //
    //   1. prep_detect_input() -- 0/90/180/270 deg pre-rotation into
    //      an SRAM scratch buffer so the detector (trained on
    //      upright faces only) sees an upright input. ROT_0 fast-
    //      paths to the S3 SIMD memcpy; the rotated cases write
    //      non-sequentially. The same pass sub-samples the rotated
    //      buffer and publishes mean luma so the adaptive-AE preset
    //      cycle has live scene-brightness data to act on.
    //
    //   2. HumanFaceDetect::run() -- ESPDet PICO 224x224 single-stage
    //      detector from ESP-DL. The component's own
    //      ImagePreprocessor handles RGB565 -> RGB888 expansion,
    //      letterboxing to 224x224, and mean/std normalisation.
    //
    //   3. Close-range padded fallback -- if (2) returned nothing but
    //      the user is still likely in frame, shrink the prepared
    //      buffer into a centred inner box of a mid-grey frame and
    //      re-run the detector. Recovers face-fills-the-frame poses
    //      that the model misses at native scale.
    //
    //   4. Far-range upscale fallback -- counterpart of (3): a small
    //      far face below ESPDet's anchor floor gets pulled into
    //      range by nearest-neighbour upscaling a centred crop. Runs
    //      on the opposite phase of the orient cycle from (3).
    //
    //   5. App-level gates: MIN_DETECT_SCORE, MIN_FACE_PX, MIN_SHARPNESS.
    //
    //   6. Recognition cache (IoU-keyed) -- if the latest detection's
    //      bbox overlaps the previous matched face strongly, skip the
    //      ~50 ms embedder pass and treat it as a continuation.
    //
    //   7. HumanFaceFeat::run() + cosine-similarity sweep over the
    //      known-face DB (SIMD via dsps_dotprod_f32). The matcher
    //      max-pools over each face's gallery of templates.
    //
    //   8. Optional enrol (if no match) or name-banner publish (if
    //      matched against a face that has a user-set name).

    // Pure rotate -- ESPDet does its own internal preprocessing
    // (mean=0/std=255 -> [0,1] normalisation + letterbox to 224x224),
    // so this function's only job is to land the upright copy into
    // `dst`. ROT_0 fast-paths to the S3 SIMD memcpy; the rotated
    // cases are scalar by necessity (non-sequential stores break
    // SIMD).
    __attribute__((hot))
    void rotate_to_scratch(Orient o,
                           const uint16_t *__restrict__ src,
                           uint16_t *__restrict__ dst,
                           int n) noexcept
    {
        switch (o) {
        case Orient::ROT_0:
            // dsps_memcpy_aes3 is the ESP-DSP S3-specific
            // assembly memcpy (see
            // managed_components/.../modules/support/mem/esp32s3/
            // dsps_memcpy_aes3.S). The aligned hot loop pumps
            // 32 bytes per iteration via two ee.vld.128.ip +
            // ee.vst.128.ip pairs against the PIE Q registers,
            // measurably faster than scalar libc memcpy on a
            // 16-byte-aligned PSRAM->SRAM 115 KB copy. The exact
            // speedup is workload- and alignment-dependent; ESP-DSP
            // doesn't publish a benchmark for memcpy specifically.
            dsps_memcpy_aes3(dst, src, static_cast<size_t>(n) * n * sizeof(uint16_t));
            break;
        case Orient::ROT_90_CW:
            // Loop axes flipped so dst writes are sequential (incrementing
            // address) while src reads stride by n. On the S3 SRAM the
            // store-buffer coalesces sequential 16-bit writes; scatter
            // writes (stride n) don't coalesce and stall the pipeline.
            // Flipping was a ~ 4-5x speedup on the timed rotate.
            //
            //   dst[x*n + (n-1-y)] = src[y*n + x]
            //   =>  for output position (dy, dx):
            //         x = dy, y = n - 1 - dx
            //         dst[dy*n + dx] = src[(n-1-dx)*n + dy]
            for (int dy = 0; dy < n; ++dy) {
                uint16_t *drow = dst + dy * n;
                for (int dx = 0; dx < n; ++dx) {
                    drow[dx] = src[(n - 1 - dx) * n + dy];
                }
            }
            break;
        case Orient::ROT_180: {
            // Pair-rotate via uint32_t: each iteration consumes two
            // pixels from the tail of src and writes them, swapped
            // within the 32-bit word, to the head of dst. Halves the
            // loop count vs the scalar version and keeps both src
            // and dst accesses 32-bit aligned (n=240 -> total even).
            const int       total       = n * n;
            const int       total_pairs = total / 2;
            const uint32_t *src32       = reinterpret_cast<const uint32_t *>(src);
            uint32_t       *dst32       = reinterpret_cast<uint32_t *>(dst);
            for (int i = 0; i < total_pairs; ++i) {
                const uint32_t w = src32[total_pairs - 1 - i];
                // src word holds (pix_lo, pix_hi); dst needs
                // (pix_hi, pix_lo) so the reversed sequence stays
                // pixel-correct, not byte-reversed.
                dst32[i] = (w >> 16) | (w << 16);
            }
            break;
        }
        case Orient::ROT_270_CW:
            // Same axis-flip rationale as ROT_90_CW above; the only
            // difference is which corner the rotation pivots around.
            //
            //   dst[(n-1-x)*n + y] = src[y*n + x]
            //   =>  for output position (dy, dx):
            //         n - 1 - x = dy => x = n - 1 - dy
            //         y = dx
            //         dst[dy*n + dx] = src[dx*n + (n - 1 - dy)]
            for (int dy = 0; dy < n; ++dy) {
                uint16_t *drow = dst + dy * n;
                const int src_col = n - 1 - dy;
                for (int dx = 0; dx < n; ++dx) {
                    drow[dx] = src[dx * n + src_col];
                }
            }
            break;
        }
    }

    // ---------------------------------------------------------------
    // Motion pre-screen
    // ---------------------------------------------------------------
    //
    // Once the detector lock has fully expired (consecutive_misses
    // beyond OVERLAY_CLEAR_MISSES) the AI loop is just polling a
    // mostly-empty room. Running the detector + the padded / upscale
    // fallbacks on every frame in that state is ~80-160 ms of dead
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
        const uint8_t *p8 = reinterpret_cast<const uint8_t *>(fb);
        for (int j = 0; j < MOTION_GRID; ++j) {
            const int y = MOTION_STRIDE / 2 + j * MOTION_STRIDE;
            for (int i = 0; i < MOTION_GRID; ++i) {
                const int x   = MOTION_STRIDE / 2 + i * MOTION_STRIDE;
                const int idx = (y * FRAME_DIM + x) * 2;
                const uint8_t hi = p8[idx];
                const uint8_t lo = p8[idx + 1];
                const uint8_t g6 =
                    static_cast<uint8_t>(((hi & 0x07) << 3) | (lo >> 5));
                cur[j * MOTION_GRID + i] =
                    static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
            }
        }
        if (!m.valid) {
            std::memcpy(m.samples, cur, sizeof(cur));
            m.valid        = true;
            m.quiet_frames = 0;
            return false;
        }
        int sad = 0;
        for (int i = 0; i < MOTION_GRID * MOTION_GRID; ++i) {
            sad += std::abs(static_cast<int>(cur[i]) - static_cast<int>(m.samples[i]));
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
    // Latest 0..255 mean luma of the detector input, written each
    // frame by prep_detect_input() (an 8-stride subsample, ~10 us).
    // Read by the AI loop's adaptive-AE check; relaxed ordering is
    // fine because both writer and reader are on the same task, and
    // the only cross-task reader (web UI heartbeat in the future)
    // tolerates a stale value.
    std::atomic<int> g_last_mean_luma{128};

    // Latest face-region mean luma, sampled from the locked face's
    // bbox each frame the detector hits. -1 means "no recent face,
    // fall back to global frame luma for AE decisions". Publishes
    // its update time too so the AE loop can age it out: if no face
    // has been seen for FACE_LUMA_STALE_MS we drop back to the
    // whole-frame mean. This is the embedded-ISP "face-priority AE"
    // pattern -- biases sensor exposure off the subject's skin tones
    // instead of the background, which is the textbook fix for the
    // "subject in shadow against a sunlit window" failure mode.
    std::atomic<int>      g_last_face_luma{-1};
    std::atomic<uint32_t> g_last_face_luma_ms{0};
    constexpr uint32_t    FACE_LUMA_STALE_MS = 5000;

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

    constexpr int      AE_CHECK_INTERVAL_MS = 3000;
    // Cooldown after any preset change: ignore further transitions
    // for this long so the sensor's internal AE has time to actually
    // settle on the new preset's metering target before we evaluate
    // the resulting luma. The presets differ by ~8x in gain ceiling
    // and ~3 stops in metering bias, so the scene luma swings ~80
    // units in the seconds following a transition -- without this
    // cooldown we end up chasing the sensor's own AE response and
    // ping-pong the preset every 3 s. 8 s is empirically enough for
    // the OV2640's metering loop to stabilise even on big jumps.
    constexpr int      AE_TRANSITION_COOLDOWN_MS = 8000;
    // Hysteresis band widths chosen so the steady-state luma each
    // preset produces (~ 100-130 in BRIGHT, ~150-200 in MID, ~200+
    // in DIM) sits comfortably inside the band: the dead-zone has
    // to be wider than the sensor's own response to a preset change
    // or the bands themselves become the oscillator.
    //
    // Up-transitions (scene got brighter than current preset assumes):
    constexpr int      AE_DIM_TO_MID_LUMA    = 180;
    constexpr int      AE_MID_TO_BRIGHT_LUMA = 215;
    // Down-transitions (scene got dimmer than current preset assumes):
    constexpr int      AE_BRIGHT_TO_MID_LUMA = 85;
    constexpr int      AE_MID_TO_DIM_LUMA    = 55;

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

    struct Tuning {
        float match_thr;
        int   enroll_debounce;
    };

    constexpr Tuning TUNING_DIM = {
        /*match_thr*/       0.32f,
        /*enroll_debounce*/ 2,
    };
    constexpr Tuning TUNING_MID = {
        0.38f,
        1,
    };
    constexpr Tuning TUNING_BRIGHT = {
        0.45f,
        1,
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
            // The actual fix for "direct light on the face" failures:
            // bias the OV2640's metering toward protecting face-skin
            // highlights. With ae_level=0 + brightness=0 the sensor
            // meters for the global frame average, and if a face is
            // significantly brighter than its background (lamp,
            // sunlight through a window) the face's micro-contrast
            // -- nostrils, eye sockets, lip line -- gets clipped
            // into the top bins. The MSR+MNP detector was trained
            // on natural-light faces with that micro-contrast
            // intact, so the clipped frame is out of its training
            // distribution and detection silently fails. No amount
            // of preprocessing can recover what was already clipped
            // at the sensor, so the lever has to be at the AE
            // stage.
            //
            // ae_level=-1 + brightness=-1 nudges the metering target
            // ~1 stop lower; combined with the 4X gain ceiling this
            // keeps facial features off the clip in lamp / sunlight
            // conditions. Background will look slightly darker on
            // the live preview -- acceptable trade for the detector
            // actually firing.
            s->set_gainceiling(s, GAINCEILING_4X);
            s->set_ae_level(s, -1);
            s->set_brightness(s, -1);
            break;
        }
    }

    // Prepare the buffer the detector will read this frame: pre-
    // rotate the camera fb into the orient-cycle's chosen orientation
    // (the detector is trained on upright faces only) and write a
    // fresh mean luma to g_last_mean_luma so the adaptive-AE cycle
    // has something to act on. ESPDet handles its own input
    // normalisation and is robust enough to lighting variation that
    // application-side CLAHE / gamma preprocessing -- which the
    // earlier MSR+MNP cascade benefited from -- actively hurts here
    // by pushing the input outside ESPDet's training distribution.
    //
    // Returns the pointer to feed into HumanFaceDetect::run() -- the
    // rotated scratch buffer in all cases (the rotation must happen
    // before the detector sees the frame, and we can't mutate the
    // camera fb because the render task on the other core is using
    // it for the live preview).
    __attribute__((hot))
    const uint16_t *prep_detect_input(Orient o,
                                      const uint16_t *src,
                                      uint16_t *scratch,
                                      int n) noexcept
    {
        rotate_to_scratch(o, src, scratch, n);

        // 8-stride sub-sampled mean of the G6-derived luma proxy.
        // ~ 900 pixel reads / call, well under 100 us, gives the AE
        // preset cycle a stable scene-luma readout to follow.
        const uint8_t *p8 = reinterpret_cast<const uint8_t *>(scratch);
        uint32_t sum = 0;
        uint32_t cnt = 0;
        for (int y = 0; y < FRAME_DIM; y += 8) {
            const int row_base = y * FRAME_DIM;
            for (int x = 0; x < FRAME_DIM; x += 8) {
                const int idx = (row_base + x) * 2;
                const uint8_t hi = p8[idx];
                const uint8_t lo = p8[idx + 1];
                const uint8_t g6 = static_cast<uint8_t>(
                    ((hi & 0x07) << 3) | (lo >> 5));
                sum += static_cast<uint32_t>((g6 << 2) | (g6 >> 4));
                ++cnt;
            }
        }
        if (cnt) {
            g_last_mean_luma.store(static_cast<int>(sum / cnt),
                                   std::memory_order_relaxed);
        }
        return scratch;
    }
    // --- Close-range padded fallback -------------------------------
    //
    // ESP-WHO's MSR+MNP face detector silently fails on very-close-
    // range faces (subject's face filling almost the whole 240x240
    // frame). The model README
    // (espressif/esp-dl/models/human_face_detect/README.md) doesn't
    // publish training-set face-size statistics, so I can't quote
    // an exact in-distribution range, but empirically: when the
    // user is sitting at normal arm's-length the primary pass works;
    // when they lean in to the point the face covers >~80% of the
    // frame, primary returns nothing. The fallback exists for that
    // case.
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

    // RGB565 mid-grey (R5=15, G6=31, B5=15 -> 0x7BEF) byte-swapped
    // for BE storage, packed as a 32-bit pair for fast PSRAM fills.
    constexpr uint16_t PADDED_GREY_BE      = __builtin_bswap16(uint16_t{0x7BEFu});
    constexpr uint32_t PADDED_GREY_BE_PAIR =
        (uint32_t{PADDED_GREY_BE} << 16) | PADDED_GREY_BE;

    // One-time PSRAM fill of the padded scratch buffer with mid-grey.
    // Called once after allocation; shrink_into_padded() then only
    // rewrites the centred inner box per call, so the slow PSRAM
    // border writes (~80 KB of the 115 KB buffer) are paid exactly
    // once over the device's lifetime instead of every padded retry.
    void padded_scratch_fill_grey(uint16_t *dst) noexcept
    {
        constexpr size_t total_pairs = size_t{FRAME_DIM} * FRAME_DIM / 2;
        uint32_t *dst32 = reinterpret_cast<uint32_t *>(dst);
        for (size_t i = 0; i < total_pairs; ++i) {
            dst32[i] = PADDED_GREY_BE_PAIR;
        }
    }

    __attribute__((hot))
    void shrink_into_padded(const uint16_t *src, uint16_t *dst) noexcept
    {
        // dst's grey border was filled once by padded_scratch_fill_grey()
        // at task startup and the resample below never touches it,
        // so we can skip the ~115 KB grey-fill every call. We only
        // need to nearest-neighbour resample src (FRAME_DIM x FRAME_DIM)
        // into the centred INNER x INNER region.
        for (int y = 0; y < PADDED_INNER; ++y) {
            const int       sy   = (y * FRAME_DIM) / PADDED_INNER;
            const uint16_t *srow = src + static_cast<size_t>(sy) * FRAME_DIM;
            uint16_t       *drow = dst +
                static_cast<size_t>(PADDED_BORDER + y) * FRAME_DIM + PADDED_BORDER;
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
        v = std::clamp(v, PADDED_BORDER, PADDED_BORDER + PADDED_INNER - 1);
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

    // --- Far-range upscale fallback --------------------------------
    //
    // Symmetric counterpart to the close-range padded fallback. The
    // MSR+MNP cascade has a minimum face size baked into its anchor
    // priors -- empirically anything smaller than ~50x50 in the
    // 240x240 input slips below the smallest anchor and the detector
    // returns nothing, even though the face is plainly visible to a
    // human. Walking back from arm's length to ~6-8 ft typically
    // pushes a face into this band.
    //
    // The trick mirrors the padded path: take the centred UPSCALE_CROP
    // x UPSCALE_CROP region of the prepared detector buffer and
    // nearest-neighbour resample it up to fill the full 240x240
    // detector input. A small far face that was 40x40 in the centre
    // of the source frame becomes 80x80 in the upscaled view --
    // comfortably inside the model's anchor range. Detections that
    // come back have their coords remapped to the source frame so
    // the rest of the pipeline (orient unrotate, embedder, etc.)
    // treats them like a regular hit.
    //
    // Cost is identical to padded (one extra detector run, ~80 ms),
    // and the two paths run on opposite phases of the orient cycle
    // (see padded_eligible / upscale_eligible) so total idle-loop
    // cost stays bounded.

    constexpr int UPSCALE_CROP    = 120;
    constexpr int UPSCALE_BORDER  = (FRAME_DIM - UPSCALE_CROP) / 2;   // 60

    __attribute__((hot))
    void upscale_into_full(const uint16_t *src, uint16_t *dst) noexcept
    {
        // Nearest-neighbour 2x upscale of the centred UPSCALE_CROP region.
        // The padded path takes 240->160 and pads with grey; this one
        // takes 120->240 and fills the full buffer. Each dst row
        // re-reads one src row; no border / margin to maintain.
        for (int y = 0; y < FRAME_DIM; ++y) {
            const int sy = UPSCALE_BORDER + (y * UPSCALE_CROP) / FRAME_DIM;
            const uint16_t *srow = src + static_cast<size_t>(sy) * FRAME_DIM;
            uint16_t       *drow = dst + static_cast<size_t>(y) * FRAME_DIM;
            for (int x = 0; x < FRAME_DIM; ++x) {
                const int sx = UPSCALE_BORDER + (x * UPSCALE_CROP) / FRAME_DIM;
                drow[x] = srow[sx];
            }
        }
    }

    int remap_upscale_axis(int v) noexcept
    {
        v = std::clamp(v, 0, FRAME_DIM - 1);
        // Inverse of the upscale: detector coord v in [0, FRAME_DIM)
        // corresponds to source coord UPSCALE_BORDER + v * UPSCALE_CROP / FRAME_DIM.
        return UPSCALE_BORDER + (v * UPSCALE_CROP) / FRAME_DIM;
    }

    void remap_upscale_result(dl::detect::result_t *r) noexcept
    {
        r->box[0] = remap_upscale_axis(r->box[0]);
        r->box[1] = remap_upscale_axis(r->box[1]);
        r->box[2] = remap_upscale_axis(r->box[2]);
        r->box[3] = remap_upscale_axis(r->box[3]);
        for (size_t i = 0; i + 1 < r->keypoint.size(); i += 2) {
            r->keypoint[i]     = remap_upscale_axis(r->keypoint[i]);
            r->keypoint[i + 1] = remap_upscale_axis(r->keypoint[i + 1]);
        }
    }

    // Sub-sampled mean luma of the face bbox, used for face-priority
    // AE. Stride 4 keeps the cost negligible (~ FACE_AREA / 16 G6
    // reads per call) while still giving a stable mean. Operates on
    // the same RGB565BE pixel layout as everything else.
    int sample_face_luma(const uint16_t *fb, int img_w, int img_h,
                         int x0, int y0, int x1, int y1) noexcept
    {
        x0 = std::clamp(x0, 0, img_w - 1);
        y0 = std::clamp(y0, 0, img_h - 1);
        x1 = std::clamp(x1, 0, img_w);
        y1 = std::clamp(y1, 0, img_h);
        if (x1 - x0 < 8 || y1 - y0 < 8) {
            return -1;
        }
        const uint8_t *p8 = reinterpret_cast<const uint8_t *>(fb);
        uint32_t sum = 0;
        uint32_t n   = 0;
        for (int y = y0; y < y1; y += 4) {
            const int row_base = y * img_w;
            for (int x = x0; x < x1; x += 4) {
                const int     idx = (row_base + x) * 2;
                const uint8_t hi  = p8[idx];
                const uint8_t lo  = p8[idx + 1];
                // G6 expanded to 8 bits = luma proxy, matching what
                // prep_detect_input() uses for g_last_mean_luma.
                const uint8_t g6  = static_cast<uint8_t>(((hi & 0x07) << 3) |
                                                         (lo >> 5));
                sum += static_cast<uint32_t>((g6 << 2) | (g6 >> 4));
                ++n;
            }
        }
        return n ? static_cast<int>(sum / n) : -1;
    }

    // Estimate 5 facial keypoints (eyes, nose, mouth corners) from
    // the face bbox using classical CV: find the two darkest blobs
    // in the upper-face "eye band", then synthesize nose and mouth
    // from canonical face proportions anchored to the eye line.
    //
    // Why this exists: ESPDet (our active detector) emits bbox +
    // score only, no keypoint head. MFN's FeatImagePreprocessor
    // requires five 2D landmarks to compute its similarity-transform
    // alignment. Without real eye positions the alignment degrades
    // to a bbox center-crop, which gives the embedder the wrong
    // signal whenever the face is tilted or off-centre inside its
    // bbox -- exactly the cases the multi-template gallery was
    // built to paper over. Promoting the eye localisation from
    // "guess from bbox" to "darkest-blob scan of the eye band"
    // lets us recover the roll-and-translate component of the
    // ArcFace alignment cheaply, at ~200 us per call. Nose and
    // mouth coordinates come from human-face anthropometry --
    // eye_dx (inter-pupillary distance) is the standard 1.0-unit
    // scale, the nose sits ~0.55 eye_dx below the eye midline, and
    // the mouth corners flank the same vertical at ~0.35 eye_dx
    // outward and ~1.10 eye_dx below.
    //
    // Algorithm:
    //   1. Eye band: y in bbox-relative [0.25, 0.50], split into
    //      left half [0.10, 0.50] and right half [0.50, 0.90] in x.
    //      In each half, scan with stride 2 for the darkest 4x4
    //      mean luma point. Those are the eye centres.
    //   2. Validate eye separation and rough horizontality. If the
    //      separation collapses or the eye line tilts more than
    //      ~30 deg (the orient cycle should have eaten anything
    //      tilted further), fall back to the proportional ArcFace
    //      template projected into the bbox.
    //   3. Synthesise nose at eye_mid + 0.55 * eye_vec_rotated_perpendicular,
    //      mouth corners at eye_mid +/- 0.35 * eye_dx along the
    //      eye line and 1.10 * eye_dx below it.
    //
    // The roll-aware synthesis (using eye_vec rather than image y)
    // is what gives MFN the rotation component of its affine
    // transform. Without it we'd be feeding a tilted face into a
    // model that expects an upright crop.
    //
    // Returns true if 5 plausible keypoints were synthesised into
    // out_kp. False -> caller should fall back to the bbox-centred
    // ArcFace template.
    bool estimate_face_keypoints(const uint16_t *fb, int img_w, int img_h,
                                 int x0, int y0, int x1, int y1,
                                 std::vector<int> *out_kp) noexcept
        __attribute__((hot));
    bool estimate_face_keypoints(const uint16_t *fb, int img_w, int img_h,
                                 int x0, int y0, int x1, int y1,
                                 std::vector<int> *out_kp) noexcept
    {
        const int bw = x1 - x0;
        const int bh = y1 - y0;
        if (bw < 32 || bh < 32) {
            return false; // too small to localise reliably
        }
        const uint8_t *p8 = reinterpret_cast<const uint8_t *>(fb);

        // 4x4 mean-luma at (cx, cy). Stride-2 sampling so we evaluate
        // 4 pixels per centre, not 16 -- the dark-blob peak is much
        // wider than 4 px so the stride costs nothing in accuracy.
        auto luma_at = [&](int cx, int cy) -> int {
            cx = std::clamp(cx, 1, img_w - 3);
            cy = std::clamp(cy, 1, img_h - 3);
            int sum = 0;
            for (int dy = -1; dy <= 1; dy += 2) {
                const int row_base = (cy + dy) * img_w;
                for (int dx = -1; dx <= 1; dx += 2) {
                    const int idx = (row_base + (cx + dx)) * 2;
                    const uint8_t hi = p8[idx];
                    const uint8_t lo = p8[idx + 1];
                    const int g6 = ((hi & 0x07) << 3) | (lo >> 5);
                    sum += g6;
                }
            }
            return sum; // 0..252, lower = darker
        };

        // Eye band: y in [bh*0.22, bh*0.55] (relative to bbox top).
        // Widened from the original [0.25, 0.50] so a face that
        // detects with a slightly-low or slightly-tall bbox still
        // has both eye centres inside the search window. The eye-
        // socket dark blob is robust enough that the extra noise
        // from the wider band doesn't pull the centroid off the
        // true eye centre.
        const int eye_y0 = std::clamp(y0 + (bh * 22) / 100, 0, img_h - 1);
        const int eye_y1 = std::clamp(y0 + (bh * 55) / 100, 0, img_h);

        // Locate the eye centre in a (sx0..sx1, sy0..sy1) zone using a
        // dark-weighted centroid of the darkest ~10 % of sampled
        // pixels in the zone. Pure argmin (the previous behaviour)
        // works for sharp, well-lit frames but fails when:
        //   * Partial occlusion (hair, hand, side of the head)
        //     leaves a small but very dark blob off-centre from the
        //     true eye.
        //   * Motion blur smears the eye into a 4-6 px wide line
        //     and the argmin lands at one end of the line.
        //   * Sensor noise on high gain creates a single 1-px dark
        //     outlier that beats the real eye blob.
        // A centroid of the darkest decile averages over the whole
        // eye blob instead of picking a single point. ~ 200 us total
        // for both eyes (negligible).
        auto find_dark_in = [&](int sx0, int sx1, int sy0, int sy1,
                                int *out_x, int *out_y) {
            // Pass 1: build a histogram of luma over the zone so we
            // can find the threshold corresponding to the darkest
            // ~10 % of samples. 13-bin histogram (each bin = 32 raw
            // 4x4-summed units = 32/4/8 = 1 8-bit-luma unit) is
            // plenty for selecting a decile cut-off.
            constexpr int N_BINS = 32;
            int hist[N_BINS] = {};
            int total = 0;
            for (int yy = sy0; yy < sy1; yy += 2) {
                for (int xx = sx0; xx < sx1; xx += 2) {
                    int lum = luma_at(xx, yy); // 0..252 in 4x g6 units
                    // Quantise to N_BINS buckets.
                    int b = (lum * N_BINS) / 256;
                    if (b >= N_BINS) b = N_BINS - 1;
                    ++hist[b];
                    ++total;
                }
            }
            int target  = std::max(1, total / 10);   // darkest decile
            int running = 0;
            int cut_bin = 0;
            for (; cut_bin < N_BINS; ++cut_bin) {
                running += hist[cut_bin];
                if (running >= target) break;
            }
            const int cut_lum =
                std::min(252, ((cut_bin + 1) * 256) / N_BINS);

            // Pass 2: weighted centroid of pixels darker than cut_lum.
            // Weight = (cut_lum - lum) so the darkest pixels pull the
            // hardest. Falls back to plain argmin if for some reason
            // no pixels qualify (shouldn't happen given pass 1).
            double sum_x = 0, sum_y = 0, sum_w = 0;
            int argmin_x = (sx0 + sx1) / 2;
            int argmin_y = (sy0 + sy1) / 2;
            int argmin_lum = INT_MAX;
            for (int yy = sy0; yy < sy1; yy += 2) {
                for (int xx = sx0; xx < sx1; xx += 2) {
                    const int lum = luma_at(xx, yy);
                    if (lum < argmin_lum) {
                        argmin_lum = lum;
                        argmin_x = xx;
                        argmin_y = yy;
                    }
                    if (lum < cut_lum) {
                        const double w = static_cast<double>(cut_lum - lum) + 1.0;
                        sum_w += w;
                        sum_x += w * xx;
                        sum_y += w * yy;
                    }
                }
            }
            if (sum_w > 0) {
                *out_x = static_cast<int>(sum_x / sum_w + 0.5);
                *out_y = static_cast<int>(sum_y / sum_w + 0.5);
            } else {
                *out_x = argmin_x;
                *out_y = argmin_y;
            }
        };

        // Left eye: x in [bw*0.10, bw*0.50].
        // Right eye: x in [bw*0.50, bw*0.90].
        const int lx0 = std::clamp(x0 + (bw * 10) / 100, 0, img_w - 1);
        const int lx1 = std::clamp(x0 + (bw * 50) / 100, 0, img_w);
        const int rx0 = lx1;
        const int rx1 = std::clamp(x0 + (bw * 90) / 100, 0, img_w);

        int le_x = 0, le_y = 0, re_x = 0, re_y = 0;
        find_dark_in(lx0, lx1, eye_y0, eye_y1, &le_x, &le_y);
        find_dark_in(rx0, rx1, eye_y0, eye_y1, &re_x, &re_y);

        // Validate eye geometry. eye_dx must be a plausible fraction
        // of the bbox width (faces with extreme yaw push it lower,
        // detection-fragmentation pushes it higher), and the eye
        // line must be roughly horizontal in detector coords.
        const int eye_dx = re_x - le_x;
        const int eye_dy = re_y - le_y;
        if (eye_dx < (bw * 18) / 100 || eye_dx > (bw * 65) / 100) {
            return false;
        }
        if (std::abs(eye_dy) * 5 > eye_dx * 4) {
            // Eye line tilts > ~38 deg -- almost always a mis-
            // localisation (the orient cycle should have caught
            // anything tilted further), so bail to keep the affine
            // transform well-conditioned. Was previously 26 deg,
            // widened so a real head tilt (e.g. shoulder-resting
            // pose) doesn't drop into the bbox-center-crop fallback.
            return false;
        }

        // Anthropometric synthesis of nose + mouth, using the eye
        // line as the local x axis. Coefficients are the standard
        // ArcFace ratios re-anchored to eye_dx as the unit scale:
        //   nose:        eye_mid + 0.55 * perp
        //   mouth_left:  eye_mid - 0.35 * along + 1.10 * perp
        //   mouth_right: eye_mid + 0.35 * along + 1.10 * perp
        // where `along` = unit vector from left eye to right eye,
        // `perp`  = along rotated 90 deg CW (i.e. "down" in face
        // coords). Using vectors means the synthesised landmarks
        // rotate with the head -- if the user tilts, nose and
        // mouth tilt with the eyes.
        const float fdx = static_cast<float>(eye_dx);
        const float fdy = static_cast<float>(eye_dy);
        const float norm = std::sqrt(fdx * fdx + fdy * fdy);
        if (norm < 1.f) {
            return false;
        }
        const float along_x = fdx / norm;
        const float along_y = fdy / norm;
        const float perp_x  = -along_y;   // CW 90 deg
        const float perp_y  =  along_x;
        const float mid_x   = (le_x + re_x) * 0.5f;
        const float mid_y   = (le_y + re_y) * 0.5f;

        const float nose_x  = mid_x + 0.55f * norm * perp_x;
        const float nose_y  = mid_y + 0.55f * norm * perp_y;
        const float ml_x    = mid_x - 0.35f * norm * along_x + 1.10f * norm * perp_x;
        const float ml_y    = mid_y - 0.35f * norm * along_y + 1.10f * norm * perp_y;
        const float mr_x    = mid_x + 0.35f * norm * along_x + 1.10f * norm * perp_x;
        const float mr_y    = mid_y + 0.35f * norm * along_y + 1.10f * norm * perp_y;

        out_kp->resize(10);
        (*out_kp)[0] = le_x;
        (*out_kp)[1] = le_y;
        (*out_kp)[2] = re_x;
        (*out_kp)[3] = re_y;
        (*out_kp)[4] = static_cast<int>(nose_x + 0.5f);
        (*out_kp)[5] = static_cast<int>(nose_y + 0.5f);
        (*out_kp)[6] = static_cast<int>(ml_x + 0.5f);
        (*out_kp)[7] = static_cast<int>(ml_y + 0.5f);
        (*out_kp)[8] = static_cast<int>(mr_x + 0.5f);
        (*out_kp)[9] = static_cast<int>(mr_y + 0.5f);
        return true;
    }

    // ---- Per-bbox luminance normalisation (percentile stretch + gamma) ----
    //
    // ESP-DL's FeatImagePreprocessor crops + warps from the raw RGB565
    // frame straight into MFN's 112x112 input. If the face is
    // significantly dark (backlit) or blown out relative to the
    // global AE-controlled scene, the warp inherits that bias and
    // MFN sees an embedding-shifted face, hurting both enrolment
    // quality and recognition cosine sim.
    //
    // We normalise the bbox region of `to_detect` IN-PLACE before
    // calling feat->run. Strategy:
    //   1. Walk bbox pixels, build a 256-bin Y luma histogram.
    //   2. Find P2 / P98 from the histogram -> [lo, hi] dynamic range.
    //   3. Compute the post-stretch mean by summing hist[i] * stretch(i).
    //   4. Pick a gamma to nudge the post-stretch mean toward TARGET=128:
    //        gamma = log(TARGET/255) / log(mean/255)
    //      clamped to [0.6, 1.6] so we never over-correct.
    //   5. Build a single LUT that maps raw Y -> stretched+gamma Y.
    //   6. Re-walk bbox pixels; for each one decode RGB565BE, compute
    //      Y, look up the new Y, scale all channels by new_Y/Y to
    //      preserve chroma, repack to RGB565BE, write back.
    //
    // This runs once per detected face (per frame), and is repeated
    // implicitly for the hflip TTA pass because that pass copies
    // from to_detect AFTER normalisation. Cost is O(bbox_area) twice
    // (~30 ms worst case for a 140x140 bbox on a 240 MHz LX7), kept
    // cheap by avoiding float math in the hot loops.
    //
    // We deliberately leave the original frame buffer (`src`) alone --
    // crop_face_thumb() runs on `src`, not `to_detect`, so the thumb
    // remains photographically faithful.
    bool normalize_face_bbox(uint16_t *to_detect_be, int frame_dim,
                              int x0, int y0, int x1, int y1) noexcept
        __attribute__((hot));
    bool normalize_face_bbox(uint16_t *to_detect_be, int frame_dim,
                              int x0, int y0, int x1, int y1) noexcept
    {
        x0 = std::max(0, x0);
        y0 = std::max(0, y0);
        x1 = std::min(frame_dim - 1, x1);
        y1 = std::min(frame_dim - 1, y1);
        if (x1 <= x0 + 4 || y1 <= y0 + 4) {
            return false;
        }

        // Pass 1: histogram of Y over bbox.
        uint32_t hist[256] = {0};
        uint32_t total = 0;
        for (int y = y0; y <= y1; ++y) {
            const uint16_t *row = to_detect_be + y * frame_dim;
            for (int x = x0; x <= x1; ++x) {
                const uint16_t be = row[x];
                const uint16_t p  = static_cast<uint16_t>((be >> 8) |
                                                          (be << 8));
                const int r = ((p >> 11) & 0x1F) << 3;
                const int g = ((p >> 5)  & 0x3F) << 2;
                const int b = ( p        & 0x1F) << 3;
                const int yl = (77 * r + 150 * g + 29 * b) >> 8;
                ++hist[yl & 0xFF];
                ++total;
            }
        }
        if (total < 256) {
            return false;
        }

        // Pass 2: find P2, P98.
        const uint32_t p_lo_target = total * 2u  / 100u;
        const uint32_t p_hi_target = total * 98u / 100u;
        int lo = 0, hi = 255;
        uint32_t acc = 0;
        for (int i = 0; i < 256; ++i) {
            acc += hist[i];
            if (acc >= p_lo_target) { lo = i; break; }
        }
        acc = 0;
        for (int i = 0; i < 256; ++i) {
            acc += hist[i];
            if (acc >= p_hi_target) { hi = i; break; }
        }
        if (hi - lo < 12) {
            // Too little dynamic range -- the face is so flat any
            // stretch would just amplify noise. Skip.
            return false;
        }

        // Estimate post-stretch mean luma to choose gamma.
        const float inv_span = 255.0f / static_cast<float>(hi - lo);
        double sum_stretched = 0.0;
        for (int i = lo; i <= hi; ++i) {
            const float v = (i - lo) * inv_span;
            sum_stretched += static_cast<double>(hist[i]) * v;
        }
        // Pixels below lo clamp to 0; pixels above hi clamp to 255.
        for (int i = hi + 1; i < 256; ++i) sum_stretched += hist[i] * 255.0;
        const float mean_stretched = static_cast<float>(
            sum_stretched / static_cast<double>(total));

        constexpr float TARGET = 128.0f;
        float gamma = 1.0f;
        if (mean_stretched > 1.0f && mean_stretched < 254.0f) {
            const float ln_mean = std::log(mean_stretched / 255.0f);
            const float ln_tgt  = std::log(TARGET         / 255.0f);
            if (std::fabs(ln_mean) > 1e-3f) {
                gamma = ln_tgt / ln_mean;
            }
        }
        gamma = std::clamp(gamma, 0.6f, 1.6f);

        // Build LUT: in_luma -> out_luma (stretch + gamma).
        uint8_t lut[256];
        for (int i = 0; i < 256; ++i) {
            float v;
            if (i <= lo) {
                v = 0.0f;
            } else if (i >= hi) {
                v = 1.0f;
            } else {
                v = (i - lo) * inv_span * (1.0f / 255.0f);
            }
            const float vg = std::pow(v, gamma);
            const int   out = static_cast<int>(vg * 255.0f + 0.5f);
            lut[i] = static_cast<uint8_t>(std::clamp(out, 0, 255));
        }

        // Pass 3: rewrite bbox pixels using the LUT, preserving chroma
        // by scaling RGB by new_Y / Y.
        for (int y = y0; y <= y1; ++y) {
            uint16_t *row = to_detect_be + y * frame_dim;
            for (int x = x0; x <= x1; ++x) {
                const uint16_t be = row[x];
                const uint16_t p  = static_cast<uint16_t>((be >> 8) |
                                                          (be << 8));
                int r = ((p >> 11) & 0x1F) << 3;
                int g = ((p >> 5)  & 0x3F) << 2;
                int b = ( p        & 0x1F) << 3;
                int yl = (77 * r + 150 * g + 29 * b) >> 8;
                if (yl < 1) yl = 1;
                const int new_y = lut[yl & 0xFF];
                r = std::clamp((r * new_y) / yl, 0, 255);
                g = std::clamp((g * new_y) / yl, 0, 255);
                b = std::clamp((b * new_y) / yl, 0, 255);
                const uint16_t out =
                    static_cast<uint16_t>(((r & 0xF8) << 8) |
                                          ((g & 0xFC) << 3) |
                                          ( b >> 3));
                row[x] = static_cast<uint16_t>((out >> 8) | (out << 8));
            }
        }
        return true;
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
        return -static_cast<float>(static_cast<int>(o)) * quarter;
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

    // (Removed: the keypoint-geometry sanity check that gated the
    // MSR+MNP cascade. ESPDet doesn't emit keypoints, so there's
    // nothing to check; the score floor + 2-frame non-ROT_0 confirm
    // + ROT_0 cross-check carry the false-positive defence on their
    // own. See git history for the previous implementation.)

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

        out_thumb->resize(static_cast<size_t>(FACE_THUMB_DIM) * FACE_THUMB_DIM);
        uint16_t *dst = out_thumb->data();
        for (int ty = 0; ty < FACE_THUMB_DIM; ++ty) {
            const int       sy   = cy0 + (ty * s) / FACE_THUMB_DIM;
            const uint16_t *srow = fb_pixels + static_cast<size_t>(sy) * FRAME_DIM;
            for (int tx = 0; tx < FACE_THUMB_DIM; ++tx) {
                const int sx = cx0 + (tx * s) / FACE_THUMB_DIM;
                dst[ty * FACE_THUMB_DIM + tx] = srow[sx];
            }
        }
    }

    // Previously we also ran a `head_roll_for()` helper that refined the
    // banner rotation using the detector's eye + nose keypoints. The
    // refinement was unstable in practice: it computed atan2(-dx, dy)
    // with a hard-coded +pi/2 offset, then snapped to the nearest
    // quarter-turn. Any case where the nose landed slightly to the left
    // of the eye midpoint (very common with glasses or even a hint of
    // keypoint jitter) tipped the snap from the upright bucket into the
    // upside-down bucket, producing inverted banner text on faces that
    // were clearly upright. Since the snap output is one of four
    // quarter-turns anyway, the orient-lock-driven banner_angle_for()
    // gives the same granularity with none of the failure modes —
    // we just trust the orient cycle.

    void face_ai_task(void *)
    {
        ESP_LOGI(TAG, "task running on core %d, prio %d",
                 xPortGetCoreID(), (int)TASK_PRIO);

        // Construction loads the .espdl files out of the model partitions.
        // Costs ~1 second up-front. unique_ptr so the destructors run
        // automatically if the task ever exits (today it doesn't, but
        // the leak-free shape costs nothing).
        auto detect = std::make_unique<HumanFaceDetect>();
        // The detector model is ESPDet PICO 224x224 (single-stage,
        // anchor-free). ESP-DL's component initialises it with a 0.5
        // sigmoid score threshold inside the postprocessor; on top of
        // that we apply MIN_DETECT_SCORE_ROT0 / MIN_DETECT_SCORE_ROTATED
        // at the app level to gate orient-cycle locks against the
        // few low-confidence rotated false-positives that slip past.
        // Test #14 (feat root-cause audit): instead of letting
        // HumanFaceFeat default to lazy_load=true (model is loaded on
        // first call to feat->run inside the loop, which hides cost
        // and timing), force the load at task init and bracket it
        // with heap + wall-clock probes. ESP-DL's Model ctor defaults
        // to param_copy=true which is meant to copy the ~1.3 MB of
        // MFN_S8_V1 weights from the human_face_feat flash partition
        // into PSRAM (octal @ 80 MHz, ~4x faster than flash). If the
        // PSRAM free count doesn't drop by ~1.3 MB across this call,
        // param_copy silently failed and inference is streaming
        // weights from flash on every layer -- that would directly
        // explain why feat->run takes ~975 ms vs the official 254 ms
        // S3 benchmark.
        const size_t psram_before  = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        const size_t intern_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
        const int64_t feat_load_t0 = esp_timer_get_time();
        auto      feat     = std::make_unique<HumanFaceFeat>(
            static_cast<HumanFaceFeat::model_type_t>(CONFIG_DEFAULT_HUMAN_FACE_FEAT_MODEL),
            /*lazy_load=*/false);
        const int64_t feat_load_us = esp_timer_get_time() - feat_load_t0;
        const size_t psram_after   = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        const size_t intern_after  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
        // Test #14 confirmed model loads cleanly into PSRAM
        // (~2 MB consumed) + 71 KB internal. We keep this one-shot
        // log because it's useful to confirm after any model swap
        // that placement still works the way we expect.
        ESP_LOGI(TAG,
                 "feat load: %u ms | PSRAM used=%u KB (free %u -> %u) | INTERNAL used=%u KB (free %u -> %u)",
                 (unsigned)(feat_load_us / 1000),
                 (unsigned)((psram_before  - psram_after)  / 1024),
                 (unsigned)psram_before,  (unsigned)psram_after,
                 (unsigned)((intern_before - intern_after) / 1024),
                 (unsigned)intern_before, (unsigned)intern_after);
        const int feat_len = feat->get_feat_len();

        // Scratch buffer for orientation-rotated frames. Prefer
        // internal SRAM (~3-4x faster than octal PSRAM at 80 MHz) so
        // the detector's input read traffic doesn't hit the slow bus.
        // ~115 KB is a big chunk of the S3's 320 KB internal heap but
        // it fits comfortably alongside the existing IDF + camera
        // stacks; if for some reason it doesn't, we transparently
        // fall back to PSRAM and lose only the bandwidth bonus.
        const size_t scratch_bytes =
            static_cast<size_t>(FRAME_DIM) * FRAME_DIM * sizeof(uint16_t);
        auto *scratch = static_cast<uint16_t *>(heap_caps_aligned_alloc(
            16, scratch_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        if (scratch) {
            ESP_LOGI(TAG, "orientation scratch in internal SRAM (%u B)",
                     (unsigned)scratch_bytes);
        } else {
            scratch = static_cast<uint16_t *>(heap_caps_aligned_alloc(
                16, scratch_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
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
        auto *padded_scratch = static_cast<uint16_t *>(heap_caps_aligned_alloc(
            16, scratch_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (!padded_scratch) {
            ESP_LOGW(TAG,
                     "no PSRAM for padded fallback (%u B); close-range "
                     "detection may suffer",
                     (unsigned)scratch_bytes);
        } else {
            // Grey border is invariant -- fill once and let
            // shrink_into_padded() only rewrite the inner box.
            padded_scratch_fill_grey(padded_scratch);
        }

        // The known-face DB lives at module scope (g_known_faces /
        // g_db_mux) so the web server can read it from another task.
        // The task acquires g_db_mux for every read or write below.
        // Pre-reserve so the matcher's pointer to the embedding stays
        // valid for the lifetime of the entry.
        {
            FaceDbLock lock;
            if (g_known_faces.capacity() < MAX_KNOWN_FACES) {
                g_known_faces.reserve(MAX_KNOWN_FACES);
            }
        }

        // Counters for the per-second heartbeat.
        int64_t stats_next_us = esp_timer_get_time() + STATS_PERIOD_US;

        // Test #16: PSRAM-contention experiment. Inference variance
        // (~25% across consecutive frames) plus published 248 ms vs
        // our 1100 ms feat:model time both pointed at the LCD render
        // task pulling 240x240x2 = 115 KB per render frame out of
        // octal PSRAM while MFN's weight streaming is trying to use
        // the same bus. We pause the render task around every
        // feat->run call to free the PSRAM bus for MFN, then resume.
        //
        // The handle is looked up lazily because main.cpp creates
        // the render task after face_ai_init; first lookup may fail
        // if face_ai_task races ahead. We retry until it succeeds
        // (typically the first time we have a face).
        TaskHandle_t render_task_handle = nullptr;
        (void)render_task_handle;  // retained for opt-in suspend testing
        auto render_suspend = [&]() {
            g_inference_busy.store(true, std::memory_order_release);
        };
        auto render_resume = [&]() {
            g_inference_busy.store(false, std::memory_order_release);
        };

        // Test #10: per-stage timing instrumentation. Wraps the hot
        // calls in esp_timer_get_time() bookends and accumulates
        // sum/max/count over the 1 s heartbeat window. Logged as a
        // second "1s timing:" line alongside the existing stats. The
        // wall-clock cost of esp_timer_get_time() is ~0.5 us, dwarfed
        // by every stage we measure here, so the instrumentation is
        // effectively free.
        enum StageIdx {
            ST_PREP = 0,   // prep_detect_input (rotate + luma subsample)
            ST_DETECT,     // detect->run (main pass only)
            ST_NORMALIZE,  // normalize_face_bbox (Test #5 stretch + gamma)
            ST_FEAT,       // feat->run (MFN embedding, straight pass)
            ST_QEMA,       // query EMA blend + L2-renorm (Test #9)
            ST_COSINE,     // gallery cosine_sim sweep under db mutex
            ST_COUNT
        };
        static const char *ST_NAMES[ST_COUNT] = {
            "prep", "detect", "norm", "feat", "qema", "cos"
        };
        uint64_t st_sum_us[ST_COUNT] = {0};
        uint32_t st_max_us[ST_COUNT] = {0};
        uint32_t st_n[ST_COUNT]      = {0};
        auto record_stage = [&](int idx, int64_t dt_us) {
            if (dt_us < 0) dt_us = 0;
            const uint32_t dt = static_cast<uint32_t>(dt_us);
            st_sum_us[idx] += dt;
            if (dt > st_max_us[idx]) st_max_us[idx] = dt;
            ++st_n[idx];
        };

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
        uint32_t s_too_small = 0;     // biggest rejected for bbox size
        uint32_t s_too_blurry = 0;    // biggest rejected for focus
        uint32_t s_recognised = 0;    // matched an existing enrollment
        uint32_t s_enrolled = 0;      // new face added
        uint32_t s_spatial_track = 0; // spatial-track override fired
        uint32_t s_enrol_q_skip = 0;  // enrolment blocked by quality floor
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
        //
        // last_sim is an EMA of the matched identity's similarity
        // across consecutive same-track recognitions. The matcher's
        // raw per-frame sim jitters ~0.05-0.10 because of MFN's
        // sensitivity to alignment / lighting microvariation; the
        // EMA gives downstream consumers (match floor, augment gate,
        // logs) a track-level confidence number that doesn't bounce
        // with single-frame noise. Cleared whenever the track itself
        // breaks (bbox moves out of IoU, orient changes, age expires,
        // identity changes), so a new face / new track always starts
        // from scratch and can't inherit confidence from the previous
        // person.
        struct {
            bool     valid       = false;
            int      box[4]      = {0, 0, 0, 0};   // detection-frame coords
            Orient   orient      = Orient::ROT_0;  // bbox is only comparable in same orient
            uint32_t stamp_ms    = 0;
            int      matched_id  = 0;              // 1-based DB id; 0 = nothing matched
            float    last_sim    = 0.0f;           // EMA of recent same-track weighted sim
            // Test #9: per-track EMA of the query embedding itself.
            // Smoothing the embedding (not just the resulting sim)
            // gives the matcher a more stable point in feature
            // space across consecutive frames, which damps the
            // MFN sensitivity to micro-misalignment / blur / focus
            // jitter that survives both the bbox luma normalisation
            // (Test #5) and the sim EMA (Test #7). Cleared on track
            // break to avoid leaking one identity's embedding into
            // the next.
            std::vector<float> ema_feat;
        } recog_cache;
        constexpr float RECOG_SIM_EMA_ALPHA  = 0.6f; // weight on previous sim EMA
        constexpr float RECOG_FEAT_EMA_ALPHA = 0.5f; // weight on new raw embedding

        // Name-banner state: tracks what we last asked banner_render_name
        // to draw so we don't re-rasterise unchanged content every frame.
        // Re-rendering is ~1-2 ms (rasterise + outline dilate); it only
        // needs to run when the recognised name or its quarter-turn
        // angle bucket changes.
        struct {
            char name[FACE_NAME_MAX] = {0};
            int  angle_bucket        = -999;       // round(angle / pi/2)
        } name_banner_cache;

        // Helper: publish the navy-blue name banner for a recognised
        // face, identified by its 1-based DB id. Reads the name out of
        // g_known_faces under the DB lock (released before the
        // expensive rasterise), then only re-runs banner_render_name
        // if the name or its quarter-turn angle bucket actually
        // changed. Refreshes the deadline atomically on every call so
        // the banner stays up while the matcher keeps returning this
        // face. Names that are empty (unset by the web UI) silently
        // skip the banner: only user-named faces get the on-screen
        // overlay.
        auto publish_name_banner = [&](int id_1based, float angle_rad) {
            if (id_1based <= 0) {
                return;
            }
            char local_name[FACE_NAME_MAX] = {0};
            {
                FaceDbLock lock;
                const size_t idx = static_cast<size_t>(id_1based - 1);
                if (idx >= g_known_faces.size()) {
                    return;
                }
                std::strncpy(local_name, g_known_faces[idx].name,
                             FACE_NAME_MAX - 1);
                local_name[FACE_NAME_MAX - 1] = '\0';
            }
            if (local_name[0] == '\0') {
                return;             // unnamed face -- skip banner.
            }

            constexpr float quarter = 3.14159265358979323846f * 0.5f;
            const int bucket = static_cast<int>(roundf(angle_rad / quarter));
            if (bucket != name_banner_cache.angle_bucket ||
                std::strcmp(name_banner_cache.name, local_name) != 0) {
                banner_render_name(local_name,
                                   static_cast<float>(bucket) * quarter);
                // memcpy + explicit null term (rather than strncpy) so
                // -O3 doesn't trip its stringop-truncation analysis,
                // which can't see that local_name is already guaranteed
                // < FACE_NAME_MAX by the source it was copied from.
                const size_t n = std::min(std::strlen(local_name),
                                          (size_t)(FACE_NAME_MAX - 1));
                std::memcpy(name_banner_cache.name, local_name, n);
                name_banner_cache.name[n] = '\0';
                name_banner_cache.angle_bucket = bucket;
            }
            g_name_banner_until_ms.store(
                now_ms() + NAME_BANNER_LINGER_MS,
                std::memory_order_relaxed);
        };

        // Orientation to TRY on this frame. After a successful detection we
        // leave this alone so the next frame uses the same orientation; on a
        // failed detection we may stick on it for a couple more attempts
        // (orient-sticky retry, see ORIENT_STICKY_MISSES below) before
        // advancing to the next of the four.
        Orient try_orient = Orient::ROT_0;

        // Non-ROT_0 orient-lock confirmation. A single fluky detection
        // at a rotated orient can pass the score floor and lock us
        // onto an upside-down / sideways orient, even when the user
        // is actually holding the device upright. Common case: the
        // detector fires on a noisy patch at ROT_180 with a score in
        // the 0.55-0.7 band, banner ends up upside-down, user has to
        // wait for the cycle to recover. Requiring two consecutive
        // frames of agreement at the SAME non-ROT_0 orient before
        // committing the lock filters that out without slowing
        // legitimate rotated tracking by more than one frame. ROT_0
        // needs no confirmation (it's the default / cold-start
        // orient and a wrong lock there is benign), and once we've
        // locked an orient subsequent frames at that same orient go
        // through unconditionally so steady-state tracking is
        // unaffected.
        Orient last_locked_orient = Orient::ROT_0;
        Orient pending_orient     = Orient::ROT_0;
        int    pending_confirm    = 0;

        // Consecutive miss frames in the current detection-loss streak. Used
        // by the orient-sticky retry: a quick blink / occlusion / motion blur
        // is FAR more likely than the user actually rotating the device
        // between frames, so we retry the last-known-good orient before
        // burning frames on the other three buckets. Reset on each success.
        int consecutive_misses = 0;
        // Number of misses we'll stay on the known-good orient before
        // cycling. Keeps us responsive to genuine rotation (worst case
        // ~3-4 detection frames to discover the new orient) while making
        // transient blinks effectively free.
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
            // is refreshed every detection frame inside prep_detect_input(),
            // so this check is cheap (one atomic load + a few
            // comparisons). The asymmetric hysteresis bands stop us
            // from oscillating between presets when a scene's luma
            // sits near a boundary.
            if (now_us >= ae_next_check_us)
            {
                // Face-weighted scene metering. The previous behaviour
                // -- "use face_luma when present, else global_luma" --
                // turned the preset cycle into an oscillator: face
                // present pulled the metering target down (skin in
                // shadow is darker than scene mean), the sensor
                // cranked up gain, the next frame's global luma shot
                // up, and on a frame where the lock dropped or the
                // face-luma reading went stale the AE check read
                // global_luma instead and flipped the preset back.
                // Result: MID <-> BRIGHT every check interval with
                // the face still in frame, visible exposure breathing
                // and noisier alignment / recognition.
                //
                // The InsightFace / Hikvision-style fix is a *weighted
                // average* between face-region and global luma rather
                // than an either/or. Face data is a 30 % bias, not a
                // replacement -- a backlit subject still pulls the
                // target down, but a steady face can't drag global
                // metering into a feedback loop with itself. Same
                // hysteresis bands as before.
                const int  global_luma = g_last_mean_luma.load(
                    std::memory_order_relaxed);
                const int  face_lum    = g_last_face_luma.load(
                    std::memory_order_relaxed);
                const uint32_t face_ms = g_last_face_luma_ms.load(
                    std::memory_order_relaxed);
                const bool face_fresh =
                    face_lum >= 0 &&
                    (now_ms() - face_ms) < FACE_LUMA_STALE_MS;
                constexpr int FACE_AE_WEIGHT_PCT = 30; // out of 100
                int luma = global_luma;
                if (face_fresh) {
                    luma = (FACE_AE_WEIGHT_PCT * face_lum +
                            (100 - FACE_AE_WEIGHT_PCT) * global_luma) / 100;
                }
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
                    // Lock further transitions out for a cooldown window
                    // so the sensor's internal metering can settle on
                    // the new preset before we re-evaluate. Without this
                    // the next check would read luma that's still in
                    // transit and likely flip us back.
                    ae_next_check_us = now_us +
                        (int64_t)AE_TRANSITION_COOLDOWN_MS * 1000;
                } else {
                    ae_next_check_us = now_us +
                        (int64_t)AE_CHECK_INTERVAL_MS * 1000;
                }
            }

            if (now_us >= stats_next_us)
            {
                const float avg_score = s_score_n ?
                    (s_score_sum / (float)s_score_n) : 0.f;

                // Snapshot live state used for the diagnostic readout.
                // Keeps the printf format string self-contained even
                // though half its inputs come from atomics / external
                // structures.
                const int  cur_mean_luma  = g_last_mean_luma.load(
                    std::memory_order_relaxed);
                const int  cur_face_luma  = g_last_face_luma.load(
                    std::memory_order_relaxed);
                const uint32_t face_lum_ms = g_last_face_luma_ms.load(
                    std::memory_order_relaxed);
                const bool face_lum_fresh =
                    cur_face_luma >= 0 &&
                    (now_ms() - face_lum_ms) < FACE_LUMA_STALE_MS;
                size_t   db_size    = 0;
                unsigned total_tpls = 0;
                float    q_min      = 1.0f;
                float    q_max      = 0.0f;
                float    q_sum      = 0.0f;
                {
                    FaceDbLock lock;
                    db_size = g_known_faces.size();
                    for (const auto &kf : g_known_faces) {
                        total_tpls += static_cast<unsigned>(kf.feats.size());
                        for (const auto &tpl : kf.feats) {
                            if (tpl.quality < q_min) q_min = tpl.quality;
                            if (tpl.quality > q_max) q_max = tpl.quality;
                            q_sum += tpl.quality;
                        }
                    }
                }
                const float q_avg = total_tpls
                    ? (q_sum / static_cast<float>(total_tpls)) : 0.0f;
                if (total_tpls == 0) { q_min = 0.0f; }
                static const char *AE_NAMES[] = { "DIM", "MID", "BRIGHT" };
                char face_lum_buf[8];
                if (face_lum_fresh) {
                    std::snprintf(face_lum_buf, sizeof(face_lum_buf),
                                  "%d", cur_face_luma);
                } else {
                    face_lum_buf[0] = '-';
                    face_lum_buf[1] = '\0';
                }

                ESP_LOGI(TAG,
                         "1s: frame=%u skip=%u raw=%u pad=%u kept=%u score_avg=%.2f "
                         "| rej:score=%u small=%u blurry=%u "
                         "| known=%u new=%u sp=%u qskip=%u (db=%u tpl=%u q=%.2f/%.2f/%.2f) "
                         "| ae=%s luma=%d/%s try=%d lock=%d "
                         "(misses=%d pend=%d/2)",
                         (unsigned)s_frames,
                         (unsigned)s_motion_skip,
                         (unsigned)s_raw_dets,
                         (unsigned)s_pad_hits,
                         (unsigned)s_detections,
                         (double)avg_score,
                         (unsigned)s_rej_score,
                         (unsigned)s_too_small,
                         (unsigned)s_too_blurry,
                         (unsigned)s_recognised, (unsigned)s_enrolled,
                         (unsigned)s_spatial_track, (unsigned)s_enrol_q_skip,
                         (unsigned)db_size,
                         total_tpls,
                         (double)q_min, (double)q_avg, (double)q_max,
                         AE_NAMES[(int)ae_preset],
                         cur_mean_luma,
                         face_lum_buf,
                         (int)try_orient,
                         (int)last_locked_orient,
                         consecutive_misses,
                         pending_confirm);
                s_frames = s_motion_skip = s_raw_dets = s_pad_hits = s_detections =
                    s_rej_score = s_too_small = s_too_blurry =
                    s_recognised = s_enrolled = s_spatial_track = s_enrol_q_skip = 0;
                s_score_sum = 0.f;
                s_score_n = 0;

                // Test #10: per-stage timing breakdown. Prints avg/max
                // microseconds and sample count for each instrumented
                // stage. Stages with no samples in this window are
                // omitted from the line.
                {
                    char tbuf[256];
                    int  pos = 0;
                    for (int s = 0; s < ST_COUNT; ++s) {
                        if (st_n[s] == 0) continue;
                        const uint32_t avg = static_cast<uint32_t>(
                            st_sum_us[s] / st_n[s]);
                        const int written = std::snprintf(
                            tbuf + pos, sizeof(tbuf) - pos,
                            " %s=%u/%uus(n=%u)",
                            ST_NAMES[s], (unsigned)avg,
                            (unsigned)st_max_us[s], (unsigned)st_n[s]);
                        if (written <= 0 ||
                            pos + written >= (int)sizeof(tbuf)) {
                            break;
                        }
                        pos += written;
                    }
                    if (pos > 0) {
                        ESP_LOGI(TAG, "1s timing:%s", tbuf);
                    }
                }
                for (int s = 0; s < ST_COUNT; ++s) {
                    st_sum_us[s] = 0;
                    st_max_us[s] = 0;
                    st_n[s]      = 0;
                }
                stats_next_us = now_us + STATS_PERIOD_US;
            }

            // The banner is now an overlay composited into the live
            // preview by the render task, so we don't pause inference
            // while it's up — the detector keeps running, and any new
            // enrolment will simply refresh the overlay's deadline /
            // angle.

            // Cross-task cache invalidation: if face_db_delete() ran
            // since our last iteration, our recog_cache (and the
            // name-banner cache) may be pointing at the wrong row in
            // g_known_faces. Clear them; the next successful match
            // will rebuild both from scratch. acquire-ordered exchange
            // pairs with the release store in face_db_delete().
            if (g_recog_cache_invalidate.exchange(false,
                    std::memory_order_acquire)) {
                recog_cache.valid        = false;
                recog_cache.matched_id   = 0;
                recog_cache.last_sim     = 0.0f;
                recog_cache.ema_feat.clear();
                name_banner_cache.name[0] = '\0';
                name_banner_cache.angle_bucket = -999;
            }

            camera_fb_t *fb = camera_fb_get_square();
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
                scene_is_static(reinterpret_cast<const uint16_t *>(fb->buf), motion))
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
            const uint16_t *src       = reinterpret_cast<const uint16_t *>(fb->buf);
            const uint16_t *to_detect = src;
            if (fb->width == FRAME_DIM && fb->height == FRAME_DIM)
            {
                const int64_t _t0 = esp_timer_get_time();
                to_detect = prep_detect_input(try_orient, src, scratch,
                                              FRAME_DIM);
                record_stage(ST_PREP, esp_timer_get_time() - _t0);
            }

            dl::image::img_t img = {};
            img.data = (void *)to_detect;
            img.width = fb->width;
            img.height = fb->height;
            // Camera outputs RGB565 high-byte-first in memory; ESP-DL's
            // preprocessor consumes that format directly.
            img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE;

            const int64_t _t_det0 = esp_timer_get_time();
            // Note: detect->run is NOT wrapped in render_suspend /
            // render_resume. Test #18 confirmed it would shave ~20 ms
            // off detect:model (from 220 ms to 200 ms) but detect
            // fires every frame, so the LCD would stutter at the
            // detect cadence even when no face is present. Reserve
            // the throttle for feat->run (recognition-frame only)
            // where the savings dominate (440 ms saved per call)
            // and the visible pause is rare and motivated.
            std::list<dl::detect::result_t> &detections = detect->run(img);
            record_stage(ST_DETECT, esp_timer_get_time() - _t_det0);
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
            // Two windows the padded retry is allowed in:
            //   * "sticky lock" -- consecutive_misses < OVERLAY_CLEAR_MISSES,
            //     i.e. the HUD still shows the face as present and we
            //     want to recover the frame the primary skipped.
            //   * "cold start, first miss of this orient" -- there's
            //     no prior lock at all but the orient cycle has just
            //     advanced. This is the case a user who walks up to
            //     the device already in close range hits: the
            //     primary pass can never lock on a face-fills-frame
            //     close-up at ANY of the four orients, so without a
            //     padded retry per orient the cycle would spin
            //     forever. The device can be physically held at any
            //     rotation, so we can't assume the close-range face
            //     is upright -- we have to try padded on whichever
            //     orient we're currently on. Running it only on the
            //     first miss of each orient (consecutive_misses %
            //     ORIENT_STICKY_MISSES == 0) keeps the cold-start
            //     overhead to ~80 ms once per orient (~20 ms/frame
            //     amortised) while still giving close-range first-
            //     contact a way in within one full orient sweep.
            const bool padded_eligible =
                padded_scratch &&
                fb->width == FRAME_DIM && fb->height == FRAME_DIM &&
                (consecutive_misses < OVERLAY_CLEAR_MISSES ||
                 (consecutive_misses % ORIENT_STICKY_MISSES) == 0);
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
                if (pbest && pbest->score >= min_detect_score_for(try_orient)) {
                    // Deep-copy into a local result we own, since the
                    // detector reuses its internal list on the next
                    // call.
                    padded_remap.category = pbest->category;
                    padded_remap.score    = pbest->score;
                    padded_remap.box      = pbest->box;
                    padded_remap.keypoint = pbest->keypoint;
                    remap_padded_result(&padded_remap);
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

            // Far-range upscale fallback. Fires on the opposite phase
            // of the orient cycle from the close-range padded path
            // (offset by ORIENT_STICKY_MISSES / 2 so the two retries
            // are interleaved instead of stacking on the same frame),
            // and reuses padded_scratch as scratch space since the
            // two paths never run on the same frame. Recovers small
            // / far faces (~25-50 px tall) that fall below the
            // detector's smallest anchor at native scale.
            dl::detect::result_t upscale_remap;
            const bool upscale_eligible =
                padded_scratch && !biggest &&
                fb->width == FRAME_DIM && fb->height == FRAME_DIM &&
                (consecutive_misses < OVERLAY_CLEAR_MISSES ||
                 (consecutive_misses % ORIENT_STICKY_MISSES) ==
                     (ORIENT_STICKY_MISSES / 2));
            if (upscale_eligible)
            {
                upscale_into_full(to_detect, padded_scratch);

                dl::image::img_t uimg = {};
                uimg.data     = padded_scratch;
                uimg.width    = FRAME_DIM;
                uimg.height   = FRAME_DIM;
                uimg.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE;

                auto &udetections = detect->run(uimg);
                const dl::detect::result_t *ubest = nullptr;
                int ubest_area = 0;
                for (const auto &d : udetections) {
                    const int a = d.box_area();
                    if (a > ubest_area) {
                        ubest_area = a;
                        ubest = &d;
                    }
                }
                if (ubest && ubest->score >= min_detect_score_for(try_orient)) {
                    upscale_remap.category = ubest->category;
                    upscale_remap.score    = ubest->score;
                    upscale_remap.box      = ubest->box;
                    upscale_remap.keypoint = ubest->keypoint;
                    remap_upscale_result(&upscale_remap);
                    ESP_LOGI(TAG,
                             "upscale fallback hit: score=%.2f "
                             "(far-range face)",
                             (double)upscale_remap.score);
                    biggest = &upscale_remap;
                    s_score_sum += upscale_remap.score;
                    ++s_score_n;
                    // Bucketed under pad_hits in the stats line
                    // -- both are exception-path recoveries.
                    ++s_pad_hits;
                }
            }

            const float score_floor = min_detect_score_for(try_orient);
            if (!biggest || biggest->score < score_floor)
            {
                if (biggest)
                {
                    // Low-confidence detection: don't bother running
                    // the embedder on it, don't update the HUD with a
                    // flaky bbox, and don't lock the orient cycle
                    // onto whatever it landed on.
                    ESP_LOGD(TAG,
                             "drop low-score det: score=%.2f < %.2f (orient=%d)",
                             (double)biggest->score,
                             (double)score_floor,
                             (int)try_orient);
                    ++s_rej_score;
                }
                unknown_streak = 0;
                // A miss votes against any pending non-ROT_0 confirmation:
                // the candidate orient didn't survive two frames in a row,
                // so drop the pending state.
                pending_orient  = Orient::ROT_0;
                pending_confirm = 0;
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

            // Detection passed every gate. If this would lock the
            // orient cycle onto a non-ROT_0 orient that we WEREN'T
            // already locked to, run two extra safety nets before
            // committing -- both targeted at the upside-down /
            // sideways false-positive class that has consistently
            // been the hardest to suppress:
            //
            //   1. ROT_0 cross-check. Re-run the detector on the
            //      SAME source frame at ROT_0. A genuinely-rotated
            //      device produces no ROT_0 hit (the face is rotated
            //      in the raw frame, so the upright-trained model
            //      sees a rotated face and bails). A spurious non-
            //      ROT_0 firing on an actually-upright user, on the
            //      other hand, almost always coexists with a high-
            //      confidence ROT_0 detection -- the user's face is
            //      *also* present and upright in the raw frame.
            //      When ROT_0 produces a confident upright hit we
            //      adopt that as the candidate and proceed at ROT_0,
            //      suppressing the rotated lock entirely. This is
            //      the same idea used by rotation-invariant
            //      detectors (PCN, Shi et al. 2018) -- "calibrate"
            //      ambiguous orient predictions against a
            //      reference orient -- adapted to a frozen-model
            //      pipeline by paying for one extra detector run
            //      only on the rare frame where a non-ROT_0
            //      candidate is actually being considered.
            //
            //   2. Two-frame confirmation (unchanged). If the cross-
            //      check is inconclusive, fall back to the original
            //      "require two consecutive confirming hits at the
            //      same non-ROT_0 orient" gate.
            //
            // ROT_0 locks and same-orient re-locks (already
            // confirmed) bypass both gates.
            dl::detect::result_t rot0_candidate;
            dl::detect::result_t saved_candidate;
            if (try_orient != Orient::ROT_0 &&
                try_orient != last_locked_orient)
            {
                // Deep-copy the non-ROT_0 candidate before the
                // verification run clobbers the detector's internal
                // list.
                saved_candidate.category = biggest->category;
                saved_candidate.score    = biggest->score;
                saved_candidate.box      = biggest->box;
                saved_candidate.keypoint = biggest->keypoint;

                const uint16_t *rot0_input = prep_detect_input(
                    Orient::ROT_0, src, scratch, FRAME_DIM);
                dl::image::img_t r0img = {};
                r0img.data     = (void *)rot0_input;
                r0img.width    = FRAME_DIM;
                r0img.height   = FRAME_DIM;
                r0img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE;

                auto &r0_dets = detect->run(r0img);
                const dl::detect::result_t *r0_best = nullptr;
                int r0_best_area = 0;
                for (const auto &d : r0_dets) {
                    const int a = d.box_area();
                    if (a > r0_best_area) {
                        r0_best_area = a;
                        r0_best = &d;
                    }
                }
                const bool rot0_wins =
                    r0_best &&
                    r0_best->score >= MIN_DETECT_SCORE_ROT0;

                if (rot0_wins) {
                    ESP_LOGI(TAG,
                             "orient cross-check: ROT_0 wins (score=%.2f) "
                             "over orient=%d (score=%.2f) -- suppressing "
                             "rotated lock as upside-down false positive",
                             (double)r0_best->score,
                             (int)try_orient,
                             (double)saved_candidate.score);
                    rot0_candidate.category = r0_best->category;
                    rot0_candidate.score    = r0_best->score;
                    rot0_candidate.box      = r0_best->box;
                    rot0_candidate.keypoint = r0_best->keypoint;
                    biggest         = &rot0_candidate;
                    try_orient      = Orient::ROT_0;
                    to_detect       = rot0_input;
                    pending_orient  = Orient::ROT_0;
                    pending_confirm = 0;
                    // Fall through into the success path at ROT_0.
                } else {
                    // ROT_0 didn't have a confident upright hit, so
                    // the rotated candidate stays a candidate. Restore
                    // it (the detector's list got clobbered by the
                    // verification run above) and run the original
                    // two-frame confirmation gate.
                    biggest = &saved_candidate;
                    if (pending_orient == try_orient) {
                        ++pending_confirm;
                    } else {
                        pending_orient  = try_orient;
                        pending_confirm = 1;
                    }
                    if (pending_confirm < 2) {
                        ESP_LOGD(TAG,
                                 "non-ROT_0 lock confirm %d/2 at orient=%d score=%.2f",
                                 pending_confirm, (int)try_orient,
                                 (double)biggest->score);
                        esp_camera_fb_return(fb);
                        continue;
                    }
                }
            }
            // Detection succeeded in this orientation; lock it in (don't
            // touch try_orient) so the next frame stays single-shot.
            const Orient locked_orient = try_orient;
            last_locked_orient = locked_orient;
            pending_orient     = Orient::ROT_0;
            pending_confirm    = 0;
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

            // Face-priority AE: sample mean luma from the bbox region
            // and publish it so the AE preset cycle can bias sensor
            // exposure on the subject's skin tones instead of the
            // whole-frame mean. This is what closes the loop on
            // "subject in shadow against a sunlit window" -- the
            // global frame mean reads ~bright (because the window
            // dominates) and the AE picks BRIGHT, but the face
            // itself is dim and we actually want DIM (or at least
            // MID) so the embedder has a properly-exposed crop. Sampling on every
            // successful detection keeps the value fresh; AE itself
            // still runs on its own AE_CHECK_INTERVAL_MS cadence
            // with hysteresis to avoid preset thrash.
            {
                const int face_lum = sample_face_luma(
                    to_detect, fb->width, fb->height,
                    biggest->box[0], biggest->box[1],
                    biggest->box[2], biggest->box[3]);
                if (face_lum >= 0) {
                    g_last_face_luma.store(face_lum,
                                           std::memory_order_relaxed);
                    g_last_face_luma_ms.store(now_ms(),
                                              std::memory_order_relaxed);
                }
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
                // Refresh the name banner if this face is named --
                // we skip the embedder on cache hits, so we have to
                // re-publish here to keep the navy overlay alive.
                publish_name_banner(recog_cache.matched_id,
                                    banner_angle_for(locked_orient));
                esp_camera_fb_return(fb);
                continue;
            }

            // Run the embedder. The returned TensorBase is owned by the
            // postprocessor and reused on the NEXT call, so we must copy
            // its floats before doing anything else that might run the
            // model again (which currently we don't, but the copy is
            // ~512 floats so it's not worth the foot-gun).
            //
            // ESPDet's anchor-free postprocessor doesn't produce facial
            // landmarks (its output is bbox + score only). The MFN
            // embedder's FeatImagePreprocessor hard-asserts
            // `landmarks.size() == 10` so it can run a 5-point
            // similarity-transform face alignment. We try to recover
            // a real alignment in two tiers:
            //
            //   * estimate_face_keypoints() runs a cheap classical-CV
            //     dark-blob scan over the bbox's upper "eye band" to
            //     localise the two eye centres, then synthesises
            //     nose + mouth from anthropometric ratios anchored
            //     to the eye line. This gives MFN the rotation +
            //     translation components of its affine transform --
            //     a measurably better starting point for the embedder
            //     than a bbox center-crop.
            //
            //   * If the eye localiser bails (face too small, eyes
            //     too close together / too far apart / too tilted),
            //     fall back to the canonical 112x112 ArcFace template
            //     projected proportionally into the bbox. That's a
            //     scale-and-translate only -- worse than the eye-pair
            //     synthesis but better than nothing.
            //
            // Either way the result feeds MFN through its standard
            // alignment pipeline.
            std::vector<int> aligned_landmarks;
            const std::vector<int> *kp_for_feat = &biggest->keypoint;
            bool kp_from_eye_scan = false;
            if (biggest->keypoint.size() != 10) {
                if (estimate_face_keypoints(to_detect, FRAME_DIM, FRAME_DIM,
                                            biggest->box[0], biggest->box[1],
                                            biggest->box[2], biggest->box[3],
                                            &aligned_landmarks)) {
                    kp_from_eye_scan = true;
                } else {
                    static constexpr float STD_LDKS_112[10] = {
                        38.2946f, 51.6963f,   // left eye
                        73.5318f, 51.5014f,   // right eye
                        56.0252f, 71.7366f,   // nose
                        41.5493f, 92.3655f,   // left mouth
                        70.7299f, 92.2041f,   // right mouth
                    };
                    const int bw = biggest->box[2] - biggest->box[0];
                    const int bh = biggest->box[3] - biggest->box[1];
                    aligned_landmarks.resize(10);
                    for (int i = 0; i < 5; ++i) {
                        aligned_landmarks[2 * i] = biggest->box[0] +
                            static_cast<int>((STD_LDKS_112[2 * i]     / 112.0f) * bw);
                        aligned_landmarks[2 * i + 1] = biggest->box[1] +
                            static_cast<int>((STD_LDKS_112[2 * i + 1] / 112.0f) * bh);
                    }
                }
                kp_for_feat = &aligned_landmarks;
            }
            // Diagnostic: log every recognition's alignment source +
            // eye geometry so we can correlate sim scores with how
            // well the localiser is performing. Cheap (one log line
            // per ~50 ms detection frame).
            if (kp_from_eye_scan) {
                const int le_x = aligned_landmarks[0];
                const int le_y = aligned_landmarks[1];
                const int re_x = aligned_landmarks[2];
                const int re_y = aligned_landmarks[3];
                const int edx  = re_x - le_x;
                const int edy  = re_y - le_y;
                // atan2 in milliradians-ish; we report degrees.
                const float roll_deg =
                    std::atan2(static_cast<float>(edy),
                               static_cast<float>(std::max(1, edx))) *
                    (180.0f / 3.14159265f);
                ESP_LOGI(TAG,
                         "align: eye-scan le=(%d,%d) re=(%d,%d) "
                         "dx=%d roll=%.1fdeg",
                         le_x, le_y, re_x, re_y, edx, (double)roll_deg);
            } else if (biggest->keypoint.size() != 10) {
                ESP_LOGI(TAG, "align: fallback (bbox center-crop)");
            }
            // Per-bbox luminance normalisation (Test #5): stretch
            // [P2,P98] -> [0,255] and apply a mild gamma to pull the
            // face's average luma toward 128. Done in-place on
            // to_detect just before the embedder runs, so both the
            // straight pass and the hflip TTA pass see consistent
            // exposure. Safe to skip silently if the bbox is too
            // small or already too flat -- normalize_face_bbox()
            // returns false in those cases.
            const int64_t _t_norm0 = esp_timer_get_time();
            const bool norm_applied = normalize_face_bbox(
                const_cast<uint16_t *>(to_detect), FRAME_DIM,
                biggest->box[0], biggest->box[1],
                biggest->box[2], biggest->box[3]);
            (void)norm_applied;
            record_stage(ST_NORMALIZE, esp_timer_get_time() - _t_norm0);
            const int64_t _t_feat0 = esp_timer_get_time();
            render_suspend();
            dl::TensorBase *t = feat->run(img, *kp_for_feat);
            render_resume();
            record_stage(ST_FEAT, esp_timer_get_time() - _t_feat0);
            if (!t || t->get_size() != feat_len)
            {
                ESP_LOGW(TAG, "feat model returned unexpected tensor");
                esp_camera_fb_return(fb);
                continue;
            }
            const float *raw_feat = t->get_element_ptr<float>();

            // Test #9: query-side embedding EMA. When this detection
            // is a spatial continuation of the previous frame's
            // recog_cache hit, blend the current MFN embedding with
            // the cached EMA and L2-renormalise. Both inputs are
            // unit-norm so the average lies in the unit ball; the
            // renorm projects it back onto the unit sphere so cosine
            // sim stays in [-1, 1]. On a new track (cache empty,
            // stale, wrong orient, or low IoU) we seed with the raw
            // embedding -- never blend across identity boundaries.
            std::vector<float> query_feat(raw_feat, raw_feat + feat_len);
            const int64_t _t_qema0 = esp_timer_get_time();
            const bool feat_ema_same_track =
                recog_cache.valid &&
                recog_cache.orient == locked_orient &&
                recog_cache.matched_id > 0 &&
                static_cast<int>(recog_cache.ema_feat.size()) == feat_len &&
                (now_ms() - recog_cache.stamp_ms) < RECOG_CACHE_MS &&
                iou_pct(biggest->box.data(), recog_cache.box) >=
                    RECOG_IOU_PCT;
            if (feat_ema_same_track) {
                const float a = RECOG_FEAT_EMA_ALPHA;
                const float b = 1.0f - a;
                double sumsq = 0.0;
                for (int i = 0; i < feat_len; ++i) {
                    const float v = a * raw_feat[i] +
                                    b * recog_cache.ema_feat[i];
                    query_feat[i] = v;
                    sumsq += static_cast<double>(v) * static_cast<double>(v);
                }
                if (sumsq > 0.0) {
                    const float inv_n = static_cast<float>(
                        1.0 / std::sqrt(sumsq));
                    for (int i = 0; i < feat_len; ++i) {
                        query_feat[i] *= inv_n;
                    }
                }
            }
            record_stage(ST_QEMA, esp_timer_get_time() - _t_qema0);
            // take the MAX cosine similarity across its template
            // gallery, so off-angle / profile templates can rescue a
            // match that the front-on template alone would miss. We
            // hold the DB mutex for the duration so a concurrent
            // rename from the web server can't tear the read; the
            // loop is at most MAX_KNOWN_FACES * FACE_MAX_TEMPLATES *
            // feat_len floats (~64 K FLOPs after SIMD), well under a
            // millisecond of critical section.
            //
            // Quality-weighted match score: each candidate template's
            // raw cosine similarity is scaled by a function of its
            // stored quality so low-quality templates have to clear a
            // higher bar to win against higher-quality templates of a
            // different identity. The weight is (Q_FLOOR + (1-Q_FLOOR)
            // * sqrt(quality)) so even a quality=0 template stays at
            // 0.5x (still meaningful, not zeroed out), and a perfect
            // quality=1.0 template gets the full raw sim.
            //
            // This is the embedded variant of the InsightFace
            // quality-aware matching idea (Boutros SER-FIQ 2020,
            // MagFace 2021). With a single identity in the gallery
            // it has no visible effect on `best_sim`; the value
            // shows up once a second person is enrolled and the
            // matcher needs to choose between identities of mixed
            // gallery quality.
            constexpr float QW_FLOOR = 0.5f;
            auto sim_quality_weight = [](float quality) -> float {
                const float q = std::clamp(quality, 0.0f, 1.0f);
                return QW_FLOOR + (1.0f - QW_FLOOR) * std::sqrt(q);
            };
            float best_sim = -2.f;
            float best_sim_raw = -2.f;       // pre-quality-weight sim (for logging + novelty)
            float best_template_sim_raw = -2.f;  // RAW sim of the winning template (novelty check)
            int   best_id = -1;
            int   best_template_idx = -1;
            float second_best_sim = -2.f;    // best weighted sim against any OTHER face
            const int64_t _t_cos0 = esp_timer_get_time();
            {
                FaceDbLock lock;
                for (size_t i = 0; i < g_known_faces.size(); ++i)
                {
                    float face_best = -2.f;
                    float face_best_raw = -2.f;
                    int   face_best_t = -1;
                    for (size_t t = 0; t < g_known_faces[i].feats.size(); ++t) {
                        const float s_raw = cosine_sim(query_feat.data(),
                                                       g_known_faces[i].feats[t].feat.data(),
                                                       feat_len);
                        const float s = s_raw *
                            sim_quality_weight(g_known_faces[i].feats[t].quality);
                        if (s > face_best) {
                            face_best     = s;
                            face_best_raw = s_raw;
                            face_best_t   = (int)t;
                        }
                    }
                    if (face_best > best_sim)
                    {
                        // The previous winner becomes the runner-up.
                        if (best_id > 0 && best_sim > second_best_sim) {
                            second_best_sim = best_sim;
                        }
                        best_sim              = face_best;
                        best_sim_raw          = face_best_raw;
                        best_template_sim_raw = face_best_raw;
                        best_template_idx     = face_best_t;
                        best_id               = (int)i + 1; // 1-based for logging niceness
                    } else if (face_best > second_best_sim) {
                        second_best_sim = face_best;
                    }
                }
            }
            record_stage(ST_COSINE, esp_timer_get_time() - _t_cos0);

            // Track-level EMA over the weighted sim. When this frame
            // looks like a continuation of the previously-matched
            // identity (same recog_cache row), blend with the cached
            // EMA so downstream gates see a smoothed track confidence
            // instead of single-frame jitter. New tracks (cache empty
            // or matched a different identity than before) start
            // from the raw value.
            float sim_track = best_sim;
            if (recog_cache.valid && recog_cache.matched_id == best_id)
            {
                sim_track = RECOG_SIM_EMA_ALPHA * recog_cache.last_sim +
                            (1.0f - RECOG_SIM_EMA_ALPHA) * best_sim;
            }

            // Spatial-track continuation override. When the matcher
            // would otherwise treat this frame as "unknown" but the
            // current bbox is still spatially close to a face we
            // recognised within the last few seconds, force-match
            // against that prior identity instead of starting an
            // enrolment cascade. Common case: user turns the device
            // sideways, the embedding shifts across orient before
            // the gallery has a template at the new pose, and
            // without this branch we'd enrol a fresh "person" every
            // time the orient cycle ratchets to a new lock.
            //
            // The SPATIAL_TRACK_SIM_FLOOR guard is a same-face sanity
            // check on the *raw* sim against the tracked identity's
            // gallery -- random embeddings hover around 0.0, so any
            // value above 0.2 strongly implies it's still the same
            // person. Below that, we let enrolment proceed normally
            // (a genuinely different person at the same bbox should
            // get their own id).
            bool spatial_track_overrode = false;
            if (sim_track < g_tuning.match_thr &&
                recog_cache.valid &&
                recog_cache.matched_id > 0 &&
                (now_ms() - recog_cache.stamp_ms) < SPATIAL_TRACK_MS &&
                iou_pct(biggest->box.data(), recog_cache.box) >=
                    SPATIAL_TRACK_IOU_PCT)
            {
                // Re-score this frame against the tracked identity's
                // gallery only. The general matcher loop above already
                // computed this for the tracked id when it iterated --
                // but it picked the winner by *weighted* sim across
                // all identities. Here we want the *raw* sim against
                // the tracked id specifically so we can decide if it's
                // plausibly the same person (vs a different person who
                // happened to walk into the same bbox).
                float track_raw = -2.f;
                int   track_t   = -1;
                float track_q   = 0.f;
                {
                    FaceDbLock lock;
                    const size_t idx = (size_t)(recog_cache.matched_id - 1);
                    if (idx < g_known_faces.size()) {
                        for (size_t t = 0; t < g_known_faces[idx].feats.size(); ++t) {
                            const float s = cosine_sim(
                                raw_feat,
                                g_known_faces[idx].feats[t].feat.data(),
                                feat_len);
                            if (s > track_raw) {
                                track_raw = s;
                                track_t   = (int)t;
                                track_q   = g_known_faces[idx].feats[t].quality;
                            }
                        }
                    }
                }
                if (track_raw >= SPATIAL_TRACK_SIM_FLOOR)
                {
                    const float qw = 0.5f + 0.5f * std::sqrt(track_q);
                    best_id               = recog_cache.matched_id;
                    best_sim_raw          = track_raw;
                    best_sim              = track_raw * qw;
                    best_template_idx     = track_t;
                    best_template_sim_raw = track_raw;
                    sim_track             = RECOG_SIM_EMA_ALPHA * recog_cache.last_sim +
                                            (1.0f - RECOG_SIM_EMA_ALPHA) * best_sim;
                    spatial_track_overrode = true;
                    ++s_spatial_track;
                    ESP_LOGI(TAG,
                             "spatial-track continue: id=%d iou=%d%% "
                             "raw=%.3f track_sim=%.3f (was best_sim=%.3f)",
                             best_id,
                             iou_pct(biggest->box.data(), recog_cache.box),
                             (double)track_raw, (double)sim_track,
                             (double)(best_sim));
                }
            }

            if (best_id > 0 &&
                (sim_track >= g_tuning.match_thr || spatial_track_overrode))
            {
                // Recognised — no enroll, no banner. We removed the
                // "OH... IT'S YOU" path: recognition is now silent so
                // the only on-screen event is a brand-new face.
                ++s_recognised;
                unknown_streak = 0;
                ESP_LOGI(TAG,
                         "known face: id=%d sim=%.3f (frame=%.3f raw=%.3f t=%d, 2nd=%.3f) focus=%d orient=%d",
                         best_id, (double)sim_track,
                         (double)best_sim, (double)best_sim_raw,
                         best_template_idx,
                         (double)second_best_sim, focus, (int)locked_orient);

                // Auto-augment the gallery for this face when the
                // track-smoothed match is *comfortably* above the
                // threshold (so we know it's the same person across
                // multiple frames, not a single fluky hit) AND no
                // other enrolled face came within FACE_AUGMENT_MARGIN
                // of the winner (so we can't be appending a template
                // from someone who happens to look similar to two
                // people). The new embedding must also be meaningfully
                // different from every existing template in this
                // face's gallery (best_template_sim_raw <
                // FACE_AUGMENT_NOVEL, checked on the *raw* sim because
                // near-duplicate detection is a property of the
                // embedding alone, independent of template quality),
                // otherwise it's a near-duplicate that just wastes
                // gallery slots and matcher cycles. The augment gate
                // is intentionally *tighter* than the match gate;
                // weakening recognition for off-angle faces happens
                // exclusively through more templates, never through
                // a lower match floor.
                const float augment_floor = g_tuning.match_thr +
                                            FACE_AUGMENT_MARGIN;
                const bool augment_ok =
                    sim_track             >= augment_floor &&
                    best_template_sim_raw <  FACE_AUGMENT_NOVEL &&
                    second_best_sim       <  best_sim - FACE_AUGMENT_MARGIN;
                if (augment_ok)
                {
                    const float new_q = face_template_quality(
                        biggest->score, focus, bw, bh);
                    FaceDbLock lock;
                    const size_t idx = (size_t)(best_id - 1);
                    if (idx < g_known_faces.size())
                    {
                        auto &gallery = g_known_faces[idx].feats;
                        if (gallery.size() < FACE_MAX_TEMPLATES) {
                            // Free slot -- straight append.
                            FaceTemplate tpl;
                            tpl.feat.assign(raw_feat, raw_feat + feat_len);
                            tpl.quality = new_q;
                            gallery.emplace_back(std::move(tpl));
                            ESP_LOGI(TAG,
                                     "gallery+: id=%d slot=%u q=%.2f (added sim=%.3f, novel=%.3f)",
                                     best_id,
                                     (unsigned)(gallery.size() - 1),
                                     (double)new_q,
                                     (double)best_sim,
                                     (double)best_template_sim_raw);
                        } else {
                            // Gallery full: find the lowest-quality
                            // slot and evict it only if the incoming
                            // template is genuinely higher quality.
                            // Refuses to evict slot 0 (the original
                            // enrolment template) so the user-facing
                            // thumbnail's identity stays anchored even
                            // after several auto-augments.
                            int   worst_slot = -1;
                            float worst_q    = new_q;
                            for (size_t t = 1; t < gallery.size(); ++t) {
                                if (gallery[t].quality < worst_q) {
                                    worst_q    = gallery[t].quality;
                                    worst_slot = (int)t;
                                }
                            }
                            if (worst_slot >= 0) {
                                const float evicted_q =
                                    gallery[worst_slot].quality;
                                gallery[worst_slot].feat.assign(
                                    raw_feat, raw_feat + feat_len);
                                gallery[worst_slot].quality = new_q;
                                ESP_LOGI(TAG,
                                         "gallery~: id=%d slot=%d q=%.2f evict@q=%.2f (sim=%.3f, novel=%.3f)",
                                         best_id, worst_slot,
                                         (double)new_q,
                                         (double)evicted_q,
                                         (double)best_sim,
                                         (double)best_template_sim_raw);
                            } else {
                                ESP_LOGD(TAG,
                                         "gallery=: id=%d full and new q=%.2f "
                                         "below all existing slots, skipped",
                                         best_id, (double)new_q);
                            }
                        }
                    }
                }

                // Cache this hit so the next few frames at roughly
                // the same bbox can skip the embedder pass.
                recog_cache.valid      = true;
                recog_cache.box[0]     = biggest->box[0];
                recog_cache.box[1]     = biggest->box[1];
                recog_cache.box[2]     = biggest->box[2];
                recog_cache.box[3]     = biggest->box[3];
                recog_cache.orient     = locked_orient;
                recog_cache.stamp_ms   = now_ms();
                recog_cache.matched_id = best_id;
                recog_cache.last_sim   = sim_track;
                // Persist the (possibly blended) query embedding so
                // next frame's same-track check can keep smoothing.
                recog_cache.ema_feat   = query_feat;

                // Show the navy name banner along the bottom of the
                // preview if this face has been named in the web UI.
                publish_name_banner(best_id,
                                    banner_angle_for(locked_orient));

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

            // Enrolment quality floor. Refuse to seed the DB with a
            // template that's below FACE_ENROL_MIN_QUALITY -- a low-
            // quality first template (small bbox, blurry, low detection
            // score) makes the gallery brittle to pose changes and
            // bootstraps the multi-enrolment cascade the spatial-track
            // continuation above is the second line of defence against.
            // The frame still tracks normally; we just don't commit it
            // to the DB. unknown_streak is preserved so a subsequent
            // higher-quality frame can still enrol immediately.
            const float candidate_q = face_template_quality(
                biggest->score, focus, bw, bh);
            if (candidate_q < FACE_ENROL_MIN_QUALITY) {
                ++s_enrol_q_skip;
                ESP_LOGD(TAG,
                         "skip enroll: q=%.2f below %.2f floor "
                         "(score=%.2f focus=%d bbox=%dx%d)",
                         (double)candidate_q,
                         (double)FACE_ENROL_MIN_QUALITY,
                         (double)biggest->score, focus, bw, bh);
                esp_camera_fb_return(fb);
                continue;
            }

            // Commit the enrollment. Copy the float* into the known list
            // (the source buffer is recycled by the model on the next run).
            // Every successful enrollment fires the "NEW FACE DETECTED"
            // overlay, refreshing both the deadline and the rotation angle
            // so the text always reads upright relative to the current
            // face pose.
            //
            // Horizontal-flip TTA on the enrolment template only.
            // Standard pattern from FaceNet 2015 onwards (used by
            // InsightFace, Google Coral demo, OAK-D pipelines etc.):
            // embed the frame, then embed its horizontal mirror, then
            // average + L2-renormalise. The resulting "anchor"
            // template is meaningfully more pose-symmetric than a
            // single forward pass, which especially helps when the
            // enrolment pose is slightly off-centre (head turned a
            // few degrees) -- the flipped version balances out the
            // bias. Cost: one extra MFN inference per enrolment
            // (~50 ms, paid exactly once per identity).
            //
            // Falls back to the straight `raw_feat` if `padded_scratch`
            // failed to allocate at startup, or if the embedder's
            // second call doesn't produce the expected tensor.
            std::vector<float> enrol_feat(raw_feat, raw_feat + feat_len);
            bool tta_applied = false;
            if (padded_scratch && kp_for_feat &&
                kp_for_feat->size() == 10)
            {
                // 1. Hflip image: dst[y][x] = src[y][N-1-x].
                const uint16_t *src16 = to_detect;
                uint16_t       *dst16 = padded_scratch;
                for (int y = 0; y < FRAME_DIM; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * FRAME_DIM;
                    for (int x = 0; x < FRAME_DIM; ++x) {
                        dst16[row_base + x] =
                            src16[row_base + (FRAME_DIM - 1 - x)];
                    }
                }
                // 2. Hflip landmarks: swap LE<->RE and LM<->RM (their
                //    "left vs right" labels in our (x-low, x-high) sense
                //    invert when the image flips), flip the x of nose.
                //    Layout: [LE.x, LE.y, RE.x, RE.y, N.x, N.y, LM.x,
                //    LM.y, RM.x, RM.y].
                const int N1 = FRAME_DIM - 1;
                const std::vector<int> &kp = *kp_for_feat;
                std::vector<int> flipped_kp(10);
                flipped_kp[0] = N1 - kp[2];   // was RE
                flipped_kp[1] = kp[3];
                flipped_kp[2] = N1 - kp[0];   // was LE
                flipped_kp[3] = kp[1];
                flipped_kp[4] = N1 - kp[4];   // nose, x flipped
                flipped_kp[5] = kp[5];
                flipped_kp[6] = N1 - kp[8];   // was RM
                flipped_kp[7] = kp[9];
                flipped_kp[8] = N1 - kp[6];   // was LM
                flipped_kp[9] = kp[7];

                // 3. Run the embedder again on the flipped frame.
                //    NOTE: this invalidates `raw_feat` (the original
                //    tensor is recycled by MFN's postprocessor on the
                //    next call), so we copied it into enrol_feat above.
                dl::image::img_t flip_img = img;
                flip_img.data = padded_scratch;
                render_suspend();
                dl::TensorBase *t2 = feat->run(flip_img, flipped_kp);
                render_resume();
                if (t2 && t2->get_size() == feat_len) {
                    const float *flip_feat = t2->get_element_ptr<float>();
                    // 4. Average + L2-renormalise. Both embeddings are
                    //    already unit-norm so the average sits inside
                    //    the unit ball; we re-project to the unit
                    //    sphere so cosine sim against gallery templates
                    //    stays bounded in [-1, 1].
                    double sumsq = 0.0;
                    for (int i = 0; i < feat_len; ++i) {
                        enrol_feat[i] = 0.5f * (enrol_feat[i] + flip_feat[i]);
                        sumsq += static_cast<double>(enrol_feat[i]) *
                                 static_cast<double>(enrol_feat[i]);
                    }
                    if (sumsq > 0.0) {
                        const float inv_norm = static_cast<float>(
                            1.0 / std::sqrt(sumsq));
                        for (int i = 0; i < feat_len; ++i) {
                            enrol_feat[i] *= inv_norm;
                        }
                        tta_applied = true;
                    }
                }
            }
            if (!tta_applied) {
                ESP_LOGD(TAG,
                         "enrol: hflip TTA skipped (padded_scratch=%p, kp.size=%u)",
                         (const void *)padded_scratch,
                         (unsigned)(kp_for_feat ? kp_for_feat->size() : 0));
            }

            bool hit_cap = false;
            size_t new_count = 0;
            {
                FaceDbLock lock;
                if (g_known_faces.size() >= MAX_KNOWN_FACES) {
                    hit_cap = true;
                } else {
                    KnownFace entry;
                    FaceTemplate tpl;
                    tpl.feat = std::move(enrol_feat);
                    tpl.quality = candidate_q;
                    entry.feats.emplace_back(std::move(tpl));
                    crop_face_thumb(src, biggest, locked_orient, &entry.thumb);
                    entry.name[0]    = '\0';
                    entry.enrolled_ms = now_ms();
                    g_known_faces.emplace_back(std::move(entry));
                    new_count = g_known_faces.size();
                }
            }

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
                recog_cache.valid      = true;
                recog_cache.box[0]     = biggest->box[0];
                recog_cache.box[1]     = biggest->box[1];
                recog_cache.box[2]     = biggest->box[2];
                recog_cache.box[3]     = biggest->box[3];
                recog_cache.orient     = locked_orient;
                recog_cache.stamp_ms   = now_ms();
                // 1-based id of the entry we just appended. Used by
                // the cache-hot path so that when the user names this
                // face from the web UI, the next recognised frame can
                // pull the name out of the DB without needing to run
                // the embedder again.
                recog_cache.matched_id = static_cast<int>(new_count);
                // Seed the per-track sim EMA at the augment floor
                // (match_thr + FACE_AUGMENT_MARGIN) instead of zero
                // so the very first post-enrolment frame doesn't dip
                // below match_thr just because the EMA hadn't built
                // any history. With seed=0 the first frame's
                // sim_track collapses to 0.4*best_sim, which can land
                // at ~0.29 (below the 0.38 MID floor) for an
                // otherwise-high-quality match -- and we'd lean on
                // the spatial-track override to rescue it. Seeding at
                // the augment floor matches what we'd compute if the
                // matcher had already considered this identity "just
                // recognised at the floor".
                recog_cache.last_sim   = g_tuning.match_thr + FACE_AUGMENT_MARGIN;

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
                    // Banner rotation = orient-cycle's locked orientation.
                    // Discrete quarter-turn from the orient lock; see
                    // banner_angle_for() for the convention.
                    const float roll = banner_angle_for(locked_orient);
                    banner_render(roll);
                    g_banner_until_ms.store(now_ms() + BANNER_HOLD_MS,
                                            std::memory_order_relaxed);

                    ESP_LOGI(TAG,
                             "NEW face enrolled id=%u (total=%u, bbox=%dx%d, focus=%d, q=%.2f, orient=%d, roll=%.0fdeg, tta=%s)",
                             (unsigned)new_count,
                             (unsigned)new_count,
                             bw, bh, focus, (double)candidate_q, (int)locked_orient,
                             (double)(roll * 180.f / 3.14159265f),
                             tta_applied ? "yes" : "no");
                }
                else
                {
                    ESP_LOGI(TAG,
                             "supplementary enrol id=%u (banner already up; total=%u, bbox=%dx%d, focus=%d, q=%.2f, tta=%s)",
                             (unsigned)new_count,
                             (unsigned)new_count,
                             bw, bh, focus, (double)candidate_q,
                             tta_applied ? "yes" : "no");
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

bool face_ai_name_banner_active(void)
{
    return g_name_banner_until_ms.load(std::memory_order_relaxed) > now_ms();
}

bool face_ai_inference_busy(void)
{
    return g_inference_busy.load(std::memory_order_relaxed);
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
    FaceDbLock lock;
    return static_cast<int>(g_known_faces.size());
}

bool face_db_get_entry(int idx, face_db_entry_t *out)
{
    if (!out || !g_db_mux) {
        return false;
    }
    FaceDbLock lock;
    if (idx < 0 || idx >= static_cast<int>(g_known_faces.size())) {
        return false;
    }
    const auto &kf   = g_known_faces[idx];
    out->idx         = idx;
    out->enrolled_ms = kf.enrolled_ms;
    out->thumb_w     = FACE_THUMB_DIM;
    out->thumb_h     = FACE_THUMB_DIM;
    std::memcpy(out->name, kf.name, FACE_NAME_MAX);
    return true;
}

bool face_db_copy_thumb(int idx, uint16_t *dst, size_t dst_capacity_pixels)
{
    if (!dst || !g_db_mux ||
        dst_capacity_pixels < static_cast<size_t>(FACE_THUMB_DIM) * FACE_THUMB_DIM) {
        return false;
    }
    FaceDbLock lock;
    if (idx < 0 || idx >= static_cast<int>(g_known_faces.size())) {
        return false;
    }
    const auto &kf = g_known_faces[idx];
    if (kf.thumb.size() != static_cast<size_t>(FACE_THUMB_DIM) * FACE_THUMB_DIM) {
        return false;
    }
    std::memcpy(dst, kf.thumb.data(), kf.thumb.size() * sizeof(uint16_t));
    return true;
}

bool face_db_set_name(int idx, const char *name)
{
    if (!name || !g_db_mux) {
        return false;
    }
    FaceDbLock lock;
    if (idx < 0 || idx >= static_cast<int>(g_known_faces.size())) {
        return false;
    }
    auto &kf = g_known_faces[idx];
    std::strncpy(kf.name, name, FACE_NAME_MAX - 1);
    kf.name[FACE_NAME_MAX - 1] = '\0';
    return true;
}

bool face_db_delete(int idx)
{
    if (!g_db_mux) {
        return false;
    }
    {
        FaceDbLock lock;
        if (idx < 0 || idx >= static_cast<int>(g_known_faces.size())) {
            return false;
        }
        g_known_faces.erase(g_known_faces.begin() + idx);
    }
    // Released the DB lock before raising the cache-invalidation flag;
    // the AI task picks it up on its next loop iteration. release
    // ordering pairs with the acquire exchange on the reader side.
    g_recog_cache_invalidate.store(true, std::memory_order_release);
    return true;
}

// Close out the Test #11 -O3 + unroll-loops scope.
#pragma GCC pop_options
