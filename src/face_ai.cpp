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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
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
    // Two floors, gated by the orient the detector ran in -- see the
    // `Orient` enum + `min_detect_score_for()` further down:
    //
    //   * `MIN_DETECT_SCORE_ROT0` is loose enough to admit detections
    //     whose confidence drops a notch -- the common case being
    //     glasses-wearers (lens reflections / refraction reliably
    //     knock ~0.05 off the score in our experience), but also
    //     low-light and close-range / out-of-distribution faces.
    //
    //   * `MIN_DETECT_SCORE_ROTATED` is stricter, used for the three
    //     non-default orients. The cost of accepting a weak detection
    //     in a wrong orient is high (the cycle locks there and the
    //     on-screen banner now reads upside-down or sideways relative
    //     to the real head pose); the cost of demanding more evidence
    //     is just one or two more cycles around the orient wheel.
    //     Legitimate rotated detections (user holding the device
    //     sideways) routinely score 0.7-0.9 so this floor doesn't
    //     gate genuine use.
    //
    // The orient-cycle's other gates -- keypoints_look_upright() and
    // the IoU recognition cache -- catch the remaining false-positive
    // classes (upside-down faces with mis-labelled keypoints, blob
    // misfires on textured backgrounds).
    constexpr float MIN_DETECT_SCORE_ROT0    = 0.35f;
    constexpr float MIN_DETECT_SCORE_ROTATED = 0.55f;

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
    constexpr size_t FACE_MAX_TEMPLATES   = 4;
    constexpr float  FACE_AUGMENT_MARGIN  = 0.10f;  // above match_thr
    constexpr float  FACE_AUGMENT_NOVEL   = 0.90f;  // template counts as new if best-existing-sim < this
    struct KnownFace
    {
        // Gallery of L2-normalised embeddings. feats[0] is the
        // enrolment template; feats[1..] are auto-augmented templates
        // captured from high-confidence matches at different head
        // poses. feat_len is invariant across the gallery.
        std::vector<std::vector<float>> feats;
        std::vector<uint16_t> thumb;        // FACE_THUMB_DIM^2 RGB565BE pixels
        char                  name[FACE_NAME_MAX] = {0};
        uint32_t              enrolled_ms = 0;
    };
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
    //   1. rotate_to_scratch() -- 0/90/180/270 deg pre-rotation into a
    //      scratch buffer so the detector (trained on upright faces
    //      only) sees an upright input. ROT_0 fast-paths to the S3
    //      SIMD memcpy; the rotated cases write non-sequentially.
    //
    //   2. apply_clahe() -- contrast-limited adaptive histogram
    //      equalisation on the rotated buffer (luma channel only,
    //      delta-applied to RGB). Skipped in BRIGHT AE mode where it
    //      would either be a no-op or risk over-stretching noise in
    //      already-balanced tiles; a tiny subsampled mean-luma scan
    //      stands in there so the adaptive-AE check still has a
    //      live value to read.
    //
    //   3. HumanFaceDetect::run() -- MSR+MNP cascade from ESP-DL.
    //
    //   4. Close-range padded fallback -- if (3) returned nothing but
    //      the user is still likely in frame, shrink the prepared
    //      buffer into a centred inner box of a mid-grey frame and
    //      re-run the detector. Recovers face-fills-the-frame poses
    //      that the model misses at native scale.
    //
    //   5. App-level gates: MIN_DETECT_SCORE, keypoints_look_upright(),
    //      MIN_FACE_PX, MIN_SHARPNESS.
    //
    //   6. Recognition cache (IoU-keyed) -- if the latest detection's
    //      bbox overlaps the previous matched face strongly, skip the
    //      ~50 ms embedder pass and treat it as a continuation.
    //
    //   7. HumanFaceFeat::run() + cosine-similarity sweep over the
    //      known-face DB (SIMD via dsps_dotprod_f32).
    //
    //   8. Optional enrol (if no match) or name-banner publish (if
    //      matched against a face that has a user-set name).

    // Pure rotate -- the CLAHE pass that follows handles all the
    // contrast work, so this function's only job is to land the
    // upright copy into `dst`. ROT_0 fast-paths to the S3 SIMD
    // memcpy; the rotated cases are scalar by necessity (non-
    // sequential stores break SIMD).
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
    // CLAHE (Contrast Limited Adaptive Histogram Equalization)
    // ---------------------------------------------------------------
    //
    // Splits the frame into a grid of tiles, equalises each tile's
    // luma histogram independently, and applies the per-tile LUT to
    // each pixel using bilinear interpolation between the four
    // surrounding tile centres (so there are no visible tile
    // boundaries). The per-tile clip-limit caps how much any single
    // histogram bin can contribute to its CDF -- this is the bit that
    // keeps the equalisation from amplifying sensor noise in flat
    // regions, the failure mode that vanilla histogram equalisation
    // has in low light.
    //
    // Layout / numbers tuned for the 240x240 detector input:
    //
    //   - 4 x 4 tile grid  -> 60 x 60 pixels per tile. Coarser than
    //     OpenCV's default 8x8 but cheap (12 KB of static state
    //     instead of 48 KB) and still adapts well to top-vs-bottom
    //     and left-vs-right lighting differences that matter to
    //     face detection.
    //   - clip limit       -> adaptive via g_tuning.clahe_clip_lim
    //                         (7 % DIM, 5 % MID, disabled in BRIGHT).
    //                         See the Tuning struct further down.
    //   - luma proxy       -> G6 expanded to 8 bits; used for both
    //                         the histogram bin index and the LUT
    //                         lookup (the two must match -- earlier
    //                         iterations of this code mismatched them
    //                         and silently looked up the wrong bin).
    //   - apply            -> luma-only equalisation, with the delta
    //                         (Y_out - Y_in) added to each of R/G/B.
    //                         Same delta on all channels preserves
    //                         chroma (face skin tones don't drift).
    //                         One bilinear interpolation per pixel
    //                         instead of three.

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

    // Latest inter-tile luma spread (max tile mean - min tile mean)
    // from the 4x4 CLAHE grid. Written by apply_clahe(), exposed as
    // an atomic so the per-second stats line can log it (useful for
    // tuning the half-lit / backlit CLAHE boost path). 0 means "all
    // tiles roughly equal" (flat lighting); 60+ means "harsh
    // directional lighting".
    std::atomic<int> g_last_luma_spread{0};

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
        /*clahe_clip_lim*/  TILE_PX_HELPER * 9 / 100,   // 324 / 3600 = 9%
    };
    constexpr Tuning TUNING_MID = {
        0.38f,
        1,
        TILE_PX_HELPER * 5 / 100,                       // 180 / 3600 = 5%
    };
    constexpr Tuning TUNING_BRIGHT = {
        0.45f,
        1,
        // Bumped from 2 % -> 4 %. 2 % was the gentlest "this still
        // does useful work" clip, but on backlit scenes (subject
        // in front of a sunlit window) it left visible shadow
        // crush on the face. 4 % digs noticeably more out of those
        // shadows without amplifying sensor noise in the already-
        // flat bright regions, because CLAHE's clip is a per-tile
        // cap -- a tile that's already evenly bright has no excess
        // to redistribute regardless of the cap value.
        //
        // Earlier this slot was 0 (CLAHE disabled outright). The
        // theory at the time -- that the upstream ESP-DL example
        // feeds raw frames and works fine in bright light, so we
        // could too -- turned out to be conservative; in practice
        // re-enabling at a low clip recovered noticeable bright-
        // light detections without observable downside.
        TILE_PX_HELPER * 4 / 100,                       // 144 / 3600 = 4%
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

    // The CLAHE apply pass is the hottest 57,600-pixel inner loop
    // in the project. Marking it `hot` plus per-function `O3` lets
    // the compiler unroll the inner kernel more aggressively and
    // -- on this GCC version with the Xtensa LX7 backend -- emit
    // tighter pipelined sequences for the bilinear + clip math.
    //
    // Single-pass design: the histogram build and the LUT apply are
    // fused into one buffer read/modify/write loop. The applied LUT
    // is the *previous* frame's; the histogram we build during this
    // pass produces the LUT we'll apply on the *next* frame. That
    // costs us one frame of LUT staleness on a hard scene change
    // (invisible at the detector's ~10 FPS rate) and saves a full
    // ~115 KB pass over the internal-SRAM scratch each call.
    //
    // Luma proxy is the G6 channel expanded to 8 bits, used for both
    // the histogram index and the LUT lookup. Earlier versions of
    // this code mismatched the two -- hist used G-only, apply used
    // 0.25R + 0.5G + 0.25B -- so the LUT entry being looked up did
    // not actually correspond to the bin that pixel had voted into.
    // Using G everywhere fixes that latent inconsistency and is also
    // measurably faster (no R5/B5 unpack on the luma path).
    __attribute__((hot, optimize("O3")))
    void apply_clahe(uint16_t *pixels) noexcept
    {
        // hist[t]   : per-tile 256-bin G6-derived luma histogram for
        //             the CURRENT frame. Rebuilt every call.
        // lut[t]    : per-tile 256-entry LUT. Read as the "apply" LUT
        //             during the fused pass (built from the PREVIOUS
        //             frame's hist), then overwritten with the new
        //             LUT built from this frame's hist for next time.
        // Both static -> internal SRAM (BSS), no malloc/free per call.
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

        // First-call LUT initialisation. Without this, the very first
        // frame would apply uninitialised (zero) LUTs and the detector
        // would see a solid-black image. Identity LUT means the first
        // frame passes through unchanged; from frame 2 on we have a
        // properly-built CLAHE LUT from the previous frame's stats.
        static bool lut_initialised = false;
        if (!lut_initialised) {
            for (int t = 0; t < CLAHE_N * CLAHE_N; ++t) {
                for (int v = 0; v < 256; ++v) {
                    lut[t][v] = static_cast<uint8_t>(v);
                }
            }
            lut_initialised = true;
        }

        // ---------------------------------------------------------------
        // Adaptive gamma (auto-tone) -- fused into the RGB565->RGB888
        // byte expansion below as three tiny LUTs (32 + 64 + 32 entries
        // = 128 bytes), so it costs nothing extra in the hot loop and
        // composes cleanly in front of the existing per-tile CLAHE.
        //
        // Rationale -- why a global tone curve in addition to CLAHE:
        //
        // CLAHE is a *local* contrast equaliser; it does nothing to the
        // *global* midpoint of the image. When the sensor's AE under-
        // exposes (lamp-lit room, mean luma drifting to ~70) or over-
        // exposes (subject in front of a window, mean ~200) CLAHE
        // still has to work against a midtone that's far from the
        // detector's training distribution.
        //
        // A power-law gamma curve is the cheapest way to retarget the
        // global midpoint to 128. We solve gamma so that
        //
        //     255 * (prev_mean/255)^gamma == 128
        //
        // which gives gamma = log(128/255) / log(prev_mean/255).
        // For prev_mean = 70 this is gamma ~= 0.53 (brightens shadows);
        // for prev_mean = 200 it's gamma ~= 2.86 (compresses highlights).
        // We clamp to [0.55, 1.7] so we never apply a transform so
        // extreme that the sensor's noise gets amplified beyond what
        // the detector tolerates -- this clamp range is empirically
        // similar to what's used in Tan & Triggs' face-recognition
        // preprocessing (2007), where gamma=0.5 is recommended as the
        // first stage of their illumination-normalisation pipeline.
        //
        // The LUT is folded INTO the RGB565->RGB888 byte expansion the
        // CLAHE loop already does for each pixel, so the inner-loop
        // cost change is "three LUT loads vs three shift+OR pairs":
        // a wash on the S3, and the LUTs are small enough to stay hot
        // in I/D cache for the full frame.
        //
        // Refs:
        //  - Tan, X. & Triggs, B., "Enhanced Local Texture Feature Sets
        //    for Face Recognition Under Difficult Lighting Conditions",
        //    IEEE TIP 2010 (gamma -> DoG -> contrast eq.).
        //  - Huang, S.-C. et al., "Efficient Contrast Enhancement Using
        //    Adaptive Gamma Correction With Weighting Distribution",
        //    IEEE TIP 2013 (auto-gamma from histogram CDF).
        //  - Pizer, S. et al., "Adaptive Histogram Equalization and Its
        //    Variations", CVGIP 1987 (CLAHE itself).
        static uint8_t gamma_r5[32];
        static uint8_t gamma_g6[64];
        static uint8_t gamma_b5[32];
        static float   last_gamma = 0.f;
        {
            const int prev_mean = g_last_mean_luma.load(std::memory_order_relaxed);
            constexpr int   GAMMA_TARGET = 128;
            constexpr float GAMMA_MIN    = 0.55f;
            constexpr float GAMMA_MAX    = 1.70f;
            float target_gamma = 1.0f;
            if (prev_mean >= 10 && prev_mean <= 245 &&
                prev_mean != GAMMA_TARGET)
            {
                const float num = std::log(GAMMA_TARGET / 255.0f);
                const float den = std::log(prev_mean / 255.0f);
                target_gamma = std::clamp(num / den, GAMMA_MIN, GAMMA_MAX);
            }
            // Rebuild only on meaningful drift -- powf is ~ 100 ns per
            // call on the S3 FPU, and rebuilding 128 entries every
            // frame would burn ~ 13 us per frame for no visible benefit.
            if (last_gamma == 0.f ||
                std::fabs(target_gamma - last_gamma) > 0.04f)
            {
                last_gamma = target_gamma;
                const float inv_255 = 1.0f / 255.0f;
                for (int i = 0; i < 32; ++i) {
                    const int   v8 = (i << 3) | (i >> 2);
                    const float g  = std::pow(v8 * inv_255, target_gamma);
                    const int   out = std::clamp(
                        static_cast<int>(g * 255.0f + 0.5f), 0, 255);
                    gamma_r5[i] = static_cast<uint8_t>(out);
                    gamma_b5[i] = static_cast<uint8_t>(out);
                }
                for (int i = 0; i < 64; ++i) {
                    const int   v8 = (i << 2) | (i >> 4);
                    const float g  = std::pow(v8 * inv_255, target_gamma);
                    const int   out = std::clamp(
                        static_cast<int>(g * 255.0f + 0.5f), 0, 255);
                    gamma_g6[i] = static_cast<uint8_t>(out);
                }
            }
        }

        dsps_memset_aes3(hist, 0, sizeof(hist));

        // ----- Fused pass: apply prev-frame LUT + build this-frame hist -----
        //
        // Hot-loop structure:
        //
        //  * The 240-pixel inner row is split into 8 fixed segments at
        //    the union of the LUT-column boundaries (tx0 changes at
        //    x = 30, 90, 150, 210) and the histogram-tile boundaries
        //    (tile_idx_tab[x] changes at x = 60, 120, 180). Inside
        //    every segment BOTH (tx0, tx1) AND the histogram column-
        //    tile index are constant -- so the four LUT base
        //    pointers (l00..l11) AND the histogram row pointer are
        //    loop-invariants, hoisted out of the pixel loop. Removes
        //    one per-pixel branch + three per-pixel table lookups
        //    versus the old "lazily refresh on tx0 change" structure.
        //
        //  * Hint to GCC that the pixel buffer is 16-byte aligned
        //    (we allocate via heap_caps_aligned_alloc(16, ...)). The
        //    Xtensa PIE has 128-bit aligned-load / store extensions;
        //    this hint lets the backend reach for them in the
        //    surrounding straight-line code without a runtime
        //    alignment check on every load.
        struct ClaheXSeg { int x_start, x_end, tx0, tx1, col_tile; };
        // Constants are derived from CLAHE_N=4 / CLAHE_TILE_SZ=60.
        // tx0 boundaries land at the centre of each pair of tiles
        // (x = 30, 90, 150, 210); column-tile boundaries land at the
        // start of each tile (x = 60, 120, 180). Their union forms
        // these 8 segments of constant LUT + hist indices.
        static constexpr ClaheXSeg X_SEGS[8] = {
            {   0,  30, 0, 1, 0 },
            {  30,  60, 0, 1, 0 },
            {  60,  90, 0, 1, 1 },
            {  90, 120, 1, 2, 1 },
            { 120, 150, 1, 2, 2 },
            { 150, 180, 2, 3, 2 },
            { 180, 210, 2, 3, 3 },
            { 210, 240, 3, 3, 3 },
        };

        uint8_t *__restrict__ w8 =
            static_cast<uint8_t *>(__builtin_assume_aligned(pixels, 16));

        for (int y = 0; y < FRAME_DIM; ++y) {
            // Per-row vertical tile indices and Q8 fraction.
            const int yc = y - CLAHE_TILE_SZ / 2;
            int ty0 = yc / CLAHE_TILE_SZ;
            int ty_frac = yc - ty0 * CLAHE_TILE_SZ;
            if (yc < 0) { ty0 = 0; ty_frac = 0; }
            int ty1 = ty0 + 1;
            if (ty1 >= CLAHE_N) { ty1 = CLAHE_N - 1; ty_frac = 0; }
            const int fy           = (ty_frac * 256) / CLAHE_TILE_SZ;
            const int fy_inv       = 256 - fy;
            const int row_base     = y * FRAME_DIM;
            const int row_ty0_base = ty0 * CLAHE_N;
            const int row_ty1_base = ty1 * CLAHE_N;
            const int hist_row_base = tile_idx_tab[y] * CLAHE_N;

            for (const ClaheXSeg &seg : X_SEGS) {
                const uint8_t *__restrict__ l00 = lut[row_ty0_base + seg.tx0];
                const uint8_t *__restrict__ l01 = lut[row_ty0_base + seg.tx1];
                const uint8_t *__restrict__ l10 = lut[row_ty1_base + seg.tx0];
                const uint8_t *__restrict__ l11 = lut[row_ty1_base + seg.tx1];
                uint16_t *__restrict__ hist_row =
                    hist[hist_row_base + seg.col_tile];

                for (int x = seg.x_start; x < seg.x_end; ++x) {
                    const int fx     = fx_tab[x];
                    const int fx_inv = 256 - fx;

                    const int idx = (row_base + x) * 2;
                    const uint8_t hi = w8[idx];
                    const uint8_t lo = w8[idx + 1];
                    const uint8_t r5 = static_cast<uint8_t>(hi >> 3);
                    const uint8_t g6 = static_cast<uint8_t>(((hi & 0x07) << 3) | (lo >> 5));
                    const uint8_t b5 = static_cast<uint8_t>(lo & 0x1F);
                    // RGB565 -> RGB888 expansion *with* the adaptive
                    // gamma curve folded in (gamma=1 collapses to the
                    // canonical (v<<3)|(v>>2) / (v<<2)|(v>>4) bit-spread).
                    const int r8e = gamma_r5[r5];
                    const int g8e = gamma_g6[g6];
                    const int b8e = gamma_b5[b5];

                    // Luma proxy: G6 expanded to 8 bits. Used both as
                    // the histogram bin index (this-frame stats) and
                    // the LUT lookup index (apply prev-frame LUT).
                    const int y_in = g8e;
                    ++hist_row[y_in];

                    // Bilinear-interp the prev-frame LUT for Y across
                    // the four surrounding tiles.
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
                    const int nr = std::clamp(r8e + dy, 0, 255);
                    const int ng = std::clamp(g8e + dy, 0, 255);
                    const int nb = std::clamp(b8e + dy, 0, 255);

                    const uint8_t nr5 = static_cast<uint8_t>(nr >> 3);
                    const uint8_t ng6 = static_cast<uint8_t>(ng >> 2);
                    const uint8_t nb5 = static_cast<uint8_t>(nb >> 3);
                    w8[idx]     = static_cast<uint8_t>((nr5 << 3) | (ng6 >> 3));
                    w8[idx + 1] = static_cast<uint8_t>((ng6 << 5) | nb5);
                }
            }
        }

        // Per-tile stats from the histograms we just built:
        //   * frame mean luma  -- drives the AE preset cycle.
        //   * inter-tile luma spread (max tile mean - min tile mean)
        //     -- proxy for "scene is half-lit / strongly directional".
        //     A flat-lit face produces spread ~5-15 across the 16
        //     tiles; a half-lit face with the shadow line through
        //     the cheeks easily hits 60-120. Used below to bump
        //     the CLAHE clip-limit for asymmetric-lighting scenes
        //     so the dark side still gets equalised even when the
        //     frame's mean luma lands in BRIGHT (where the static
        //     2 % clip is too gentle to do useful work on a deep
        //     shadow). Free as a side-effect of the per-tile sum.
        int tile_mean_min = 255;
        int tile_mean_max = 0;
        {
            uint32_t total_sum = 0;
            for (int t = 0; t < CLAHE_N * CLAHE_N; ++t) {
                uint32_t tile_sum = 0;
                for (int v = 0; v < 256; ++v) {
                    tile_sum += static_cast<uint32_t>(hist[t][v]) *
                                static_cast<uint32_t>(v);
                }
                total_sum += tile_sum;
                const int tile_mean = static_cast<int>(tile_sum / CLAHE_TILE_PX);
                if (tile_mean < tile_mean_min) tile_mean_min = tile_mean;
                if (tile_mean > tile_mean_max) tile_mean_max = tile_mean;
            }
            g_last_mean_luma.store(
                static_cast<int>(total_sum /
                                 (static_cast<uint32_t>(FRAME_DIM) * FRAME_DIM)),
                std::memory_order_relaxed);
        }
        const int luma_spread = tile_mean_max - tile_mean_min;
        g_last_luma_spread.store(luma_spread, std::memory_order_relaxed);

        // Build NEXT frame's LUT from this frame's histograms. The
        // resulting `lut[][]` will be picked up by the fused pass on
        // the next apply_clahe() call. Doing this last (rather than
        // mid-function as the old pass 2) is what enables the single-
        // pass design: by the time we look up `lut[]` in the fused
        // loop above, it's already been built from the previous
        // frame's stats.
        //
        // Spread-adaptive clip-limit: start from the preset's base
        // value, then for half-lit / backlit scenes ramp up toward
        // the DIM-preset clip (7 %) so CLAHE can actually pull the
        // dark side of the face out of crush. The ramp is linear
        // between LUMA_SPREAD_FLAT (= no adjustment) and
        // LUMA_SPREAD_HARSH (= full DIM-preset clip). Below the
        // flat threshold the preset's own value wins; above the
        // harsh threshold we cap at the DIM clip so a single
        // blown-out highlight can't push the clip past what we'd
        // use for an actually-dim scene.
        constexpr int LUMA_SPREAD_FLAT  = 30;
        constexpr int LUMA_SPREAD_HARSH = 90;
        constexpr int CLIP_HARSH        = TILE_PX_HELPER * 9 / 100;   // DIM-preset value
        int clip_lim = g_tuning.clahe_clip_lim;
        if (luma_spread > LUMA_SPREAD_FLAT && clip_lim < CLIP_HARSH) {
            const int ramp = std::min(luma_spread, LUMA_SPREAD_HARSH) -
                             LUMA_SPREAD_FLAT;
            const int span = LUMA_SPREAD_HARSH - LUMA_SPREAD_FLAT;
            const int boosted = clip_lim +
                                ((CLIP_HARSH - clip_lim) * ramp) / span;
            clip_lim = boosted;
        }
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
                hist[t][v] += bonus + (v < static_cast<int>(leftover) ? 1 : 0);
            }
            // CDF, scaled into the LUT's 0..255 output range.
            uint32_t cdf = 0;
            for (int v = 0; v < 256; ++v) {
                cdf += hist[t][v];
                lut[t][v] = static_cast<uint8_t>((cdf * 255) / CLAHE_TILE_PX);
            }
        }
    }

    // Sub-sampled mean-luma scan over the rotated detector input.
    // Prepare the buffer the detector will read this frame. Returns
    // the pointer to feed into HumanFaceDetect::run() -- always
    // `scratch` here, since we always rotate (the rotate is required
    // by the orient-cycle) and always run CLAHE (gentle clip for
    // BRIGHT, moderate for MID, aggressive for DIM -- the clip-limit
    // is the per-preset tuning knob, not the on/off switch).
    __attribute__((hot))
    const uint16_t *prep_detect_input(Orient o,
                                      const uint16_t *src,
                                      uint16_t *scratch,
                                      int n) noexcept
    {
        rotate_to_scratch(o, src, scratch, n);
#if CONFIG_ESPDET_PICO_224_224_FACE || CONFIG_ESPDET_PICO_416_416_FACE
        // ESPDet's single-stage anchor-free detector was trained on
        // natural-image input statistics; our adaptive gamma + per-
        // tile CLAHE preprocessing pushes pixel distributions far
        // enough out of distribution that the model returns zero
        // detections on a face that's plainly in frame. Skip the
        // heavy preprocessing for ESPDet builds -- the ImagePre-
        // processor inside the component handles its own [0..1]
        // normalisation, and ESPDet's published mAP50-95 of 0.495
        // already factors in robust handling of lighting variation.
        //
        // We still need to keep g_last_mean_luma fresh so the AE
        // preset cycle works, so do a coarse mean-luma scan in
        // place of apply_clahe's "free side effect" mean.
        {
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
            // No per-tile spread proxy here; CLAHE is disabled so
            // the spread-adaptive clip-limit path is moot. Keep the
            // atomic at zero so anyone reading it sees "flat".
            g_last_luma_spread.store(0, std::memory_order_relaxed);
        }
#else
        apply_clahe(scratch);
#endif
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
                // apply_clahe() uses for g_last_mean_luma.
                const uint8_t g6  = static_cast<uint8_t>(((hi & 0x07) << 3) |
                                                         (lo >> 5));
                sum += static_cast<uint32_t>((g6 << 2) | (g6 >> 4));
                ++n;
            }
        }
        return n ? static_cast<int>(sum / n) : -1;
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
        // We previously bumped the MSR (proposal) stage's score floor to
        // 0.65 to speed up MNP refinement. That helped throughput at
        // mid range but starved real close-range detections whose MSR
        // confidence is naturally lower (face out of training
        // distribution). The default 0.5 was the right tradeoff; we now
        // get the same false-positive resistance from MIN_DETECT_SCORE
        // + keypoints_look_upright at the app level.
        auto      feat     = std::make_unique<HumanFaceFeat>();
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
            int      matched_id  = 0;              // 1-based DB id; 0 = nothing matched
        } recog_cache;

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
                std::strncpy(name_banner_cache.name, local_name,
                             FACE_NAME_MAX - 1);
                name_banner_cache.name[FACE_NAME_MAX - 1] = '\0';
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
        // at a rotated orient can pass score + keypoints_look_upright
        // checks and lock us onto an upside-down / sideways orient,
        // even when the user is actually holding the device upright.
        // Common case: the detector fires on a noisy patch at ROT_180
        // with a score in the 0.55-0.7 band, banner ends up upside-
        // down, user has to wait for the cycle to recover. Requiring
        // two consecutive frames of agreement at the SAME non-ROT_0
        // orient before committing the lock filters that out without
        // slowing legitimate rotated tracking by more than one frame.
        // ROT_0 needs no confirmation (it's the default / cold-start
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
            // is refreshed every detection frame inside apply_clahe(),
            // so this check is cheap (one atomic load + a few
            // comparisons). The asymmetric hysteresis bands stop us
            // from oscillating between presets when a scene's luma
            // sits near a boundary.
            if (now_us >= ae_next_check_us)
            {
                // Prefer face-region luma when we've seen a face
                // recently; fall back to whole-frame mean when the
                // room is empty (or the lock dropped long enough that
                // the face-luma reading is stale). The face-priority
                // path is what handles backlit subjects correctly --
                // see g_last_face_luma's declaration block.
                const int  global_luma = g_last_mean_luma.load(
                    std::memory_order_relaxed);
                const int  face_lum    = g_last_face_luma.load(
                    std::memory_order_relaxed);
                const uint32_t face_ms = g_last_face_luma_ms.load(
                    std::memory_order_relaxed);
                const bool face_fresh =
                    face_lum >= 0 &&
                    (now_ms() - face_ms) < FACE_LUMA_STALE_MS;
                const int  luma        = face_fresh ? face_lum
                                                    : global_luma;
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
                    if (keypoints_look_upright(&upscale_remap)) {
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
            }

            const float score_floor = min_detect_score_for(try_orient);
            if (!biggest ||
                biggest->score < score_floor ||
                !keypoints_look_upright(biggest))
            {
                if (biggest)
                {
                    if (biggest->score < score_floor) {
                        // Low-confidence detection: don't bother
                        // running the embedder on it, don't update
                        // the HUD with a flaky bbox, and don't lock
                        // the orient cycle onto whatever it landed on.
                        ESP_LOGD(TAG,
                                 "drop low-score det: score=%.2f < %.2f (orient=%d)",
                                 (double)biggest->score,
                                 (double)score_floor,
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
                    r0_best->score >= MIN_DETECT_SCORE_ROT0 &&
                    keypoints_look_upright(r0_best);

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
            // MID) so CLAHE has room to lift it. Sampling on every
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
            dl::TensorBase *t = feat->run(img, biggest->keypoint);
            if (!t || t->get_size() != feat_len)
            {
                ESP_LOGW(TAG, "feat model returned unexpected tensor");
                esp_camera_fb_return(fb);
                continue;
            }
            const float *raw_feat = t->get_element_ptr<float>();

            // Compare against every stored embedding. Per face we
            // take the MAX cosine similarity across its template
            // gallery, so off-angle / profile templates can rescue a
            // match that the front-on template alone would miss. We
            // hold the DB mutex for the duration so a concurrent
            // rename from the web server can't tear the read; the
            // loop is at most MAX_KNOWN_FACES * FACE_MAX_TEMPLATES *
            // feat_len floats (~64 K FLOPs after SIMD), well under a
            // millisecond of critical section.
            float best_sim = -2.f;
            float best_template_sim = -2.f;  // best sim against the winning face's gallery
            int   best_id = -1;
            int   best_template_idx = -1;
            float second_best_sim = -2.f;    // best sim against any OTHER face
            {
                FaceDbLock lock;
                for (size_t i = 0; i < g_known_faces.size(); ++i)
                {
                    float face_best = -2.f;
                    int   face_best_t = -1;
                    for (size_t t = 0; t < g_known_faces[i].feats.size(); ++t) {
                        const float s = cosine_sim(raw_feat,
                                                   g_known_faces[i].feats[t].data(),
                                                   feat_len);
                        if (s > face_best) {
                            face_best   = s;
                            face_best_t = (int)t;
                        }
                    }
                    if (face_best > best_sim)
                    {
                        // The previous winner becomes the runner-up.
                        if (best_id > 0 && best_sim > second_best_sim) {
                            second_best_sim = best_sim;
                        }
                        best_sim          = face_best;
                        best_template_sim = face_best;
                        best_template_idx = face_best_t;
                        best_id           = (int)i + 1; // 1-based for logging niceness
                    } else if (face_best > second_best_sim) {
                        second_best_sim = face_best;
                    }
                }
            }

            if (best_id > 0 && best_sim >= g_tuning.match_thr)
            {
                // Recognised — no enroll, no banner. We removed the
                // "OH... IT'S YOU" path: recognition is now silent so
                // the only on-screen event is a brand-new face.
                ++s_recognised;
                unknown_streak = 0;
                ESP_LOGI(TAG,
                         "known face: id=%d sim=%.3f (t=%d, 2nd=%.3f) focus=%d orient=%d",
                         best_id, (double)best_sim, best_template_idx,
                         (double)second_best_sim, focus, (int)locked_orient);

                // Auto-augment the gallery for this face when the
                // match is *comfortably* above the threshold (so we
                // know it's the same person) AND no other enrolled
                // face came within FACE_AUGMENT_MARGIN of the winner
                // (so we can't be appending a template from someone
                // who happens to look similar to two people). The
                // new embedding must also be meaningfully different
                // from every existing template in this face's
                // gallery (best_template_sim < FACE_AUGMENT_NOVEL),
                // otherwise it's a near-duplicate that just wastes
                // gallery slots and matcher cycles. The augment gate
                // is intentionally *tighter* than the match gate;
                // weakening recognition for off-angle faces happens
                // exclusively through more templates, never through
                // a lower match floor.
                const float augment_floor = g_tuning.match_thr +
                                            FACE_AUGMENT_MARGIN;
                const bool augment_ok =
                    best_sim          >= augment_floor &&
                    best_template_sim <  FACE_AUGMENT_NOVEL &&
                    second_best_sim   <  best_sim - FACE_AUGMENT_MARGIN;
                if (augment_ok)
                {
                    FaceDbLock lock;
                    const size_t idx = (size_t)(best_id - 1);
                    if (idx < g_known_faces.size() &&
                        g_known_faces[idx].feats.size() < FACE_MAX_TEMPLATES)
                    {
                        g_known_faces[idx].feats.emplace_back(
                            raw_feat, raw_feat + feat_len);
                        ESP_LOGI(TAG,
                                 "gallery+: id=%d now has %u templates (added sim=%.3f, novel=%.3f)",
                                 best_id,
                                 (unsigned)g_known_faces[idx].feats.size(),
                                 (double)best_sim,
                                 (double)best_template_sim);
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

            // Commit the enrollment. Copy the float* into the known list
            // (the source buffer is recycled by the model on the next run).
            // Every successful enrollment fires the "NEW FACE DETECTED"
            // overlay, refreshing both the deadline and the rotation angle
            // so the text always reads upright relative to the current
            // face pose.
            bool hit_cap = false;
            size_t new_count = 0;
            {
                FaceDbLock lock;
                if (g_known_faces.size() >= MAX_KNOWN_FACES) {
                    hit_cap = true;
                } else {
                    KnownFace entry;
                    entry.feats.emplace_back(raw_feat, raw_feat + feat_len);
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

bool face_ai_name_banner_active(void)
{
    return g_name_banner_until_ms.load(std::memory_order_relaxed) > now_ms();
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
