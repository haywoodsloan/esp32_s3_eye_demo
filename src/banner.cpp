#include "banner.h"
#include "board_pins.h"
#include "fonts/FreeSans18pt7b.h"

#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

static const char *TAG = "banner";

#define BANNER_W            BOARD_LCD_H_RES   /* 240 */
#define BANNER_H            BOARD_LCD_V_RES   /* 240 */

/* Text is rendered from a real bitmap font (FreeSans 18pt regular)
   that was pre-converted from a GNU FreeFont TTF using Adafruit's
   `fontconvert` tool and shipped verbatim in src/fonts/. The format
   is 1 bit per pixel, packed MSB-first into a shared byte array, with
   a per-glyph metadata table giving width / height / pen-advance /
   baseline-relative xOffset / yOffset. yAdvance gives the line
   height; cap height is ~25 px.

   Each lit glyph bit is splatted into the alpha buffer as a small
   soft circular dot. The dot's only job here is to add 1 px of soft
   AA at the boundary between lit and unlit pixels -- without it,
   the text would have hard pixel-edge jaggies once the buffer is
   rotated for off-axis face poses. The radius is much smaller than
   the previous hand-rolled 8x16 splat font used (which relied on
   overlapping dots for the stroke body); the FreeSans glyph shapes
   already carry the typographic weight. */

static const GFXfont &kFont = FreeSans18pt7b;

#define LETTER_GAP_PX       1                   /* extra inter-letter tracking */
#define LINE_OFFSET_PX      88                  /* centre -> line-centre, in local coords */
#define BASELINE_OFFSET_PX  12                  /* line-centre -> baseline (~half the cap height) */
#define DOT_RADIUS_PX       1.2f                /* edge-AA splat; not stroke weight */
#define OUTLINE_RADIUS_PX   2                   /* black outline halo width */

/* The LCD and the camera both use big-endian RGB565 byte order in memory
   (high byte first). We're authoring buffers on a little-endian host,
   so writing a uint16_t lays its bytes down low-byte-first. The BE() /
   FROM_BE() macros swap between the in-memory big-endian form and a
   native little-endian uint16_t so per-channel math can be done in
   normal arithmetic. */
#define BE(v)               ((uint16_t)(((v) >> 8) | ((v) << 8)))
#define FROM_BE(v)          BE(v)

/* 8-bit alpha mask the soft-dot rasterizer paints into. 0 = leave
   underlying camera pixel untouched, 255 = fully replace with the
   banner foreground colour. Intermediate values produce anti-aliased
   blending in banner_compose_overlay(). */
static uint8_t *s_alpha = NULL;

/* Dilated copy of s_alpha used to draw a black outline around the
   coloured text. Computed once per banner_render() as a max-filter
   over s_alpha. */
static uint8_t *s_outline = NULL;

/* Second alpha + outline buffer pair, used by banner_render_name()
   for the navy-blue "recognized" banner. Independent of s_alpha /
   s_outline so the two banners can be active simultaneously (e.g.
   the moment after a new enrollment, when the green NEW FACE banner
   is still up and the matcher has just hit the freshly-enrolled
   entry). */
static uint8_t *s_name_alpha   = NULL;
static uint8_t *s_name_outline = NULL;

/* Per-banner dirty-y bounds. Tracked during render_lines_into() so the
   compose pass can skip the unconditionally-empty rows above and below
   the rasterised glyphs -- a real win because the render task does the
   compose every LCD frame on the live preview, and the banners only
   ever touch ~25-30 % of the 240-row buffer at any given rotation.

   min_y > max_y signals an empty mask (nothing has been rendered yet
   or all glyphs were filtered out); compose loops skip out entirely in
   that case. */
struct BannerBounds {
    int min_y;
    int max_y;
};
static BannerBounds s_green_bounds = { BANNER_H, -1 };
static BannerBounds s_name_bounds  = { BANNER_H, -1 };

/* Banner foreground: dark green. Readable against a wide range of skin
   / room tones; the black outline added by s_outline keeps it legible
   even when the scene behind it is also greenish. */
static constexpr uint8_t BANNER_FG_R5 = 0;
static constexpr uint8_t BANNER_FG_G6 = 25;   /* ~40% of full G6 */
static constexpr uint8_t BANNER_FG_B5 = 0;
/* Name banner foreground: navy blue. Matched in luminance to the
   green banner (~40% of full B5) so the two read as the same
   "weight" of overlay against the camera scene. Pure blue channel
   only -- no red / green mix, so it stays visibly navy rather than
   sliding toward royal blue when the underlying scene is bright. */
static constexpr uint8_t NAME_FG_R5 = 0;
static constexpr uint8_t NAME_FG_G6 = 0;
static constexpr uint8_t NAME_FG_B5 = 16;     /* navy: 128/255 of full B5 */
/* Banner outline: black -- all channels zero. Spelled out so the
   compose loop reads the same as the foreground blend. */
static constexpr uint8_t BANNER_BG_R5 = 0;
static constexpr uint8_t BANNER_BG_G6 = 0;
static constexpr uint8_t BANNER_BG_B5 = 0;

static const char *const s_lines[]    = { "NEW FACE", "DETECTED" };
static constexpr int      s_line_count = sizeof(s_lines) / sizeof(s_lines[0]);

/* Look up a glyph by codepoint. Returns nullptr for codepoints outside
   the shipped font's range; callers either skip (drawing) or substitute
   a space (input filtering). */
static inline const GFXglyph *find_glyph(char c)
{
    const uint8_t u = (uint8_t)c;
    if (u < kFont.first || u > kFont.last) {
        return NULL;
    }
    return &kFont.glyph[u - kFont.first];
}

/* Width of a character cell in pen-pixels: the glyph's xAdvance plus
   inter-letter tracking. The space character has its own xAdvance in
   the font metrics (no special-casing needed unlike the old 8x16
   monospace font). */
static inline int cell_width_for(char c)
{
    const GFXglyph *g = find_glyph(c);
    return g ? (int)g->xAdvance + LETTER_GAP_PX : 0;
}

/* Splat one soft circular dot into the supplied alpha buffer at
   destination centre (cx, cy). Intensity falls off as 1 - d^2 / r^2
   from the centre out to DOT_RADIUS_PX (quadratic / parabolic falloff:
   avoids a sqrt per pixel, looks the same to the eye as linear at this
   scale). Alpha is accumulated additively and clamped to 255, so
   overlapping dots in the stroke core saturate to opaque while the
   dot edge falls smoothly to the background. Takes the destination
   buffer + bounds pair as parameters so the same primitive can be
   reused for the name banner without aliasing the green-banner state.
   The bounds are updated to bracket every row the splat actually
   touches; render_lines_into() expands them by the outline radius
   afterwards so the dilation pass's output range is covered too. */
static inline void splat_dot(uint8_t *alpha, BannerBounds *bounds,
                             float cx, float cy)
{
    int x0 = (int)floorf(cx - DOT_RADIUS_PX);
    int x1 = (int)ceilf (cx + DOT_RADIUS_PX);
    int y0 = (int)floorf(cy - DOT_RADIUS_PX);
    int y1 = (int)ceilf (cy + DOT_RADIUS_PX);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > BANNER_W - 1) x1 = BANNER_W - 1;
    if (y1 > BANNER_H - 1) y1 = BANNER_H - 1;
    if (x0 > x1 || y0 > y1) return;

    if (y0 < bounds->min_y) bounds->min_y = y0;
    if (y1 > bounds->max_y) bounds->max_y = y1;

    const float r2     = DOT_RADIUS_PX * DOT_RADIUS_PX;
    const float inv_r2 = 1.0f / r2;

    for (int y = y0; y <= y1; ++y) {
        const float dy  = (float)y + 0.5f - cy;
        const float dy2 = dy * dy;
        uint8_t *row = alpha + (size_t)y * BANNER_W;
        for (int x = x0; x <= x1; ++x) {
            const float dx = (float)x + 0.5f - cx;
            const float d2 = dx * dx + dy2;
            if (d2 >= r2) continue;
            const float t   = 1.0f - d2 * inv_r2;     /* 0..1 */
            const int   add = (int)(t * 255.0f + 0.5f);
            const int   sum = (int)row[x] + add;
            row[x] = sum > 255 ? 255 : (uint8_t)sum;
        }
    }
}

/* Shared worker: rasterise an arbitrary list of text lines into the
   supplied alpha buffer, then dilate into the supplied outline buffer.
   `line_y_signs` is parallel to `lines` and selects which edge of the
   240x240 buffer each line hugs (-1 = top, +1 = bottom). All lines are
   centred horizontally and rendered into the same buffer in one pass.
   `angle_rad` rotates the entire text block around the buffer centre
   (positive = clockwise in screen coordinates). `bounds` is updated
   to the dirty-y range of the resulting alpha+outline pair so the
   per-frame compose can skip rows the banner doesn't touch. */
static void render_lines_into(uint8_t *alpha, uint8_t *outline,
                              BannerBounds *bounds,
                              const char *const *lines,
                              const int *line_y_signs,
                              int n_lines, float angle_rad)
{
    memset(alpha, 0, (size_t)BANNER_W * BANNER_H);
    /* Empty-mask sentinels; splat_dot widens them as it draws, then
       the dilation pass below expands by OUTLINE_RADIUS_PX so the
       outline halo's vertical extent is covered too. */
    bounds->min_y = BANNER_H;
    bounds->max_y = -1;

    const float ca = cosf(angle_rad);
    const float sa = sinf(angle_rad);
    const int   cx = BANNER_W / 2;
    const int   cy = BANNER_H / 2;

    for (int li = 0; li < n_lines; ++li) {
        const char *line = lines[li];
        const int   len  = (int)strlen(line);

        /* Compute the line's total pixel width by summing per-character
           xAdvance values from the font's glyph metrics (so variable-
           width glyphs centre properly). Done as a separate pass
           because we need the total before deciding the centred left
           edge. The final character doesn't carry a trailing gap, so
           subtract one tracking gap from the running total. */
        int line_w_px = 0;
        for (int ci = 0; ci < len; ++ci) {
            line_w_px += cell_width_for(line[ci]);
        }
        if (line_w_px > 0) {
            line_w_px -= LETTER_GAP_PX;
        }

        /* Pen position in local coords. sign==-1 -> line above centre
           (top edge of face view); sign==+1 -> line below centre.
           baseline_local is where each glyph's baseline sits; glyphs
           extend yOffset above it (yOffset is negative for caps).
           BASELINE_OFFSET_PX is ~half the cap height, so the visual
           centre of the line lands at sign * LINE_OFFSET_PX. */
        const float pen_x0         = -(float)line_w_px * 0.5f;
        const int   sign           = line_y_signs[li];
        const float baseline_local = (float)sign * (float)LINE_OFFSET_PX
                                     + (float)BASELINE_OFFSET_PX;

        float pen_x = pen_x0;
        for (int ci = 0; ci < len; ++ci) {
            const GFXglyph *g = find_glyph(line[ci]);
            if (!g) {
                /* Unknown codepoint: advance by a space's width so the
                   rest of the line stays aligned. */
                const GFXglyph *space = find_glyph(' ');
                pen_x += space ? (float)space->xAdvance + (float)LETTER_GAP_PX
                               : 0.0f;
                continue;
            }

            const uint8_t *gbits  = kFont.bitmap + g->bitmapOffset;
            const int      gw     = (int)g->width;
            const int      gh     = (int)g->height;
            const float    glyph_lx = pen_x + (float)g->xOffset;
            const float    glyph_ly = baseline_local + (float)g->yOffset;

            /* Walk the glyph's packed 1bpp bitmap MSB-first. Each lit
               bit corresponds to one source-font pixel; we splat a
               soft edge-AA dot at the rotated destination location. */
            int bit_cursor = 0;
            for (int gy = 0; gy < gh; ++gy) {
                for (int gx = 0; gx < gw; ++gx) {
                    const int byte_idx = bit_cursor >> 3;
                    const int bit_idx  = 7 - (bit_cursor & 7);
                    ++bit_cursor;
                    if (!((gbits[byte_idx] >> bit_idx) & 1)) continue;

                    /* Source-pixel centre in local coords. */
                    const float lx = glyph_lx + (float)gx + 0.5f;
                    const float ly = glyph_ly + (float)gy + 0.5f;
                    /* Rotate into destination-pixel space and splat. */
                    const float dst_cx = ca * lx - sa * ly + (float)cx;
                    const float dst_cy = sa * lx + ca * ly + (float)cy;
                    splat_dot(alpha, bounds, dst_cx, dst_cy);
                }
            }

            pen_x += (float)g->xAdvance + (float)LETTER_GAP_PX;
        }
    }

    /* Empty mask -- no glyphs landed in the buffer (rendering an empty
       string, or a name that was all-filtered). Zero the outline so a
       stale previous render doesn't bleed through, leave bounds in the
       empty-sentinel state, and skip the dilation pass. */
    if (bounds->max_y < bounds->min_y) {
        memset(outline, 0, (size_t)BANNER_W * BANNER_H);
        return;
    }

    /* Dilate the alpha mask into the outline buffer: each pixel is the
       maximum of `alpha` over a (2R+1) x (2R+1) window centred on it.
       Keeps the soft-edge gradient at the dilation boundary so the
       outline AA-matches the foreground stroke. We only need to write
       outline pixels in [min_y - R, max_y + R] because anywhere
       outside that band the alpha window is uniformly zero -- and we
       memset the whole outline first so stale pixels from a previous
       render with a wider dirty band can't survive. */
    const int R          = OUTLINE_RADIUS_PX;
    const int dilation_y0 = (bounds->min_y > R) ? bounds->min_y - R : 0;
    const int dilation_y1 = (bounds->max_y + R < BANNER_H)
                            ? bounds->max_y + R : BANNER_H - 1;

    memset(outline, 0, (size_t)BANNER_W * BANNER_H);

    for (int y = dilation_y0; y <= dilation_y1; ++y) {
        const int y0 = (y > R) ? y - R : 0;
        const int y1 = (y + R < BANNER_H) ? y + R : BANNER_H - 1;
        uint8_t *orow = outline + (size_t)y * BANNER_W;
        for (int x = 0; x < BANNER_W; ++x) {
            const int x0 = (x > R) ? x - R : 0;
            const int x1 = (x + R < BANNER_W) ? x + R : BANNER_W - 1;
            uint8_t m = 0;
            for (int yy = y0; yy <= y1; ++yy) {
                const uint8_t *arow = alpha + (size_t)yy * BANNER_W;
                for (int xx = x0; xx <= x1; ++xx) {
                    if (arow[xx] > m) {
                        m = arow[xx];
                        if (m == 255) goto done;
                    }
                }
            }
        done:
            orow[x] = m;
        }
    }

    /* Expand bounds to cover the outline halo so the compose pass sees
       the full dirty range, not just the inner glyph stroke band. */
    bounds->min_y = dilation_y0;
    bounds->max_y = dilation_y1;
}

/* Shared worker: composite a banner alpha + outline pair into an
   RGB565BE frame buffer with the given foreground colour. Outline is
   black (all-zero channels); foreground is the 5/6/5 triple passed in.
   `bounds` restricts the y loop to the rows where the banner actually
   has non-zero content (the rasterise pass tracks this so we don't
   walk the ~70 % of the buffer that's empty in any given rotation).
   See banner_compose_overlay() for the per-channel math.

   Hot + O3 because this runs every LCD frame in the render task, and
   the inner loop is precisely the kind of straight-line per-pixel
   blend GCC can pipeline tightly given the chance. */
__attribute__((hot, optimize("O3")))
static void compose_with(uint16_t *__restrict__ frame, int width, int height,
                         const uint8_t *__restrict__ alpha,
                         const uint8_t *__restrict__ outline,
                         BannerBounds bounds,
                         int fg_r5, int fg_g6, int fg_b5)
{
    /* Empty mask -> nothing to do. The empty sentinel is min_y=BANNER_H,
       max_y=-1, both leaving the loop with zero iterations even without
       this short-circuit, but the explicit early-out avoids the per-row
       bounds-clamp math. */
    if (bounds.max_y < bounds.min_y) {
        return;
    }

    const int w  = (width  < BANNER_W) ? width  : BANNER_W;
    const int h  = (height < BANNER_H) ? height : BANNER_H;
    const int y0 = (bounds.min_y > 0) ? bounds.min_y : 0;
    const int y1 = (bounds.max_y + 1 < h) ? bounds.max_y + 1 : h;

    for (int y = y0; y < y1; ++y) {
        const uint8_t *arow = alpha   + (size_t)y * BANNER_W;
        const uint8_t *orow = outline + (size_t)y * BANNER_W;
        uint16_t      *frow = frame   + (size_t)y * width;
        for (int x = 0; x < w; ++x) {
            const int oa = orow[x];
            if (!oa) continue;            /* common case: nothing to draw */
            const int ta = arow[x];

            const uint16_t src = FROM_BE(frow[x]);
            int r = (src >> 11) & 0x1F;
            int g = (src >> 5)  & 0x3F;
            int b =  src        & 0x1F;

            /* Pass 1: blend toward black (outline) by (255 - oa). */
            const int inv_o = 255 - oa;
            r = (r * inv_o + 127) / 255;
            g = (g * inv_o + 127) / 255;
            b = (b * inv_o + 127) / 255;

            /* Pass 2: blend toward the foreground colour by text alpha. */
            if (ta) {
                const int inv_t = 255 - ta;
                r = (r * inv_t + fg_r5 * ta + 127) / 255;
                g = (g * inv_t + fg_g6 * ta + 127) / 255;
                b = (b * inv_t + fg_b5 * ta + 127) / 255;
            }

            const uint16_t out = (uint16_t)((r << 11) | (g << 5) | b);
            frow[x] = BE(out);
        }
    }
}

esp_err_t banner_init(void)
{
    if (s_alpha) {
        return ESP_OK;
    }

    const size_t alpha_bytes = (size_t)BANNER_W * BANNER_H;
    s_alpha = (uint8_t *)heap_caps_aligned_alloc(
        16, alpha_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    ESP_RETURN_ON_FALSE(s_alpha, ESP_ERR_NO_MEM, TAG, "alloc banner alpha buf");

    s_outline = (uint8_t *)heap_caps_aligned_alloc(
        16, alpha_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    ESP_RETURN_ON_FALSE(s_outline, ESP_ERR_NO_MEM, TAG, "alloc banner outline buf");

    s_name_alpha = (uint8_t *)heap_caps_aligned_alloc(
        16, alpha_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    ESP_RETURN_ON_FALSE(s_name_alpha, ESP_ERR_NO_MEM, TAG, "alloc name alpha buf");

    s_name_outline = (uint8_t *)heap_caps_aligned_alloc(
        16, alpha_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    ESP_RETURN_ON_FALSE(s_name_outline, ESP_ERR_NO_MEM, TAG, "alloc name outline buf");

    /* Zero both name buffers so an early composite (before any face has
       been recognised) produces no output. The green banner gets an
       initial upright render below; the name banner stays blank until
       face_ai publishes the first recognised name. */
    memset(s_name_alpha,   0, alpha_bytes);
    memset(s_name_outline, 0, alpha_bytes);

    /* Initial green-banner render is upright (angle 0) so the mask has
       reasonable contents if anyone composites it before the AI task
       has had a chance to call banner_render with a real angle. */
    banner_render(0.0f);

    ESP_LOGI(TAG, "banner ready (%dx%d, %u bytes alpha+outline x2 in PSRAM)",
             BANNER_W, BANNER_H, (unsigned)(alpha_bytes * 4));
    return ESP_OK;
}

void banner_render(float angle_rad)
{
    if (!s_alpha || !s_outline) {
        return;
    }
    /* Green banner: two lines, line 0 along the top edge, line 1 along
       the bottom. Together they form the "NEW FACE / DETECTED"
       enrollment overlay. */
    static const int signs[s_line_count] = { -1, +1 };
    render_lines_into(s_alpha, s_outline, &s_green_bounds, s_lines, signs,
                      s_line_count, angle_rad);
}

/* Map an arbitrary input character to a glyph we know how to render.
   FreeSans18pt7b covers ASCII 0x20..0x7E (printable 7-bit). We
   fold lowercase to uppercase on the way through so the name banner
   stays visually consistent with the all-caps "NEW FACE DETECTED"
   enrollment banner -- mixed-case bodies on a screen this small read
   noticeably less crisp than caps. Everything outside the printable
   range becomes a space so an unknown byte opens up tracking rather
   than shifting later letters around. */
static inline char name_char_filter(char c)
{
    const uint8_t u = (uint8_t)c;
    if (u >= 'a' && u <= 'z') return (char)(u - 'a' + 'A');
    return (u >= 0x20 && u <= 0x7E) ? c : ' ';
}

/* Hard cap on the working buffer regardless of what fits visually.
   The visual truncation below uses the actual cumulative pen-advance
   to stop adding glyphs once the line's about to overflow; this is
   just here to bound stack usage. */
static constexpr int NAME_BANNER_BUF_MAX = 64;

/* Pixel budget for a single banner line: full panel width minus a
   small margin so the outline doesn't kiss the edge. */
static constexpr int NAME_BANNER_WIDTH_BUDGET_PX = BANNER_W - 20;

void banner_render_name(const char *name, float angle_rad)
{
    if (!s_name_alpha || !s_name_outline || !name) {
        return;
    }

    /* Filter into a local buffer, truncating as soon as the next glyph
       would push us past the line's width budget. Empty / all-space
       input renders to an empty line, which the compose loop skips. */
    char filtered[NAME_BANNER_BUF_MAX + 1];
    int  out    = 0;
    int  width  = 0;
    for (int i = 0; name[i] != '\0' && out < NAME_BANNER_BUF_MAX; ++i) {
        const char c = name_char_filter(name[i]);
        const int  w = cell_width_for(c);
        if (width + w > NAME_BANNER_WIDTH_BUDGET_PX) {
            break;
        }
        filtered[out++] = c;
        width += w;
    }
    filtered[out] = '\0';

    const char *const lines[1] = { filtered };
    const int         signs[1] = { +1 };          /* bottom edge */
    render_lines_into(s_name_alpha, s_name_outline, &s_name_bounds,
                      lines, signs, 1, angle_rad);
}

void banner_compose_overlay(uint16_t *frame, int width, int height)
{
    if (!s_alpha || !s_outline || !frame) {
        return;
    }
    compose_with(frame, width, height, s_alpha, s_outline, s_green_bounds,
                 BANNER_FG_R5, BANNER_FG_G6, BANNER_FG_B5);
}

void banner_compose_name_overlay(uint16_t *frame, int width, int height)
{
    if (!s_name_alpha || !s_name_outline || !frame) {
        return;
    }
    compose_with(frame, width, height, s_name_alpha, s_name_outline,
                 s_name_bounds,
                 NAME_FG_R5, NAME_FG_G6, NAME_FG_B5);
}
