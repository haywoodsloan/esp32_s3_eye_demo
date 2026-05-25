#include "banner.h"
#include "board_pins.h"

#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const char *TAG = "banner";

#define BANNER_W            BOARD_LCD_H_RES   /* 240 */
#define BANNER_H            BOARD_LCD_V_RES   /* 240 */

/* Source bitmap font is 8x16, strokes are 1-2 source-pixels wide. The
   renderer treats each lit source pixel as a soft circular DOT splatted
   into an 8-bit alpha buffer (s_alpha): adjacent lit pixels' dots
   overlap and their alpha values saturate to 255 in the stroke core,
   while the dot's linear-falloff edge produces 1-2 pixels of smooth
   gradient. The alpha buffer is later composited onto the live camera
   frame by banner_compose_overlay(), so only the text pixels overwrite
   the underlying image and the dot's edge gradient gives anti-aliased
   green-on-camera text.

   With SCALE=3 the longest line ("DETECTED", 8 glyphs) is
   8 * GLYPH_W * SCALE = 192 px from glyph bitmaps alone; adding the
   LETTER_GAP_PX between adjacent letters brings the total line width to
   192 + 7 * 3 = 213 px on the 240 px buffer (~89%, leaving ~13 px /
   ~5.5% padding on each side). The text is positioned along the bottom
   of the face's view via Y_OFFSET below; on 90 deg rotations the same
   text block fits along the corresponding LCD edge. */
#define GLYPH_W             8
#define GLYPH_H             16
/* SCALE is the source-pixel -> destination-pixel ratio used both when
   sampling each lit glyph pixel (splat centre stride below) and when
   sizing CELL_W / CELL_H. A non-integer value is fine because the
   splat is done with float coordinates; the only place we need an
   integer derivative is CELL_W / CELL_H, which we truncate. */
#define SCALE               2.55f
#define CELL_W              ((int)(GLYPH_W * SCALE))   /* 20 */
#define CELL_H              ((int)(GLYPH_H * SCALE))   /* 40 */
#define LETTER_GAP_PX       3                   /* between glyph cells */
#define LINE_GAP_PX         4                   /* between line baselines */
/* Local-coord distance from the buffer centre (cy) to each line's
   centre. With the two-line message we place the first line above the
   centre and the second line below it, so they hug the top and bottom
   edges of the face's view (and, after rotation, the corresponding
   pair of LCD edges). Sized so the text + outline still fits inside
   the 240 px buffer at every multiple-of-90-deg rotation. */
#define LINE_OFFSET_PX      92
/* A full CELL_W between words looked far too airy (the glyphs already
   carry ~1 col of empty source padding on each side, plus LETTER_GAP_PX
   between cells). Render the space character with a half-width cell so
   word breaks read as a clear gap without dominating the line. */
#define SPACE_CELL_W        (CELL_W / 2)        /* 12 */

/* Dot radius (in destination pixels) used to splat each lit source
   pixel. Two adjacent dots (SCALE px apart) sum to a ~5-6 px stroke
   whose core is fully foreground-coloured and whose 1-2 px edge fades
   through intermediate alpha values — i.e. anti-aliased. Scaled in
   step with SCALE so the stroke weight stays visually proportional to
   the glyph size. */
#define DOT_RADIUS_PX       2.1f

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
   white text. Computed once per banner_render() as a max-filter over
   s_alpha; intermediate values give the outline soft edges that match
   the text's anti-aliasing. */
static uint8_t *s_outline = NULL;

/* Second alpha + outline buffer pair, used by banner_render_name()
   for the navy-blue "recognized" banner. Independent of s_alpha /
   s_outline so the two banners can be active simultaneously (e.g.
   the moment after a new enrollment, when the green NEW FACE banner
   is still up and the matcher has just hit the freshly-enrolled
   entry). */
static uint8_t *s_name_alpha   = NULL;
static uint8_t *s_name_outline = NULL;

/* Outline halo half-width in destination pixels. 2 -> a 5x5 max
   filter -> a ~2 px black ring around every stroke. The dilation
   cost (BANNER_W * BANNER_H * (2R+1)^2 comparisons) is paid only on
   banner_render(), which fires at most once per enrolment. */
#define OUTLINE_RADIUS_PX   2

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

/* 8x16 monospace bitmap font. MSB of each row byte is the leftmost pixel.
   Only the 11 glyphs that "I WILL REMEMBER YOU" needs are populated;
   anything else falls back to a blank space. */
typedef struct {
    char    ch;
    uint8_t rows[GLYPH_H];
} glyph_t;

static const glyph_t s_font[] = {
    {' ',  {0}},
    {'I',  {0, 0, 0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0, 0, 0}},
    {'W',  {0, 0, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x5A, 0x5A, 0x66, 0x42, 0, 0, 0}},
    {'L',  {0, 0, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0, 0, 0}},
    {'R',  {0, 0, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x66, 0x66, 0x66, 0, 0, 0}},
    {'E',  {0, 0, 0x7E, 0x60, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0, 0, 0}},
    {'M',  {0, 0, 0x42, 0x66, 0x7E, 0x7E, 0x5A, 0x5A, 0x42, 0x42, 0x42, 0x42, 0x42, 0, 0, 0}},
    {'B',  {0, 0, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x7C, 0, 0, 0}},
    {'Y',  {0, 0, 0x42, 0x42, 0x42, 0x42, 0x24, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0, 0, 0}},
    {'O',  {0, 0, 0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0, 0, 0}},
    {'U',  {0, 0, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x66, 0x3C, 0, 0, 0}},
    /* Added for the "OH... IT'S YOU" recognition banner (kept around
       even though the recognition banner has been removed; cheap, and
       avoids regressing the font if other text is added later). */
    {'H',  {0, 0, 0x42, 0x42, 0x42, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x42, 0x42, 0, 0, 0}},
    {'T',  {0, 0, 0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0, 0, 0}},
    {'S',  {0, 0, 0x7E, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x06, 0x06, 0x06, 0x06, 0x7E, 0, 0, 0}},
    {'.',  {0, 0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0x18, 0x18, 0, 0, 0}},
    {'\'', {0, 0, 0x18, 0x18, 0x18, 0,    0,    0,    0,    0,    0,    0,    0,    0, 0, 0}},
    /* Added for the "NEW FACE DETECTED" overlay. */
    {'N',  {0, 0, 0x42, 0x62, 0x62, 0x52, 0x52, 0x4A, 0x4A, 0x46, 0x46, 0x42, 0x42, 0, 0, 0}},
    {'F',  {0, 0, 0x7E, 0x60, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0, 0, 0}},
    {'A',  {0, 0, 0x18, 0x3C, 0x66, 0x66, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x42, 0, 0, 0}},
    {'C',  {0, 0, 0x3C, 0x66, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x66, 0x3C, 0, 0, 0}},
    {'D',  {0, 0, 0x78, 0x6C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x6C, 0x78, 0, 0, 0}},
    /* Filling out the rest of A-Z so user-entered names render properly
       in the navy "recognized" banner. Glyphs are 8x16, MSB = leftmost
       pixel, same convention as the originals above. */
    {'G',  {0, 0, 0x3C, 0x66, 0x60, 0x60, 0x60, 0x6E, 0x66, 0x66, 0x66, 0x66, 0x3C, 0, 0, 0}},
    {'J',  {0, 0, 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x6C, 0x38, 0, 0, 0}},
    {'K',  {0, 0, 0x66, 0x66, 0x6C, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x6C, 0x66, 0x66, 0, 0, 0}},
    {'P',  {0, 0, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x60, 0x60, 0, 0, 0}},
    {'Q',  {0, 0, 0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0, 0, 0}},
    {'V',  {0, 0, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x66, 0x66, 0x3C, 0x18, 0, 0, 0}},
    {'X',  {0, 0, 0x42, 0x42, 0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x42, 0x42, 0, 0, 0}},
    {'Z',  {0, 0, 0x7E, 0x06, 0x0C, 0x0C, 0x18, 0x18, 0x30, 0x30, 0x60, 0x60, 0x7E, 0, 0, 0}},
    /* Digits 0-9, same 8x16 convention. */
    {'0',  {0, 0, 0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0, 0, 0}},
    {'1',  {0, 0, 0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0, 0, 0}},
    {'2',  {0, 0, 0x3C, 0x66, 0x06, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x60, 0x60, 0x7E, 0, 0, 0}},
    {'3',  {0, 0, 0x3C, 0x66, 0x06, 0x06, 0x1C, 0x06, 0x06, 0x06, 0x06, 0x66, 0x3C, 0, 0, 0}},
    {'4',  {0, 0, 0x0C, 0x1C, 0x2C, 0x4C, 0x4C, 0x7E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0, 0, 0}},
    {'5',  {0, 0, 0x7E, 0x60, 0x60, 0x60, 0x7C, 0x06, 0x06, 0x06, 0x06, 0x66, 0x3C, 0, 0, 0}},
    {'6',  {0, 0, 0x3C, 0x66, 0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0, 0, 0}},
    {'7',  {0, 0, 0x7E, 0x06, 0x06, 0x0C, 0x0C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0, 0, 0}},
    {'8',  {0, 0, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0, 0, 0}},
    {'9',  {0, 0, 0x3C, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x06, 0x66, 0x3C, 0, 0, 0}},
    /* Punctuation common in human names. */
    {'-',  {0, 0, 0,    0,    0,    0,    0,    0x7E, 0,    0,    0,    0,    0,    0, 0, 0}},
};

/* The banner text is fixed: a single two-line message overlaid at the
   bottom of the face's view whenever a new face has just been enrolled.
   Split into two lines because the full string is too wide to fit on a
   240 px buffer at SCALE=3. */
static const char *const s_lines[] = { "NEW FACE", "DETECTED" };
static constexpr int      s_line_count = sizeof(s_lines) / sizeof(s_lines[0]);

static const uint8_t *find_glyph(char c)
{
    for (size_t i = 0; i < sizeof(s_font) / sizeof(s_font[0]); ++i) {
        if (s_font[i].ch == c) {
            return s_font[i].rows;
        }
    }
    return s_font[0].rows;
}

/* Width of the cell occupied by a single character. All printable glyphs
   use the full CELL_W; the space character gets a shorter cell so word
   breaks are visible without being twice as wide as letter spacing. */
static inline int cell_width_for(char c)
{
    return (c == ' ') ? SPACE_CELL_W : CELL_W;
}

/* Splat one soft circular dot into the supplied alpha buffer at
   destination centre (cx, cy). Intensity falls off as 1 - d^2 / r^2
   from the centre out to DOT_RADIUS_PX (quadratic / parabolic falloff:
   avoids a sqrt per pixel, looks the same to the eye as linear at this
   scale). Alpha is accumulated additively and clamped to 255, so
   overlapping dots in the stroke core saturate to opaque while the
   dot edge falls smoothly to the background. Takes the destination
   buffer as a parameter so the same primitive can be reused for the
   name banner without aliasing the green-banner state. */
static inline void splat_dot(uint8_t *alpha, float cx, float cy)
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
   (positive = clockwise in screen coordinates). */
static void render_lines_into(uint8_t *alpha, uint8_t *outline,
                              const char *const *lines,
                              const int *line_y_signs,
                              int n_lines, float angle_rad)
{
    memset(alpha, 0, (size_t)BANNER_W * BANNER_H);

    const float ca = cosf(angle_rad);
    const float sa = sinf(angle_rad);
    const int   cx = BANNER_W / 2;
    const int   cy = BANNER_H / 2;

    for (int li = 0; li < n_lines; ++li) {
        const char *line = lines[li];
        const int   len  = (int)strlen(line);

        /* Compute the line's total pixel width by summing per-character
           cell widths (spaces are narrower than letters) plus (len-1)
           inter-letter gaps. Done as a separate pass because we need
           the total to compute the centred left edge before drawing. */
        int line_w_px = 0;
        for (int ci = 0; ci < len; ++ci) {
            line_w_px += cell_width_for(line[ci]);
            if (ci < len - 1) {
                line_w_px += LETTER_GAP_PX;
            }
        }

        /* Top-left of the line's bounding box in local coords.
           sign == -1 -> line above centre (top of face view).
           sign == +1 -> line below centre (bottom of face view). */
        const float line_x0 = -(float)line_w_px * 0.5f;
        const float sign    = (float)line_y_signs[li];
        const float line_y0 = sign * (float)LINE_OFFSET_PX
                              - (float)CELL_H * 0.5f;

        float x_offset = 0.0f;
        for (int ci = 0; ci < len; ++ci) {
            const uint8_t *rows = find_glyph(line[ci]);
            const float    cx0  = line_x0 + x_offset;

            for (int gy = 0; gy < GLYPH_H; ++gy) {
                const uint8_t bits = rows[gy];
                if (!bits) continue;
                const float ly = line_y0 + ((float)gy + 0.5f) * (float)SCALE;
                for (int gx = 0; gx < GLYPH_W; ++gx) {
                    if (!(bits & (0x80 >> gx))) continue;
                    const float lx = cx0 + ((float)gx + 0.5f) * (float)SCALE;
                    const float dst_cx = ca * lx - sa * ly + (float)cx;
                    const float dst_cy = sa * lx + ca * ly + (float)cy;
                    splat_dot(alpha, dst_cx, dst_cy);
                }
            }

            x_offset += (float)cell_width_for(line[ci]) + (float)LETTER_GAP_PX;
        }
    }

    /* Dilate the alpha mask into the outline buffer: each pixel is the
       maximum of `alpha` over a (2R+1) x (2R+1) window centred on it.
       Keeps the soft-edge gradient at the dilation boundary so the
       outline AA-matches the foreground stroke. Naive 2D max is fine
       here -- 240*240*25 = ~1.4M comparisons, well under 1 ms. */
    const int R = OUTLINE_RADIUS_PX;
    for (int y = 0; y < BANNER_H; ++y) {
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
}

/* Shared worker: composite a banner alpha + outline pair into an
   RGB565BE frame buffer with the given foreground colour. Outline is
   black (all-zero channels); foreground is the 5/6/5 triple passed in.
   See banner_compose_overlay() for the per-channel math. */
static void compose_with(uint16_t *frame, int width, int height,
                         const uint8_t *alpha, const uint8_t *outline,
                         int fg_r5, int fg_g6, int fg_b5)
{
    const int w = (width  < BANNER_W) ? width  : BANNER_W;
    const int h = (height < BANNER_H) ? height : BANNER_H;

    for (int y = 0; y < h; ++y) {
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
    render_lines_into(s_alpha, s_outline, s_lines, signs,
                      s_line_count, angle_rad);
}

/* Map an arbitrary input character to a glyph we know how to render.
   We're working with a fixed-up bitmap font that covers A-Z, 0-9, a
   handful of punctuation. ASCII lowercase folds to uppercase; anything
   else outside the supported set becomes a space (so unknown bytes
   just open up letter-spacing instead of dropping the glyph and
   shifting later letters around). */
static inline char name_char_filter(char c)
{
    if (c >= 'a' && c <= 'z') return (char)(c - 'a' + 'A');
    if ((c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') ||
        c == ' ' || c == '-' || c == '.' || c == '\'') {
        return c;
    }
    return ' ';
}

/* How many filtered characters fit on a single 240-px banner line.
   We compute the budget assuming every char is a full CELL_W (worst
   case -- spaces are narrower, so the actual line we render is always
   <= this width). Conservatively leaves ~10 px of margin per side. */
static constexpr int NAME_BANNER_MAX_CHARS =
    (BANNER_W - 20) / (CELL_W + LETTER_GAP_PX);  /* ~10 chars */

void banner_render_name(const char *name, float angle_rad)
{
    if (!s_name_alpha || !s_name_outline || !name) {
        return;
    }

    /* Filter + uppercase into a local buffer, capping at the line's
       character budget. Empty / all-space input is treated as a
       request to clear the banner: render an empty line so the
       compose loop has nothing to draw. */
    char filtered[NAME_BANNER_MAX_CHARS + 1];
    int  out = 0;
    for (int i = 0; name[i] != '\0' && out < NAME_BANNER_MAX_CHARS; ++i) {
        filtered[out++] = name_char_filter(name[i]);
    }
    filtered[out] = '\0';

    const char *const lines[1] = { filtered };
    const int         signs[1] = { +1 };          /* bottom edge */
    render_lines_into(s_name_alpha, s_name_outline,
                      lines, signs, 1, angle_rad);
}

void banner_compose_overlay(uint16_t *frame, int width, int height)
{
    if (!s_alpha || !s_outline || !frame) {
        return;
    }
    compose_with(frame, width, height, s_alpha, s_outline,
                 BANNER_FG_R5, BANNER_FG_G6, BANNER_FG_B5);
}

void banner_compose_name_overlay(uint16_t *frame, int width, int height)
{
    if (!s_name_alpha || !s_name_outline || !frame) {
        return;
    }
    compose_with(frame, width, height, s_name_alpha, s_name_outline,
                 NAME_FG_R5, NAME_FG_G6, NAME_FG_B5);
}
