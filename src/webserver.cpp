// On-device HTTP server. Serves a one-page web UI that lets the user
// inspect every face the detector has enrolled this boot and (re)name
// them.
//
// Design notes:
//
//   * Thumbnails are stored internally as RGB565 big-endian (the
//     camera's native format). We expand them to 24-bit BMP at
//     serve time -- ~13 KB per 64x64 thumb -- because every browser
//     can render BMP without any JS-side decoding and we don't have
//     to pull in a JPEG encoder just for this.
//
//   * URI dispatch uses ESP-IDF's wildcard matcher (`*`). One GET
//     handler covers /api/face/<id>/thumb and one POST handler covers
//     /api/face/<id>/name; each parses the id out of req->uri.
//
//   * Cross-task safety is delegated to the face_db_* accessors in
//     face_ai.cpp. We never hold a face DB lock across a socket
//     write -- handlers snapshot what they need into stack /
//     heap-local buffers, drop the lock, then stream.

#include "webserver.h"
#include "face_ai.h"
#include "web/index_html_data.h"

#include "esp_check.h"
#include "esp_err.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "mdns.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

static const char *TAG = "web";
static httpd_handle_t s_server = nullptr;
static bool           s_mdns_up = false;

// Bring up Multicast DNS so the on-device server is reachable at
// `<HOSTNAME>.local` from any client on the same L2 segment. Idempotent.
// Failures are non-fatal -- the server still works by IP.
static constexpr const char *HOSTNAME      = "esp-face-detect";
static constexpr const char *INSTANCE_NAME = "ESP32-S3-EYE face detect";

static void mdns_bringup(void)
{
    if (s_mdns_up) {
        return;
    }
    if (mdns_init() != ESP_OK) {
        ESP_LOGW(TAG, "mdns_init failed; reachable by IP only");
        return;
    }
    mdns_hostname_set(HOSTNAME);
    mdns_instance_name_set(INSTANCE_NAME);
    // Advertise the HTTP service so clients that browse for _http._tcp
    // (most mDNS browsers, e.g. avahi-browse, Bonjour) discover us
    // automatically. The port matches httpd's default 80.
    mdns_service_add(nullptr, "_http", "_tcp", 80, nullptr, 0);
    s_mdns_up = true;
    ESP_LOGI(TAG, "mDNS up: http://%s.local/", HOSTNAME);
}
// --- HTML page ------------------------------------------------------

// The UI source of truth is src/web/index.html. A PlatformIO pre-script
// (scripts/embed_web_html.py) regenerates src/web/index_html_data.h on
// every build, embedding the file as a raw-string literal in the
// web:: namespace. That gives us a single-round-trip self-contained
// page without dragging in SPIFFS or littering this file with HTML.

// --- helpers --------------------------------------------------------

// Parse the integer id out of "/api/face/<id>/<suffix>". Returns -1
// on any parsing failure or out-of-range id.
static int parse_face_id(const char *uri)
{
    int id = -1;
    // sscanf with a fixed prefix is enough -- the URIs are short and
    // well-controlled; we don't need a full URI parser here.
    if (std::sscanf(uri, "/api/face/%d", &id) != 1) {
        return -1;
    }
    return id;
}

// Expand one 64-pixel-wide RGB565BE row into a 24-bit BGR BMP row.
// BMP rows are 4-byte aligned; for FACE_THUMB_DIM * 3 = 192 we're
// already aligned, but we pad explicitly so any future thumb size
// still works.
static void bmp_pack_row(const uint16_t *src, int width, uint8_t *dst)
{
    const uint8_t *sb = (const uint8_t *)src;
    int           out = 0;
    for (int x = 0; x < width; ++x) {
        // BE: byte[0] = RRRRRGGG, byte[1] = GGGBBBBB.
        const uint8_t hi = sb[x * 2];
        const uint8_t lo = sb[x * 2 + 1];
        const uint8_t r5 = (uint8_t)(hi >> 3);
        const uint8_t g6 = (uint8_t)(((hi & 0x07) << 3) | (lo >> 5));
        const uint8_t b5 = (uint8_t)(lo & 0x1F);
        // 5 / 6 / 5 -> 8 / 8 / 8 via bit replication.
        const uint8_t r8 = (uint8_t)((r5 << 3) | (r5 >> 2));
        const uint8_t g8 = (uint8_t)((g6 << 2) | (g6 >> 4));
        const uint8_t b8 = (uint8_t)((b5 << 3) | (b5 >> 2));
        // BMP pixel order is BGR.
        dst[out++] = b8;
        dst[out++] = g8;
        dst[out++] = r8;
    }
    // Pad to a multiple of 4 bytes (BMP requirement).
    while (out & 3) {
        dst[out++] = 0;
    }
}

// --- handlers -------------------------------------------------------

static esp_err_t handle_root(httpd_req_t *req)
{
    // The page is embedded as a pre-gzipped blob (see
    // scripts/embed_web_html.py and src/web/index_html_data.h). Every
    // browser we target accepts gzip; the IDF httpd is too minimal to do
    // content negotiation and we ship no plaintext fallback. If a future
    // client without gzip support shows up, sniff Accept-Encoding here
    // and decompress before sending.
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
    return httpd_resp_send(req,
                           reinterpret_cast<const char *>(web::index_html_gz),
                           web::index_html_gz_len);
}

static esp_err_t handle_faces_list(httpd_req_t *req)
{
    // Worst-case body size: MAX_KNOWN_FACES (32) entries of
    // {"id":NN,"name":"<=31 chars JSON-escaped"} ~= 80 bytes each,
    // plus brackets/commas. 4 KB is comfortable and stack-allocatable.
    char   buf[4096];
    size_t pos = 0;
    pos += std::snprintf(buf + pos, sizeof(buf) - pos, "[");

    // Snapshot under a single lock so a concurrent DELETE can't shift
    // the gallery underneath us mid-loop (which would skip entries or
    // emit a stale name for a freshly-deleted id).
    face_db_entry_t entries[MAX_KNOWN_FACES];
    const int       n = face_db_snapshot(entries, MAX_KNOWN_FACES);
    bool            first = true;
    for (int i = 0; i < n; ++i) {
        const face_db_entry_t &e = entries[i];
        // Escape backslash and double quote in the name; everything
        // else 7-bit ASCII passes through. We don't expect non-ASCII
        // from the form input, but truncate to FACE_NAME_MAX-1 above
        // is the only enforced sanitisation.
        char esc[FACE_NAME_MAX * 2 + 1];
        size_t o = 0;
        for (size_t k = 0; e.name[k] != '\0' && k < FACE_NAME_MAX - 1; ++k) {
            const char c = e.name[k];
            if (c == '"' || c == '\\') {
                esc[o++] = '\\';
            }
            esc[o++] = c;
        }
        esc[o] = '\0';

        pos += std::snprintf(buf + pos, sizeof(buf) - pos,
                             "%s{\"id\":%d,\"name\":\"%s\",\"enrolled_ms\":%u}",
                             first ? "" : ",",
                             e.idx, esc, (unsigned)e.enrolled_ms);
        first = false;
        if (pos >= sizeof(buf) - 128) {
            break; // safety -- truncate rather than overflow.
        }
    }
    pos += std::snprintf(buf + pos, sizeof(buf) - pos, "]");

    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, buf, pos);
}

static esp_err_t handle_face_thumb(httpd_req_t *req)
{
    const int id = parse_face_id(req->uri);
    if (id < 0) {
        return httpd_resp_send_404(req);
    }

    // Peek the face's enrolled_ms so we can build a strong ETag without
    // touching the thumbnail bytes yet. enrolled_ms is set at enrol
    // time and never mutated, so it uniquely identifies the thumb
    // contents at this slot.
    face_db_entry_t meta;
    if (!face_db_get_entry(id, &meta)) {
        return httpd_resp_send_404(req);
    }
    char etag[40];
    std::snprintf(etag, sizeof(etag), "\"i%d-e%u\"",
                  id, (unsigned)meta.enrolled_ms);

    // If the client already has this thumb cached, short-circuit before
    // doing the heap-alloc + DB lock + BMP encode. IDF's httpd treats a
    // missing header as ESP_ERR_NOT_FOUND, so a fresh client (no
    // If-None-Match) falls through to the full-body path. RFC 7232
    // allows comma-separated lists in the header; we don't bother
    // parsing -- in practice browsers send a single value matching the
    // last ETag we returned.
    char inm[64];
    if (httpd_req_get_hdr_value_str(req, "If-None-Match",
                                    inm, sizeof(inm)) == ESP_OK &&
        std::strcmp(inm, etag) == 0) {
        httpd_resp_set_status(req, "304 Not Modified");
        httpd_resp_set_hdr(req, "ETag", etag);
        httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
        return httpd_resp_send(req, nullptr, 0);
    }

    static constexpr int W = FACE_THUMB_DIM;
    static constexpr int H = FACE_THUMB_DIM;
    static constexpr int ROW_BYTES = ((W * 3 + 3) / 4) * 4;
    static constexpr int PIX_BYTES = ROW_BYTES * H;
    static constexpr int FILE_BYTES = 54 + PIX_BYTES;

    // Heap-allocate the thumb buffer. Stack-allocating 64*64*2 = 8 KB
    // here used to silently overflow the httpd task stack (cfg.stack_size
    // is 8192) and the BMP serializer would then read back zeroed /
    // garbage memory, which decoded as a solid black square in the
    // browser. unique_ptr keeps the cleanup automatic on every exit
    // path (success, copy_thumb failure, or chunk-send abort).
    auto thumb = std::make_unique<uint16_t[]>(static_cast<size_t>(W) * H);
    uint32_t fresh_enrolled_ms = 0;
    if (!face_db_copy_thumb(id, thumb.get(), static_cast<size_t>(W) * H,
                            &fresh_enrolled_ms)) {
        return httpd_resp_send_404(req);
    }
    // If a delete+re-enrol slipped into this slot between the ETag
    // peek above and this copy, the bytes we just read belong to a
    // different face. Refresh the ETag so the client's next If-None-
    // Match reflects what we're actually serving.
    if (fresh_enrolled_ms != meta.enrolled_ms) {
        std::snprintf(etag, sizeof(etag), "\"i%d-e%u\"",
                      id, (unsigned)fresh_enrolled_ms);
    }

    // BMP file + DIB header. Hand-packed little-endian so we don't
    // depend on host endianness assumptions.
    uint8_t hdr[54] = {0};
    auto put_u16 = [](uint8_t *p, uint16_t v) {
        p[0] = (uint8_t)(v & 0xff);
        p[1] = (uint8_t)((v >> 8) & 0xff);
    };
    auto put_u32 = [](uint8_t *p, uint32_t v) {
        p[0] = (uint8_t)(v & 0xff);
        p[1] = (uint8_t)((v >> 8) & 0xff);
        p[2] = (uint8_t)((v >> 16) & 0xff);
        p[3] = (uint8_t)((v >> 24) & 0xff);
    };
    hdr[0] = 'B'; hdr[1] = 'M';
    put_u32(hdr + 2, FILE_BYTES);
    put_u32(hdr + 10, 54);
    put_u32(hdr + 14, 40);
    put_u32(hdr + 18, (uint32_t)W);
    // Negative height -> top-down row order, matches our buffer layout.
    put_u32(hdr + 22, (uint32_t)(-H));
    put_u16(hdr + 26, 1);
    put_u16(hdr + 28, 24);
    put_u32(hdr + 34, PIX_BYTES);

    httpd_resp_set_type(req, "image/bmp");
    httpd_resp_set_hdr(req, "ETag", etag);
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_send_chunk(req, (const char *)hdr, sizeof(hdr));

    uint8_t row[ROW_BYTES];
    for (int y = 0; y < H; ++y) {
        bmp_pack_row(thumb.get() + static_cast<size_t>(y) * W, W, row);
        httpd_resp_send_chunk(req, (const char *)row, ROW_BYTES);
    }
    return httpd_resp_send_chunk(req, nullptr, 0);
}

static esp_err_t handle_face_name(httpd_req_t *req)
{
    const int id = parse_face_id(req->uri);
    if (id < 0) {
        return httpd_resp_send_404(req);
    }

    // Names are short by spec (FACE_NAME_MAX). Allow a bit of
    // surrounding whitespace / a trailing newline that some HTTP
    // clients tack on and still fit on the stack.
    char body[FACE_NAME_MAX + 16] = {0};
    int  recv_total = 0;
    while (recv_total < (int)sizeof(body) - 1) {
        const int r = httpd_req_recv(req, body + recv_total,
                                     sizeof(body) - 1 - recv_total);
        if (r <= 0) {
            if (r == HTTPD_SOCK_ERR_TIMEOUT) {
                continue;
            }
            break;
        }
        recv_total += r;
    }
    body[recv_total] = '\0';

    // Trim trailing whitespace / CR / LF so a form-style trailing
    // newline doesn't pollute the stored name.
    while (recv_total > 0) {
        const char c = body[recv_total - 1];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') {
            body[--recv_total] = '\0';
        } else {
            break;
        }
    }

    if (!face_db_set_name(id, body)) {
        return httpd_resp_send_404(req);
    }
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, "{\"ok\":true}", HTTPD_RESP_USE_STRLEN);
}

// DELETE /api/face/<id> -- remove a face from the in-memory DB.
// face_db_delete() shifts subsequent entries down by one, so any
// browser cards holding the old index need to refresh via /api/faces
// after this returns. The web UI does that via load() on success.
static esp_err_t handle_face_delete(httpd_req_t *req)
{
    const int id = parse_face_id(req->uri);
    if (id < 0) {
        return httpd_resp_send_404(req);
    }
    if (!face_db_delete(id)) {
        return httpd_resp_send_404(req);
    }
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, "{\"ok\":true}", HTTPD_RESP_USE_STRLEN);
}

// --- public entrypoint ----------------------------------------------

esp_err_t webserver_init(void)
{
    if (s_server) {
        return ESP_OK;
    }

    // Bring up mDNS first so the hostname is advertised the moment the
    // listener accepts its first connection. mdns_bringup is idempotent
    // and non-fatal -- a failure just leaves us reachable by raw IP.
    mdns_bringup();

    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    // Bump stack / max URIs over the IDF defaults so the dynamic JSON
    // builder and the wildcard handlers fit comfortably.
    cfg.stack_size       = 8192;
    cfg.max_uri_handlers = 8;
    // Pin the server task to core 1 (same core as face_ai). Default is
    // tskNO_AFFINITY, which lets it float onto core 0 and contend with
    // cam_hal + the LCD render task -- exactly the pipeline we want
    // kept clean for live preview. face_ai is the only consumer on
    // core 1 and it sleeps NO_FACE_DELAY_MS (40 ms) between misses,
    // so HTTP requests slot into those gaps cleanly without
    // perturbing the preview path.
    cfg.core_id          = 1;
    // Enable wildcard URI matching so a single handler can serve
    // /api/face/<id>/thumb for every id.
    cfg.uri_match_fn     = httpd_uri_match_wildcard;

    ESP_RETURN_ON_ERROR(httpd_start(&s_server, &cfg), TAG, "httpd_start");

    const httpd_uri_t uri_root = {
        .uri      = "/",
        .method   = HTTP_GET,
        .handler  = handle_root,
        .user_ctx = nullptr,
    };
    const httpd_uri_t uri_list = {
        .uri      = "/api/faces",
        .method   = HTTP_GET,
        .handler  = handle_faces_list,
        .user_ctx = nullptr,
    };
    // Both /thumb and /name share the /api/face/* prefix. We register
    // a GET handler that serves the thumbnail, a POST handler that
    // updates the name, and a DELETE handler that removes the entry;
    // the wildcard matcher dispatches by method.
    const httpd_uri_t uri_face_get = {
        .uri      = "/api/face/*",
        .method   = HTTP_GET,
        .handler  = handle_face_thumb,
        .user_ctx = nullptr,
    };
    const httpd_uri_t uri_face_post = {
        .uri      = "/api/face/*",
        .method   = HTTP_POST,
        .handler  = handle_face_name,
        .user_ctx = nullptr,
    };
    const httpd_uri_t uri_face_delete = {
        .uri      = "/api/face/*",
        .method   = HTTP_DELETE,
        .handler  = handle_face_delete,
        .user_ctx = nullptr,
    };

    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_server, &uri_root),
                        TAG, "register /");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_server, &uri_list),
                        TAG, "register /api/faces");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_server, &uri_face_get),
                        TAG, "register GET /api/face/*");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_server, &uri_face_post),
                        TAG, "register POST /api/face/*");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(s_server, &uri_face_delete),
                        TAG, "register DELETE /api/face/*");

    ESP_LOGI(TAG, "HTTP server up on port %u", (unsigned)cfg.server_port);
    return ESP_OK;
}
