#pragma once

#include "esp_err.h"

// Start the on-device HTTP server. Must be called AFTER wifi_init()
// has successfully associated -- the netif's listen socket only binds
// after a DHCP lease is in hand. Idempotent: a second call after
// success is a no-op.
//
// Exposed endpoints:
//   GET  /                          -- single-page UI
//   GET  /api/faces                 -- JSON list of known faces
//   GET  /api/face/<id>/thumb       -- 24-bit BMP thumbnail
//   POST /api/face/<id>/name        -- set the human-readable name
//
// Returns ESP_OK once the listener is up.
esp_err_t webserver_init(void);
