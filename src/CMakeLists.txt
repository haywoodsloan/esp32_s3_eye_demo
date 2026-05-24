idf_component_register(
    SRCS
        "main.cpp"
        "display.cpp"
        "camera.cpp"
        "banner.cpp"
        "face_ai.cpp"
        "wifi.cpp"
        "webserver.cpp"
    INCLUDE_DIRS "."
    REQUIRES driver esp_lcd esp_timer spiffs esp_wifi esp_netif esp_event nvs_flash esp_http_server mdns
)
