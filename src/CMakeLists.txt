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

# webserver.cpp serves src/web/index.html as a pre-gzipped byte array
# baked into src/web/index_html_data.h. Regenerate that header from the
# HTML at build time: gzip runs on the build host so the firmware ships
# compressed bytes and serves them with a Content-Encoding: gzip header.
#
# Under PlatformIO this ran as a `pre:` extra script; the IDF-native
# equivalent is a custom command wired in as a build dependency of this
# component. The generator rewrites the header only when the HTML
# actually changes, so a no-op build doesn't force a webserver.cpp
# recompile.
idf_build_get_property(python PYTHON)
set(web_html   "${COMPONENT_DIR}/web/index.html")
set(web_header "${COMPONENT_DIR}/web/index_html_data.h")
add_custom_command(
    OUTPUT "${web_header}"
    COMMAND ${python} "${CMAKE_SOURCE_DIR}/scripts/embed_web_html.py" "${web_html}" "${web_header}"
    DEPENDS "${web_html}" "${CMAKE_SOURCE_DIR}/scripts/embed_web_html.py"
    COMMENT "Embedding web/index.html -> index_html_data.h (gzip)"
    VERBATIM)
add_custom_target(web_html_header DEPENDS "${web_header}")
add_dependencies(${COMPONENT_LIB} web_html_header)
