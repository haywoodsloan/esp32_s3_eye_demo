cmake_minimum_required(VERSION 3.16.0)

# Application sources live in src/ rather than the IDF-default main/.
# Because src/ has its own CMakeLists.txt, listing it here makes the IDF
# build system register it directly as a single component (the supported
# "renamed main" pattern). A renamed main gets NO automatic dependency on
# every other component, so src/CMakeLists.txt declares its REQUIRES
# explicitly.
set(EXTRA_COMPONENT_DIRS "src")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Enable ESP-DL's per-inference latency breakdown (logs
# `detect: pre/model/post = <us>` and `feat: pre/model/post = <us>`),
# the signal used by the perf-tuning loop. ESP-DL hard-defines
# DL_LOG_INFER_LATENCY to 0 in dl/dl_define.hpp; this global compile
# definition overrides it, but only takes effect because that header was
# patched with an `#ifndef` guard. Both halves are required.
#
# The guard lives in the re-fetchable managed component, so after any
# `idf.py update-dependencies` it must be re-applied (see repo notes).
# To silence the per-inference logs, delete this line (app-side
# `face_ai: 1s timing:` aggregates remain regardless).
idf_build_set_property(COMPILE_DEFINITIONS "DL_LOG_INFER_LATENCY=1" APPEND)

project(esp32_s3_eye_demo)
