// Wi-Fi credentials consumed by src/wifi.cpp.
//
// This file is tracked in version control with PLACEHOLDER values so a
// fresh clone still compiles. Each developer is expected to:
//
//   1. Edit the macros below to point at their network.
//   2. Run `git update-index --skip-worktree src/wifi_credentials.h`
//      (once, per clone). After that command, git treats this file as
//      "always clean" -- local edits never appear in `git status` /
//      `git diff` and never get committed.
//   3. To pick up an upstream change to this file, temporarily undo
//      the skip-worktree flag:
//        git update-index --no-skip-worktree src/wifi_credentials.h
//        git pull
//        # re-enter your local credentials
//        git update-index --skip-worktree src/wifi_credentials.h
//
// Why this pattern instead of .gitignore + a .example template?
// .gitignore would leave the file untracked, so the build wouldn't
// have anything to compile against on a fresh clone -- you'd hit a
// missing-header error before you got as far as the Wi-Fi connect
// attempt. Committing a placeholder + skip-worktree gives us a
// guaranteed-compilable default with local secrets that never leak.
#pragma once

#define WIFI_SSID      "your-ssid-here"
#define WIFI_PASSWORD  "your-password-here"
