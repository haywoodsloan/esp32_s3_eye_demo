"""
PIO pre-script: pack the ESP-WHO face-detect/recognition .espdl model
blobs and register them as extra flash images so `pio run -t upload`
writes them to their dedicated partitions.

Why this is needed: the human_face_detect / human_face_recognition
components ship CMake rules that (a) call esp-dl's `pack_espdl_models.py`
to concatenate one-or-more raw .espdl files into a single packed blob,
then (b) hook that blob into either the `flash` CMake target (for
FLASH_PARTITION mode) or as an embedded .S asm source (for FLASH_RODATA
mode). PlatformIO drives the build via SCons and only invokes the elf
build target, so neither hook fires. The result: empty
$BUILD_DIR/espdl_models/ and an empty partition on the device, which
crashes face_ai when dl::Model fails to load.

We sidestep that by packing the same source models ourselves into the
same paths the components would have produced, then appending those
paths to SCons's FLASH_EXTRA_IMAGES (which the espressif32 platform
*does* read when assembling the esptool command line).

Must be registered as a `pre:` script in platformio.ini so the env
mutation happens before the platform's main.py expands
FLASH_EXTRA_IMAGES into UPLOADERFLAGS.

Partition offsets MUST match partitions.csv (human_face_det / human_face_feat).
"""

import subprocess
import sys
from pathlib import Path

Import("env")  # noqa: F821  - injected by SCons

PROJECT_DIR = Path(env["PROJECT_DIR"])  # noqa: F821
BUILD_DIR   = Path(env.subst("$BUILD_DIR"))  # noqa: F821

ESP_DL_DIR  = PROJECT_DIR / "managed_components" / "espressif__esp-dl"
DETECT_DIR  = PROJECT_DIR / "managed_components" / "espressif__human_face_detect"
RECOG_DIR   = PROJECT_DIR / "managed_components" / "espressif__human_face_recognition"

PACK_SCRIPT = ESP_DL_DIR / "fbs_loader" / "pack_espdl_models.py"

# Target = esp32s3. Models for s3 live in models/s3/. If we ever build
# for a different chip, the matching subfolder name would differ.
MODELS_SUBDIR = "models/s3"

# Source models bundled with each component. The detect pipeline is
# two-stage (MSR proposal + MNP refinement), recognition is single-model
# (MFN). These selections match the Kconfig defaults we set in
# sdkconfig.defaults - change both together if you swap models.
DETECT_SOURCES = [
    DETECT_DIR / MODELS_SUBDIR / "human_face_detect_msr_s8_v1.espdl",
    DETECT_DIR / MODELS_SUBDIR / "human_face_detect_mnp_s8_v1.espdl",
]
RECOG_SOURCES = [
    RECOG_DIR / MODELS_SUBDIR / "human_face_feat_mfn_s8_v1.espdl",
]

# Output filenames must match what the upstream CMake would have
# produced, because the partition contents are looked up by name on
# device (human_face_det / human_face_feat) and the *files* are flashed
# to those offsets.
OUT_DIR     = BUILD_DIR / "espdl_models"
DETECT_OUT  = OUT_DIR / "human_face_detect.espdl"
RECOG_OUT   = OUT_DIR / "human_face_feat.espdl"


def _needs_rebuild(out_path, sources):
    """Skip pack if output is newer than every source - keeps incremental
    builds fast (pack takes ~1 s but adds up across rebuilds)."""
    if not out_path.exists():
        return True
    out_mtime = out_path.stat().st_mtime
    return any(s.stat().st_mtime > out_mtime for s in sources)


def _pack(out_path, sources):
    for s in sources:
        if not s.exists():
            raise FileNotFoundError("ESP-DL model source missing: %s" % s)
    if not _needs_rebuild(out_path, sources):
        print("espdl pack: %s up-to-date, skipping" % out_path.name)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(PACK_SCRIPT),
        "--model_path", *[str(s) for s in sources],
        "--out_file",   str(out_path),
    ]
    print("espdl pack:", " ".join(cmd))
    subprocess.check_call(cmd)


_pack(DETECT_OUT, DETECT_SOURCES)
_pack(RECOG_OUT,  RECOG_SOURCES)

# Register the packed blobs for upload. SCons substitutes $BUILD_DIR when
# the upload command is finally executed, so passing the symbolic path is
# fine (and keeps the SCons dependency machinery happy if anything ever
# starts caring about it).
env.Append(  # noqa: F821
    FLASH_EXTRA_IMAGES=[
        ("0x410000", "$BUILD_DIR/espdl_models/human_face_detect.espdl"),
        ("0x450000", "$BUILD_DIR/espdl_models/human_face_feat.espdl"),
    ]
)
