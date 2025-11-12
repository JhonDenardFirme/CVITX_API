#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/ubuntu/cvitx"
SRC="$ROOT/api/analysis"
DST="$ROOT/video_analysis/main_worker/utils"
install -D -m 0644 "$SRC/engine.py"        "$DST/engine.py"
install -D -m 0644 "$SRC/bbox_utils.py"    "$DST/bbox_utils.py"
install -D -m 0644 "$SRC/utils_color.py"   "$DST/utils_color.py"
install -D -m 0644 "$SRC/utils_plate.py"   "$DST/utils_plate.py"
install -D -m 0644 "$SRC/utils_runtime.py" "$DST/utils_runtime.py"
echo "Synced shared engine + helpers → main_worker/utils/."
