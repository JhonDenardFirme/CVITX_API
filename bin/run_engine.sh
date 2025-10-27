#!/usr/bin/env bash
set -Eeuo pipefail
APP_ROOT="${APP_ROOT:-$HOME/cvitx}"
cd "$APP_ROOT"
python3 -m venv .venv 2>/dev/null || true
. .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null
# timm/pillow are enough to import the engine; torch is needed to run
python - <<'PY'
print("[ok] engine import check")
import sys; sys.path.append("api/analysis")
import importlib.util, runpy
spec = importlib.util.spec_from_file_location("ec2_code_engine", "api/analysis/ec2_code_engine.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("[ok] engine file present and importable")
PY
