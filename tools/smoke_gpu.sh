#!/usr/bin/env bash
set -e
CD=/home/ubuntu/cvitx/api
VENV="$CD/.venv"
echo "== GPU driver =="; nvidia-smi || true
echo "== Python venv =="; source "$VENV/bin/activate"
python - <<'PY'
import torch, pkgutil
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("ultralytics:", bool(pkgutil.find_loader("ultralytics")))
print("deep_sort_realtime:", bool(pkgutil.find_loader("deep_sort_realtime")))
PY
echo "== API health =="; curl -fsS http://localhost/healthz && echo " OK" || echo " NOT OK"
echo "== Listener :80 =="; sudo ss -tulpn | egrep ':(80)\s' || true
echo "== SQS (video) =="; aws sqs get-queue-attributes --queue-url "https://sqs.ap-southeast-2.amazonaws.com/118730128890/cvitx-video-tasks" --attribute-names ApproximateNumberOfMessages --region ap-southeast-2 | jq .
echo "== DONE =="
