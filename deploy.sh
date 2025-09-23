#!/usr/bin/env bash
set -euo pipefail
cd ~/cvitx
echo "[deploy] pulling..."
git pull
echo "[deploy] installing deps..."
source ~/cvitx/api/.venv/bin/activate
pip install -r ~/cvitx/api/requirements.txt
echo "[deploy] restarting services..."
sudo systemctl restart cvitx-api
sudo systemctl restart cvitx-worker-video || true
sudo systemctl restart cvitx-worker-snapshot || true
echo "[deploy] done."
