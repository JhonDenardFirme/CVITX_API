# CVITX Operations Quickstart

## Environments / dependencies
- Python venv at `api/.venv`
- AWS role with S3+SQS+CloudWatch permissions
- NVIDIA driver + CUDA for GPU nodes (ULTRALYTICS_DEVICE=cuda:0)

## Core environment variables
### AWS / S3
- `AWS_REGION` — e.g., `ap-southeast-2`
- `S3_BUCKET`  — e.g., `cvitx-uploads-dev-jdfirme`

### Video identifiers
- `WORKSPACE_ID` (UUID)
- `WORKSPACE_CODE` (e.g., CTX1005)
- `VIDEO_ID` (UUID)
- `FRAME_STRIDE` (int >=1)
- `S3_KEY_RAW` = `demo_user/<wid>/<vid>/raw/<file>.mp4`

### YOLO / GPU
- `ULTRALYTICS_DEVICE`  — `cuda:0` or `cpu`
- `YOLO_WEIGHTS`        — model path or blank for default
- `YOLO_IMGSZ`          — 640
- `YOLO_CONF`           — 0.25–0.5 (scene dependent)
- `YOLO_IOU`            — 0.45

### Detector hygiene (optional)
- `YOLO_ALLOWED_CLS`           — CSV of allowed classes; blank means no filter
- `YOLO_CROWD_DET_THRESHOLD`   — if det count exceeds this in a frame, apply bump
- `YOLO_CROWD_CONF_BUMP`       — how much to bump the conf floor (0.0–0.1)
- `YOLO_SMALL_MIN_SIDE`        — px for “small” boxes
- `YOLO_SMALL_CONF_MIN`        — extra conf floor for small boxes

### Tracker (DeepSort)
- `DEEPSORT_MAX_AGE` (50)
- `DEEPSORT_N_INIT` (3)
- `DEEPSORT_MAX_IOU_DISTANCE` (0.7)
- `DEEPSORT_NN_BUDGET` (200)
- `DEEPSORT_EMBEDDER` (mobilenet)

### Snapshot quality / neighbor suppression
- `SNAPSHOT_SIZE` (640)
- `JPG_QUALITY` (95)
- `MIN_TRACK_AGE` (3)
- `SNAPSHOT_MARGIN` (0.15)
- `SNAPSHOT_MARGIN_MIN` (0.08)
- `SNAPSHOT_MARGIN_MAX` (0.20)
- `SNAPSHOT_NEIGHBOR_IOU` (0.03)
- `SNAPSHOT_MARGIN_SHRINK_STEP` (0.8)
- `SNAPSHOT_MARGIN_MAX_STEPS` (5)
- `SNAPSHOT_MIN_SIDE` (32)
- `SNAPSHOT_SHARPEN` (1)
- `SNAPSHOT_SHARPEN_AMOUNT` (1.2)
- `SNAPSHOT_SHARPEN_RADIUS` (1.0)
- `SNAPSHOT_SHARPEN_THRESHOLD` (3)
- `SNAPSHOT_SR` (0|1), `SNAPSHOT_SR_MODEL`, `SNAPSHOT_SR_UPSCALE` (2..4)

### Progress & heartbeat
- `PROGRESS_LOG_INTERVAL_SEC` (10)
- `PROGRESS_LOG_JSON` (0|1)
- `SQS_VIS_HEARTBEAT_SEC` (60; 0 disables)
- `SQS_VIS_TIMEOUT` (300)
- `DB_STATUS_HEARTBEAT` (0|1)

### CloudWatch push (log-based)
- `CW_NAMESPACE` (default `cvitx`)
- `CW_DIM_APP`   (default `worker_video`)
- `CW_DIM_ENV`   (default `dev`)

## Recommended presets
### GPU nodes (T4/A10)
- `ULTRALYTICS_DEVICE=cuda:0`
- `FRAME_STRIDE=3–5`, `YOLO_IMGSZ=640`, `YOLO_CONF=0.25–0.35`

### CPU nodes
- `FORCE_CPU=1`
- `FRAME_STRIDE=5–8`, `YOLO_IMGSZ=512`, `YOLO_CONF=0.35–0.50`

## Run patterns
- Poller: `python -m workers.video_runner --poll`
- One-shot: see runbook in Parent Roadmap

## Contracts
See `tools/verify_queue_contracts.py --help` and Parent Roadmap contract section.

