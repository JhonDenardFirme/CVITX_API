# CVITX Video Analysis
Top-level package for the YOLO→DeepSORT snapshotter and main model workers.

- `yolo_worker/` — SQS poller that emits 640×640 snapshots to S3 + SNAPSHOT_READY to SQS
- `main_worker/` — consumes SNAPSHOT_READY, runs Baseline/CMT, writes DB results
- `worker_utils/` — common helpers (logging, schemas, S3/SQS wrappers)
- `schemas/` — JSON Schemas for PROCESS_VIDEO and SNAPSHOT_READY
