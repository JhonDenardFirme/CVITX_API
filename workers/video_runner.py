"""
CVITX · Legacy video_runner bootstrap (YOLO + DeepSORT)

This module keeps the old systemd entrypoint:
    python -m workers.video_runner

but forwards all real work to the new YOLO worker at:
    video_analysis.yolo_worker.worker.main

That worker:
- Reads CONFIG from video_analysis.worker_config
- Uses SQS_VIDEO_QUEUE_URL / SQS_SNAPSHOT_QUEUE_URL
- Downloads from S3_BUCKET
- Emits SNAPSHOT_READY events

Do NOT put any queue URL logic here; CONFIG is the single source of truth.
"""

from video_analysis.yolo_worker.worker import main as _run

def main() -> None:
    _run()

if __name__ == "__main__":
    main()
