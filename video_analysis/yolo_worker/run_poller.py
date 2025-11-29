# File: /home/ubuntu/cvitx/video_analysis/yolo_worker/run_poller.py
"""
CVITX · Video Analysis Monorepo — YOLO Video Worker Runner
Directory: /home/ubuntu/cvitx/video_analysis/yolo_worker
Filename : run_poller.py

Purpose
-------
Tiny, deterministic entrypoint so systemd (or local dev) can launch the YOLO worker with:
    python -m video_analysis.yolo_worker.run_poller

It simply imports the worker’s main loop and runs it. All configuration, AWS clients,
and logging live in worker_config.py and worker_utils/common.py, not here.

================================================================================
🟡 AWS SETUP REMINDER (once per environment)
   • S3 bucket must exist: s3://cvitx-uploads-dev-jdfirme
   • SQS queues must exist (referenced by env/CONFIG):
       - Video tasks    : $SQS_VIDEO_QUEUE_URL        (PROCESS_VIDEO / PROCESS_VIDEO_DB)
       - Snapshot tasks : $SQS_SNAPSHOT_QUEUE_URL     (SNAPSHOT_READY)
     with DLQs + RedrivePolicy (maxReceiveCount≈5), VisibilityTimeout≈300s, LongPolling=10s
   • IAM role for this service needs:
       - s3:GetObject on demo_user/*/raw/*
       - s3:PutObject on demo_user/*/snapshots/*
       - sqs:ReceiveMessage/DeleteMessage/ChangeMessageVisibility on video-tasks
       - sqs:SendMessage on snapshot-tasks
   • ExecStart (systemd) should be:
       /home/ubuntu/cvitx/api/.venv/bin/python -m video_analysis.yolo_worker.run_poller
     with ENV including DATABASE_URL/DB_URL, S3_BUCKET, SQS_VIDEO_QUEUE_URL, SQS_SNAPSHOT_QUEUE_URL.
================================================================================
"""

from video_analysis.yolo_worker.worker import main as _run
from video_analysis.worker_utils.common import log  # already used in worker.py


if __name__ == "__main__":
    log.info("[yolo-runner] starting YOLO video worker (run_poller)")
    _run()
