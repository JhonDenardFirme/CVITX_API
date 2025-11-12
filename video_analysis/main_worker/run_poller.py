# File: /home/ubuntu/cvitx/video_analysis/main_worker/run_poller.py
"""
CVITX · Video Analysis Monorepo — Main-Model Runner
Directory: /home/ubuntu/cvitx/video_analysis/main_worker
Filename : run_poller.py

What it does
------------
Thin, deterministic entrypoint that boots the Main-Model Worker and starts the
SQS poll loop for SNAPSHOT_READY messages.

How to run (both work)
----------------------
• Module mode (preferred; matches systemd ExecStart):
    python -m video_analysis.main_worker.run_poller

• Direct script:
    python /home/ubuntu/cvitx/video_analysis/main_worker/run_poller.py

================================================================================
🟡 AWS SETUP REMINDER (once per environment)
   • S3 bucket must exist: s3://cvitx-uploads-dev-jdfirme
   • SQS queue must exist:
       - Snapshot tasks : https://sqs.ap-southeast-2.amazonaws.com/118730128890/cvitx-snapshot-tasks
     with DLQ + RedrivePolicy (maxReceiveCount≈5), VisibilityTimeout≈300s, LongPolling=10s
   • IAM role for this service needs:
       - s3:GetObject on demo_user/*/snapshots/*
       - s3:PutObject on */*/*/{baseline,cmt}/*
       - sqs:ReceiveMessage/DeleteMessage/ChangeMessageVisibility on snapshot-tasks
================================================================================
"""

from __future__ import annotations

import os
import signal
import sys
import time

from video_analysis.worker_config import CONFIG
from video_analysis.worker_utils.common import log
from .worker import main as worker_main


def _handle_term(_signum, _frame) -> None:
    # Let systemd know we exit cleanly on TERM/INT.
    log.info("[runner] termination signal received; exiting…")
    sys.exit(0)


def _preflight() -> None:
    """Fail fast if required config is missing; print resolved values for transparency."""
    required = ("AWS_REGION", "S3_BUCKET", "SQS_SNAPSHOT_QUEUE_URL")
    missing = [k for k in required if not str(CONFIG.get(k) or "").strip()]
    if missing:
        log.error(f"[preflight] missing required config: {', '.join(missing)}")
        sys.exit(2)

    log.info(
        "[preflight] region=%s  bucket=%s  snapshot_queue=%s",
        CONFIG["AWS_REGION"],
        CONFIG["S3_BUCKET"],
        CONFIG["SQS_SNAPSHOT_QUEUE_URL"],
    )


def run() -> None:
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    _preflight()

    try:
        worker_main()
    except SystemExit:
        raise
    except Exception as e:
        log.error(f"[runner] fatal error: {e}", exc_info=True)
        time.sleep(1.0)
        sys.exit(1)


if __name__ == "__main__":
    run()
