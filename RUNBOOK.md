# CVITX Runbook (dev)
- Tail API: sudo journalctl -u cvitx-api -f
- Tail worker: sudo journalctl -u cvitx-worker-video -f
- Restart: sudo systemctl restart cvitx-api cvitx-worker-video
- DB last 5: psql -P pager=off "$PGURL" -c "SELECT id,status,updated_at FROM videos ORDER BY updated_at DESC LIMIT 5;"
- SQS depth: aws sqs get-queue-attributes --queue-url <url> --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible
- S3 snapshots: aws s3 ls s3://<bucket>/demo_user/<ws_id>/<video_id>/snapshots/
