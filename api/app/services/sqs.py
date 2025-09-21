import json
import boto3
from ..config import settings

_sqs = boto3.client("sqs", region_name=settings.aws_region)

def send_json(queue_url: str | None, payload: dict) -> dict:
    if not queue_url:
        # Safe guard: if .env is missing the queue url, make it obvious
        return {"error": "QUEUE_URL_NOT_CONFIGURED"}
    resp = _sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(payload),
    )
    return {"message_id": resp.get("MessageId")}

def send_process_video(payload: dict) -> dict:
    return send_json(settings.sqs_video_queue_url, payload)

def send_snapshot_task(payload: dict) -> dict:
    return send_json(settings.sqs_snapshot_queue_url, payload)
