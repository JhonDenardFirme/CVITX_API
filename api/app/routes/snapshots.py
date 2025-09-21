# /home/ubuntu/cvitx/api/app/routes/snapshots.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

from ..deps import require_api_key
from ..services.sqs import send_snapshot_task

router = APIRouter()

class EnqueueSnapshot(BaseModel):
    video_id: UUID
    workspace_id: UUID
    workspace_code: str
    camera_code: str
    s3_key_raw: str
    recordedAt: datetime

@router.post("/snapshots/enqueue", dependencies=[Depends(require_api_key)])
def enqueue_snapshot(body: EnqueueSnapshot):
    job = {
        "event": "SNAPSHOT_READY",
        "video_id": str(body.video_id),
        "workspace_id": str(body.workspace_id),
        "workspace_code": body.workspace_code,
        "camera_code": body.camera_code,
        "s3_key_raw": body.s3_key_raw,
        "recordedAt": body.recordedAt.isoformat(),
    }
    res = send_snapshot_task(job)
    if not res.get("message_id"):
        raise HTTPException(status_code=500, detail=str(res))
    return {"ok": True, "message_id": res["message_id"], "event": "SNAPSHOT_READY"}

