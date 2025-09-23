from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from uuid import UUID
import os, json
import boto3
from sqlalchemy import create_engine, text

router = APIRouter()

# --- inline API key guard (no external imports) ---
def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_KEY", "")
    if not expected:  # dev fallback
        return True
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")
    return True
# --------------------------------------------------

engine = create_engine(os.getenv("DB_URL"), pool_pre_ping=True, future=True)
sqs = boto3.client("sqs", region_name=os.getenv("AWS_REGION", "ap-southeast-2"))
SQS_VIDEO_QUEUE_URL = os.getenv("SQS_VIDEO_QUEUE_URL")

class AnalyzeBody(BaseModel):
    video_id: UUID
    workspace_code: str | None = None  # optional

def fetch_video_row(video_id: str):
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT id::text, workspace_id::text, workspace_code, camera_code,
                   s3_key_raw, frame_stride, recorded_at
              FROM videos WHERE id = :vid
        """), {"vid": str(video_id)}).mappings().first()
    return row

def fetch_workspace_code(workspace_id: str) -> str | None:
    if not workspace_id:
        return None
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT workspace_code FROM workspaces WHERE id = :wid
        """), {"wid": workspace_id}).first()
    return row[0] if row else None

@router.post("/api/jobs/analyze_video", dependencies=[Depends(require_api_key)])
def analyze_video(body: AnalyzeBody):
    # Canonical support: derive workspace_code when omitted
    # (keeps legacy {video_id, workspace_code} working, and allows canonical {video_id})
    if not getattr(body, "workspace_code", None):
        from sqlalchemy import text as _text  # safe local import
        with engine.begin() as _conn:
            _ws = _conn.execute(_text("""
                SELECT COALESCE(v.workspace_code, w.workspace_code, '')
                  FROM videos v
                  LEFT JOIN workspaces w ON w.id = v.workspace_id
                 WHERE v.id = :vid
            """), {"vid": str(body.video_id)}).scalar_one_or_none()
        if not _ws:
            raise HTTPException(status_code=400, detail="workspace_code missing and not found in DB")
        body.workspace_code = _ws
    v = fetch_video_row(str(body.video_id))
    if not v:
        raise HTTPException(404, "video not found")

    ws_code = (body.workspace_code or v["workspace_code"] or fetch_workspace_code(v["workspace_id"]))
    if not ws_code:
        raise HTTPException(400, "workspace_code missing (not in request nor DB)")

    with engine.begin() as conn:
        status_row = conn.execute(text("SELECT status FROM videos WHERE id=:vid"), {"vid": str(body.video_id)}).first()
        status = status_row[0] if status_row else None
        if status in ("queued","processing"):
            return {"ok": True, "queued": True, "note": f"already {status}"}
        conn.execute(text("""
            UPDATE videos
               SET status='queued', updated_at=NOW(), workspace_code=:code
             WHERE id=:vid
        """), {"vid": str(body.video_id), "code": ws_code})

    payload = {
        "event": "PROCESS_VIDEO",
        "video_id": str(v["id"]),
        "workspace_id": v["workspace_id"],
        "workspace_code": ws_code,
        "camera_code": v["camera_code"],
        "s3_key_raw": v["s3_key_raw"],
        "frame_stride": int(v["frame_stride"] or 3),
        "recordedAt": v["recorded_at"].isoformat() if v["recorded_at"] else None,
    }
    resp = sqs.send_message(QueueUrl=SQS_VIDEO_QUEUE_URL, MessageBody=json.dumps(payload))
    return {"ok": True, "queued": True, "message_id": resp.get("MessageId")}
