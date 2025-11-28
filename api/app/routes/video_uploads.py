# File: api/app/routes/video_uploads.py

import os
import uuid
from datetime import datetime
from typing import Optional

import boto3
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from app.auth.deps import require_user
from app.config import settings
from app.services.sqs import send_json

# ─────────────────────────────────────────────────────────────
# DB & S3 / SQS wiring
# ─────────────────────────────────────────────────────────────

DB_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("Missing DATABASE_URL/DB_URL for video uploads")
DB_URL = DB_URL.replace("postgresql+psycopg2", "postgresql")

engine = create_engine(DB_URL, pool_pre_ping=True)

s3 = boto3.client("s3", region_name=settings.aws_region)

# Raw video layout: {S3_RAW_PREFIX}/{workspace_id}/{video_id}/raw/{filename}
S3_RAW_PREFIX = os.getenv("S3_VIDEO_RAW_PREFIX", "demo_user")

SQS_VIDEO_QUEUE_URL = os.getenv("SQS_VIDEO_QUEUE_URL")

# Default max size (bytes) – override via env if needed
MAX_VIDEO_BYTES = int(os.getenv("VIDEO_UPLOAD_MAX_BYTES", str(5 * 1024 * 1024 * 1024)))  # 5GB


router = APIRouter(prefix="/workspaces", tags=["video-uploads"])


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _assert_workspace(conn, workspace_id: str, user_id: str) -> None:
    """
    Ensure the workspace exists and belongs to this user.
    """
    row = conn.execute(
        text(
            """
            SELECT 1
              FROM workspaces
             WHERE id = :wid
               AND owner_user_id = :uid
               AND deleted_at IS NULL
            """
        ),
        {"wid": workspace_id, "uid": user_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found")


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────

class VideoPresignIn(BaseModel):
    filename: str
    content_type: str
    file_size_bytes: int
    camera_code: str
    camera_label: Optional[str] = None
    frame_stride: int = 3
    recorded_at: Optional[datetime] = None
    # Optional hint; stored in videos.workspace_code and forwarded to workers
    workspace_code: Optional[str] = None


class VideoCommitIn(BaseModel):
    videoId: str
    s3KeyRaw: str
    fileName: str
    frameStride: int = 3
    recordedAt: Optional[datetime] = None
    cameraCode: str
    cameraLabel: Optional[str] = None
    workspaceCode: Optional[str] = None
    fileSizeBytes: Optional[int] = None
    contentType: Optional[str] = None


class VideoEnqueueIn(BaseModel):
    # Optional hint for downstream workers (e.g., "cmt", "baseline", "both")
    variant: Optional[str] = "cmt"


# ─────────────────────────────────────────────────────────────
# POST /workspaces/{workspace_id}/videos/presign
# ─────────────────────────────────────────────────────────────

@router.post("/{workspace_id}/videos/presign")
def presign_video(workspace_id: str, body: VideoPresignIn, me=Depends(require_user)):
    """
    Step 1: Allocate a video_id and return a presigned PUT URL for the raw MP4.

    - Validates workspace ownership.
    - Enforces a max size guard.
    - Does NOT create the videos row yet (that happens on commit).
    """
    if body.file_size_bytes <= 0 or body.file_size_bytes > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file size; max {MAX_VIDEO_BYTES} bytes",
        )

    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

    video_id = str(uuid.uuid4())
    filename = os.path.basename(body.filename)
    if not filename:
        filename = f"video-{video_id}.mp4"

    s3_key_raw = f"{S3_RAW_PREFIX}/{workspace_id}/{video_id}/raw/{filename}"

    try:
        url = s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": settings.s3_bucket,
                "Key": s3_key_raw,
                "ContentType": body.content_type,
            },
            ExpiresIn=3600,
            HttpMethod="PUT",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {e}") from e

    return {
        "videoId": video_id,
        "workspaceId": workspace_id,
        "workspaceCode": body.workspace_code,
        "cameraCode": body.camera_code,
        "cameraLabel": body.camera_label,
        "s3KeyRaw": s3_key_raw,
        "presignedUrl": url,
        "expiresIn": 3600,
        "frameStride": body.frame_stride,
        "recordedAt": body.recorded_at.isoformat() if body.recorded_at else None,
    }


# ─────────────────────────────────────────────────────────────
# POST /workspaces/{workspace_id}/videos/commit
# ─────────────────────────────────────────────────────────────

@router.post("/{workspace_id}/videos/commit")
def commit_video(workspace_id: str, body: VideoCommitIn, me=Depends(require_user)):
    """
    Step 2: After the MP4 is uploaded to S3 via the presigned URL,
    this endpoint creates/updates the videos row.

    - Validates workspace.
    - Optionally verifies size & content-type against S3.
    - Upserts into videos with status='uploaded'.
    """
    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

    # Optional S3 integrity checks
    try:
        head = s3.head_object(Bucket=settings.s3_bucket, Key=body.s3KeyRaw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"S3 object not found for key {body.s3KeyRaw}") from e

    actual_len = int(head.get("ContentLength", 0))
    if body.fileSizeBytes is not None and int(body.fileSizeBytes) != actual_len:
        raise HTTPException(
            status_code=400,
            detail=f"File size mismatch (client={body.fileSizeBytes}, s3={actual_len})",
        )

    actual_ct = head.get("ContentType")
    if body.contentType and actual_ct and body.contentType != actual_ct:
        raise HTTPException(
            status_code=400,
            detail=f"Content-Type mismatch (client={body.contentType}, s3={actual_ct})",
        )

    params = {
        "id": body.videoId,
        "wid": workspace_id,
        "wcode": body.workspaceCode,
        "file_name": body.fileName,
        "camera_label": body.cameraLabel,
        "camera_code": body.cameraCode,
        "recorded_at": body.recordedAt,
        "s3_key_raw": body.s3KeyRaw,
        "frame_stride": body.frameStride,
        "uid": str(me.id),
    }

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO videos (
                    id,
                    workspace_id,
                    workspace_code,
                    file_name,
                    camera_label,
                    camera_code,
                    recorded_at,
                    s3_key_raw,
                    frame_stride,
                    status,
                    created_by_user_id
                )
                VALUES (
                    :id,
                    :wid,
                    :wcode,
                    :file_name,
                    :camera_label,
                    :camera_code,
                    :recorded_at,
                    :s3_key_raw,
                    :frame_stride,
                    'uploaded',
                    :uid
                )
                ON CONFLICT (id) DO UPDATE
                SET workspace_id   = EXCLUDED.workspace_id,
                    workspace_code = EXCLUDED.workspace_code,
                    file_name      = EXCLUDED.file_name,
                    camera_label   = EXCLUDED.camera_label,
                    camera_code    = EXCLUDED.camera_code,
                    recorded_at    = EXCLUDED.recorded_at,
                    s3_key_raw     = EXCLUDED.s3_key_raw,
                    frame_stride   = EXCLUDED.frame_stride,
                    status         = 'uploaded',
                    updated_at     = now()
                """
            ),
            params,
        )

        row = conn.execute(
            text(
                """
                SELECT id,
                       workspace_id,
                       workspace_code,
                       file_name,
                       camera_label,
                       camera_code,
                       recorded_at,
                       s3_key_raw,
                       frame_stride,
                       status,
                       created_at,
                       updated_at
                  FROM videos
                 WHERE id = :id
                """
            ),
            {"id": body.videoId},
        ).mappings().first()

    if not row:
        raise HTTPException(status_code=500, detail="Video record not found after commit")

    video = {
        "id": row["id"],
        "workspaceId": row["workspace_id"],
        "workspaceCode": row.get("workspace_code"),
        "fileName": row["file_name"],
        "cameraLabel": row.get("camera_label"),
        "cameraCode": row["camera_code"],
        "recordedAt": row["recorded_at"].isoformat() if row["recorded_at"] else None,
        "s3KeyRaw": row["s3_key_raw"],
        "frameStride": row["frame_stride"],
        "status": row["status"],
        "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
        "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }

    return {"video": video, "autoEnqueued": False}


# ─────────────────────────────────────────────────────────────
# POST /workspaces/{workspace_id}/videos/{video_id}/enqueue
# ─────────────────────────────────────────────────────────────

@router.post("/{workspace_id}/videos/{video_id}/enqueue")
def enqueue_video(
    workspace_id: str,
    video_id: str,
    body: VideoEnqueueIn,
    me=Depends(require_user),
):
    """
    Step 3: Enqueue the video for processing.

    - Loads the videos row.
    - Builds a PROCESS_VIDEO payload.
    - Sends to SQS_VIDEO_QUEUE_URL.
    - Marks videos.status='processing'.
    - Safe to call multiple times (re-runs).
    """
    if not SQS_VIDEO_QUEUE_URL:
        raise HTTPException(status_code=500, detail="Video queue not configured")

    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

        v = conn.execute(
            text(
                """
                SELECT id,
                       workspace_id,
                       workspace_code,
                       file_name,
                       camera_label,
                       camera_code,
                       recorded_at,
                       s3_key_raw,
                       frame_stride
                  FROM videos
                 WHERE id = :vid
                   AND workspace_id = :wid
                """
            ),
            {"vid": video_id, "wid": workspace_id},
        ).mappings().first()

        if not v:
            raise HTTPException(status_code=404, detail="Video not found")

        payload = {
            "event": "PROCESS_VIDEO",
            "video_id": str(v["id"]),
            "workspace_id": str(v["workspace_id"]),
            "workspace_code": v.get("workspace_code"),
            "camera_code": v["camera_code"],
            "s3_key_raw": v["s3_key_raw"],
            "frame_stride": v["frame_stride"],
            "recordedAt": v["recorded_at"].isoformat() if v["recorded_at"] else None,
        }

        # Optional hint; worker may ignore or use for variant routing
        if body.variant:
            payload["variant"] = body.variant

        resp = send_json(SQS_VIDEO_QUEUE_URL, payload)

        conn.execute(
            text(
                """
                UPDATE videos
                   SET status = 'processing',
                       updated_at = now()
                 WHERE id = :vid
                """
            ),
            {"vid": video_id},
        )

    return {
        "videoId": str(v["id"]),
        "enqueueEvent": payload["event"],
        "queueUrl": SQS_VIDEO_QUEUE_URL,
        "status": "queued",
        "sqsResponse": resp,
    }

