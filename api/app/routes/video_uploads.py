import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from app.auth.deps import require_user
from app.config import settings
from app.services.sqs import send_json
from app.routes.video_analysis import VideoRowOut

DB_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("Missing DATABASE_URL/DB_URL for video uploads")
DB_URL = DB_URL.replace("postgresql+psycopg2", "postgresql")
engine = create_engine(DB_URL, pool_pre_ping=True)

s3 = boto3.client("s3", region_name=settings.aws_region)
S3_RAW_PREFIX = os.getenv("S3_VIDEO_RAW_PREFIX", "demo_user")
SQS_VIDEO_QUEUE_URL = os.getenv("SQS_VIDEO_QUEUE_URL")
MAX_VIDEO_BYTES = int(
    os.getenv("VIDEO_UPLOAD_MAX_BYTES", str(5 * 1024 * 1024 * 1024))
)

router = APIRouter(prefix="/workspaces", tags=["video-uploads"])


def _assert_workspace(conn, workspace_id: str, user_id: str) -> None:
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


class VideoPresignIn(BaseModel):
    filename: str
    content_type: str
    file_size_bytes: int
    camera_code: str
    camera_label: Optional[str] = None
    frame_stride: int = 3
    recorded_at: Optional[datetime] = None
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
    variant: Optional[str] = "cmt"


class VideoUpdateIn(BaseModel):
    cameraLabel: Optional[str] = None
    recordedAt: Optional[datetime] = None


class DeleteVideoIn(BaseModel):
    confirmCameraCode: str


class VideoUrlOut(BaseModel):
    url: str
    ttl: int


class VideosListOut(BaseModel):
    workspaceId: str
    items: List[VideoRowOut]


@router.post("/{workspace_id}/videos/presign")
def presign_video(
    workspace_id: str,
    body: VideoPresignIn,
    me=Depends(require_user),
):
    if body.file_size_bytes <= 0 or body.file_size_bytes > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file size; max {MAX_VIDEO_BYTES} bytes",
        )

    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

    video_id = str(uuid.uuid4())
    filename = os.path.basename(body.filename) or f"video-{video_id}.mp4"
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate presigned URL: {e}",
        ) from e

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


@router.post("/{workspace_id}/videos/commit")
def commit_video(
    workspace_id: str,
    body: VideoCommitIn,
    me=Depends(require_user),
):
    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

    try:
        head = s3.head_object(Bucket=settings.s3_bucket, Key=body.s3KeyRaw)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"S3 object not found for key {body.s3KeyRaw}",
        ) from e

    actual_len = int(head.get("ContentLength", 0))
    if body.fileSizeBytes is not None and int(body.fileSizeBytes) != actual_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File size mismatch "
                f"(client={body.fileSizeBytes}, s3={actual_len})"
            ),
        )

    actual_ct = head.get("ContentType")
    if body.contentType and actual_ct and body.contentType != actual_ct:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Content-Type mismatch "
                f"(client={body.contentType}, s3={actual_ct})"
            ),
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
    }

    with engine.begin() as conn:
        insert_sql = """
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
                status
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
                'uploaded'
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
        conn.execute(text(insert_sql), params)

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
                       updated_at,
                       error_msg,
                       processing_started_at,
                       processing_finished_at
                  FROM videos
                 WHERE id = :id
                """
            ),
            {"id": body.videoId},
        ).mappings().first()

    if not row:
        raise HTTPException(
            status_code=500,
            detail="Video record not found after commit",
        )

    video = {
        "id": row["id"],
        "workspaceId": row["workspace_id"],
        "workspaceCode": row.get("workspace_code"),
        "fileName": row["file_name"],
        "cameraLabel": row.get("camera_label"),
        "cameraCode": row["camera_code"],
        "recordedAt": row["recorded_at"].isoformat()
        if row["recorded_at"]
        else None,
        "s3KeyRaw": row["s3_key_raw"],
        "frameStride": row["frame_stride"],
        "status": row["status"],
        "createdAt": row["created_at"].isoformat()
        if row.get("created_at")
        else None,
        "updatedAt": row["updated_at"].isoformat()
        if row.get("updated_at")
        else None,
        "errorMsg": row.get("error_msg"),
        "processingStartedAt": row["processing_started_at"].isoformat()
        if row.get("processing_started_at")
        else None,
        "processingFinishedAt": row["processing_finished_at"].isoformat()
        if row.get("processing_finished_at")
        else None,
    }

    return {"video": video, "autoEnqueued": False}


@router.get("/{workspace_id}/videos/{video_id}/url", response_model=VideoUrlOut)
def get_video_url(
    workspace_id: str,
    video_id: str,
    ttl: int = 900,
    me=Depends(require_user),
):
    """
    Returns a presigned GET URL for the raw uploaded video (for preview).
    """
    effective_ttl = max(60, min(int(ttl), 3600))

    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))
        row = conn.execute(
            text(
                """
                SELECT s3_key_raw
                  FROM videos
                 WHERE id = :vid
                   AND workspace_id = :wid
                """
            ),
            {"vid": video_id, "wid": workspace_id},
        ).mappings().first()

        if not row or not row["s3_key_raw"]:
            raise HTTPException(status_code=404, detail="Video not found")

        key = row["s3_key_raw"]

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=effective_ttl,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to presign GET: {e}",
        ) from e

    return {"url": url, "ttl": effective_ttl}


@router.get("/{workspace_id}/videos", response_model=VideosListOut)
def list_videos(
    workspace_id: str,
    me=Depends(require_user),
):
    """
    Lists all videos for a workspace.

    Returns:
    - workspaceId
    - items: VideoRowOut objects (camelCase), consistent with commit/get_video_detail.
    """
    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))
        rows = conn.execute(
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
                       updated_at,
                       error_msg,
                       processing_started_at,
                       processing_finished_at
                  FROM videos
                 WHERE workspace_id = :wid
                 ORDER BY created_at DESC
                """
            ),
            {"wid": workspace_id},
        ).mappings().all()

    items: List[VideoRowOut] = []
    for v in rows:
        items.append(
            VideoRowOut(
                id=v["id"],
                workspaceId=v["workspace_id"],
                workspaceCode=v.get("workspace_code"),
                fileName=v["file_name"],
                cameraLabel=v.get("camera_label"),
                cameraCode=v["camera_code"],
                recordedAt=v.get("recorded_at"),
                s3KeyRaw=v["s3_key_raw"],
                frameStride=v["frame_stride"],
                status=v["status"],
                createdAt=v.get("created_at"),
                updatedAt=v.get("updated_at"),
                errorMsg=v.get("error_msg"),
                processingStartedAt=v.get("processing_started_at"),
                processingFinishedAt=v.get("processing_finished_at"),
            )
        )

    return VideosListOut(workspaceId=workspace_id, items=items)


@router.post("/{workspace_id}/videos/{video_id}/enqueue")
def enqueue_video(
    workspace_id: str,
    video_id: str,
    body: VideoEnqueueIn,
    me=Depends(require_user),
):
    if not SQS_VIDEO_QUEUE_URL:
        raise HTTPException(
            status_code=500,
            detail="Video queue not configured",
        )

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

        payload: Dict[str, Any] = {
            "event": "PROCESS_VIDEO",
            "video_id": str(v["id"]),
            "workspace_id": str(v["workspace_id"]),
            "workspace_code": v.get("workspace_code"),
            "camera_code": v["camera_code"],
            "s3_key_raw": v["s3_key_raw"],
            "frame_stride": v["frame_stride"],
            "recordedAt": v["recorded_at"].isoformat()
            if v["recorded_at"]
            else None,
        }

        if body.variant:
            payload["variant"] = body.variant

        resp = send_json(SQS_VIDEO_QUEUE_URL, payload)

        conn.execute(
            text(
                """
                UPDATE videos
                   SET status = 'queued',
                       updated_at = now()
                 WHERE id = :vid
                """
            ),
            {"vid": video_id},
        )

    message_id: Optional[str] = None
    if isinstance(resp, dict):
        message_id = (
            resp.get("MessageId")
            or resp.get("MessageID")
            or resp.get("message_id")
        )

    return {
        "ok": bool(message_id),
        "status": "queued",
        "message_id": message_id,
        "videoId": str(v["id"]),
        "enqueueEvent": payload["event"],
        "queueUrl": SQS_VIDEO_QUEUE_URL,
        "sqsResponse": resp,
    }


@router.patch("/{workspace_id}/videos/{video_id}")
def update_video(
    workspace_id: str,
    video_id: str,
    body: VideoUpdateIn,
    me=Depends(require_user),
):
    """
    Partial update for a video.

    Allowed fields:
    - cameraLabel -> videos.camera_label
    - recordedAt  -> videos.recorded_at

    All other columns remain unchanged.
    """
    data = body.dict(exclude_unset=True)

    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

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
                       updated_at
                  FROM videos
                 WHERE id = :vid
                   AND workspace_id = :wid
                """
            ),
            {"vid": video_id, "wid": workspace_id},
        ).mappings().first()

        if not row:
            raise HTTPException(status_code=404, detail="Video not found")

        if not data:
            return {
                "video": {
                    "id": row["id"],
                    "workspaceId": row["workspace_id"],
                    "workspaceCode": row.get("workspace_code"),
                    "fileName": row["file_name"],
                    "cameraLabel": row.get("camera_label"),
                    "cameraCode": row["camera_code"],
                    "recordedAt": row["recorded_at"].isoformat()
                    if row["recorded_at"]
                    else None,
                    "s3KeyRaw": row["s3_key_raw"],
                    "frameStride": row["frame_stride"],
                    "status": row["status"],
                    "updatedAt": row["updated_at"].isoformat()
                    if row.get("updated_at")
                    else None,
                },
                "autoEnqueued": False,
            }

        set_clauses: List[str] = []
        params: Dict[str, Any] = {"vid": video_id, "wid": workspace_id}

        if "cameraLabel" in data:
            set_clauses.append("camera_label = :camera_label")
            params["camera_label"] = data["cameraLabel"]

        if "recordedAt" in data:
            set_clauses.append("recorded_at = :recorded_at")
            params["recorded_at"] = data["recordedAt"]

        if not set_clauses:
            return {
                "video": {
                    "id": row["id"],
                    "workspaceId": row["workspace_id"],
                    "workspaceCode": row.get("workspace_code"),
                    "fileName": row["file_name"],
                    "cameraLabel": row.get("camera_label"),
                    "cameraCode": row["camera_code"],
                    "recordedAt": row["recorded_at"].isoformat()
                    if row["recorded_at"]
                    else None,
                    "s3KeyRaw": row["s3_key_raw"],
                    "frameStride": row["frame_stride"],
                    "status": row["status"],
                    "updatedAt": row["updated_at"].isoformat()
                    if row.get("updated_at")
                    else None,
                },
                "autoEnqueued": False,
            }

        set_clauses.append("updated_at = now()")
        update_sql = f"""
            UPDATE videos
               SET {", ".join(set_clauses)}
             WHERE id = :vid
               AND workspace_id = :wid
        """

        conn.execute(text(update_sql), params)

        updated = conn.execute(
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
                       updated_at
                  FROM videos
                 WHERE id = :vid
                   AND workspace_id = :wid
                """
            ),
            {"vid": video_id, "wid": workspace_id},
        ).mappings().first()

    return {
        "video": {
            "id": updated["id"],
            "workspaceId": updated["workspace_id"],
            "workspaceCode": updated.get("workspace_code"),
            "fileName": updated["file_name"],
            "cameraLabel": updated.get("camera_label"),
            "cameraCode": updated["camera_code"],
            "recordedAt": updated["recorded_at"].isoformat()
            if updated["recorded_at"]
            else None,
            "s3KeyRaw": updated["s3_key_raw"],
            "frameStride": updated["frame_stride"],
            "status": updated["status"],
            "updatedAt": updated["updated_at"].isoformat()
            if updated.get("updated_at")
            else None,
        },
        "autoEnqueued": False,
    }


@router.delete("/{workspace_id}/videos/{video_id}")
def delete_video(
    workspace_id: str,
    video_id: str,
    body: DeleteVideoIn,
    me=Depends(require_user),
):
    """
    Deletes a video and its downstream analysis rows.

    Safety guard:
    - Requires the caller to provide confirmCameraCode matching videos.camera_code.
    - Refuses deletion if there are active analyses (processing or running).
    - Refuses deletion if the video is queued or processing at the videos.status level.
    """
    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

        video = conn.execute(
            text(
                """
                SELECT id,
                       workspace_id,
                       camera_code,
                       s3_key_raw,
                       status
                  FROM videos
                 WHERE id = :vid
                   AND workspace_id = :wid
                """
            ),
            {"vid": video_id, "wid": workspace_id},
        ).mappings().first()

        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if body.confirmCameraCode != video["camera_code"]:
            raise HTTPException(
                status_code=400,
                detail="Confirmation camera code does not match",
            )

        if video["status"] in ("queued", "processing"):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot delete video while analysis job is queued or "
                    "processing. Please wait for completion or implement an "
                    "explicit cancel flow."
                ),
            )

        active = conn.execute(
            text(
                """
                SELECT id
                  FROM video_analyses
                 WHERE video_id = :vid
                   AND workspace_id = :wid
                   AND status IN ('processing', 'running')
                """
            ),
            {"vid": video_id, "wid": workspace_id},
        ).fetchone()

        if active:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete video while analyses are still running",
            )

        analysis_ids: List[str] = [
            row["id"]
            for row in conn.execute(
                text(
                    """
                    SELECT id
                      FROM video_analyses
                     WHERE video_id = :vid
                       AND workspace_id = :wid
                    """
                ),
                {"vid": video_id, "wid": workspace_id},
            ).mappings()
        ]

        if analysis_ids:
            conn.execute(
                text(
                    "DELETE FROM video_detections WHERE analysis_id = ANY(:aids)"
                ),
                {"aids": analysis_ids},
            )
            conn.execute(
                text(
                    "DELETE FROM video_analysis_results "
                    "WHERE analysis_id = ANY(:aids)"
                ),
                {"aids": analysis_ids},
            )
            conn.execute(
                text(
                    "DELETE FROM video_analyses WHERE id = ANY(:aids)"
                ),
                {"aids": analysis_ids},
            )

        conn.execute(
            text(
                """
                DELETE FROM videos
                 WHERE id = :vid
                   AND workspace_id = :wid
                """
            ),
            {"vid": video_id, "wid": workspace_id},
        )

    if video.get("s3_key_raw"):
        try:
            s3.delete_object(
                Bucket=settings.s3_bucket,
                Key=video["s3_key_raw"],
            )
        except Exception:
            pass

    prefix = f"{S3_RAW_PREFIX}/{workspace_id}/{video_id}/"
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=settings.s3_bucket,
            Prefix=prefix,
        ):
            contents = page.get("Contents") or []
            if not contents:
                continue
            delete_keys = [{"Key": obj["Key"]} for obj in contents]
            s3.delete_objects(
                Bucket=settings.s3_bucket,
                Delete={"Objects": delete_keys, "Quiet": True},
            )
    except Exception:
        pass

    return {
        "videoId": str(video["id"]),
        "deleted": True,
    }

