import os, uuid, re
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from app.auth.deps import require_user
from app.services.workspaces import get_workspace_code_by_id
from app.services.sqs import send_process_video

# DB boilerplate (match style in routes/workspaces.py)
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL not set")
SQLA_URL = DB_URL.replace("postgresql+psycopg2","postgresql")
engine = create_engine(SQLA_URL, pool_pre_ping=True)

router = APIRouter(prefix="/workspaces", tags=["videos"])

# ------------------ Models ------------------

class VideoOut(BaseModel):
    id: str
    workspace_id: str
    workspace_code: Optional[str] = None
    file_name: Optional[str] = None
    camera_label: Optional[str] = None
    camera_code: Optional[str] = None
    s3_key_raw: Optional[str] = None
    frame_stride: Optional[int] = None
    status: str
    error_msg: Optional[str] = None
    recorded_at: Optional[datetime] = None
    updated_at: datetime
    processing_started_at: Optional[datetime] = None
    processing_finished_at: Optional[datetime] = None

class VideoCreate(BaseModel):
    file_name: Optional[str] = None
    camera_label: Optional[str] = None
    camera_code: Optional[str] = None  # accepted if provided; otherwise auto-generate
    recorded_at: Optional[datetime] = None
    s3_key_raw: str
    frame_stride: Optional[int] = 3

class VideoPatch(BaseModel):
    file_name: Optional[str] = None
    camera_label: Optional[str] = None
    recorded_at: Optional[datetime] = None

# ------------------ Helpers ------------------

def _owns_workspace(conn, wid: str, uid: str) -> bool:
    r = conn.execute(text("""
        SELECT 1 FROM workspaces
         WHERE id = :wid
           AND owner_user_id = :uid
           AND deleted_at IS NULL
         LIMIT 1
    """), {"wid": wid, "uid": uid}).fetchone()
    return bool(r)

def _row_to_out(r) -> dict:
    # r is a mappings() row
    return {
        "id": str(r["id"]),
        "workspace_id": str(r["workspace_id"]),
        "workspace_code": r["workspace_code"],
        "file_name": r["file_name"],
        "camera_label": r["camera_label"],
        "camera_code": r["camera_code"],
        "s3_key_raw": to_s3_uri(os.environ.get("S3_BUCKET",""), r["s3_key_raw"]),
        "frame_stride": r["frame_stride"],
        "status": r["status"] if isinstance(r["status"], str) else str(r["status"]),
        "error_msg": r["error_msg"],
        "recorded_at": r["recorded_at"],
        "updated_at": r["updated_at"],
        "processing_started_at": r["processing_started_at"],
        "processing_finished_at": r["processing_finished_at"],
    }

def _select_cols():
    return """
        id, workspace_id, workspace_code, file_name, camera_label, camera_code,
        s3_key_raw, frame_stride, status, error_msg, recorded_at, updated_at,
        processing_started_at, processing_finished_at
    """

def _next_camera_code(conn, wid: str) -> str:
    rows = conn.execute(text("""
        SELECT camera_code
          FROM videos
         WHERE workspace_id = :wid
           AND camera_code IS NOT NULL
    """), {"wid": wid}).fetchall()
    maxn = 0
    for (code,) in rows:
        if not code:
            continue
        m = re.match(r'^CAM(\d+)$', str(code).strip())
        if m:
            try:
                n = int(m.group(1))
                if n > maxn:
                    maxn = n
            except:
                pass
    return f"CAM{maxn + 1 if maxn >= 1 else 1}"

# ------------------ Endpoints ------------------

@router.get("/{workspace_id}/videos", response_model=List[VideoOut])
def list_videos(workspace_id: str, me = Depends(require_user)):
    with engine.begin() as conn:
        if not _owns_workspace(conn, workspace_id, str(me.id)):
            raise HTTPException(404, "Workspace not found")
        rows = conn.execute(text(f"""
            SELECT {_select_cols()}
              FROM videos
             WHERE workspace_id = :wid
          ORDER BY recorded_at DESC NULLS LAST,
                   updated_at DESC NULLS LAST,
                   id
        """), {"wid": workspace_id}).mappings().all()
    return [_row_to_out(r) for r in rows]

@router.get("/{workspace_id}/videos/{video_id}", response_model=VideoOut)
def get_video(workspace_id: str, video_id: str, me = Depends(require_user)):
    with engine.begin() as conn:
        if not _owns_workspace(conn, workspace_id, str(me.id)):
            raise HTTPException(404, "Workspace not found")
        r = conn.execute(text(f"""
            SELECT {_select_cols()}
              FROM videos
             WHERE id = :vid
               AND workspace_id = :wid
             LIMIT 1
        """), {"vid": video_id, "wid": workspace_id}).mappings().first()
        if not r:
            raise HTTPException(404, "Video not found")
    return _row_to_out(r)

from urllib.parse import urlparse
def to_s3_uri(bucket: str, key_or_uri: str) -> str:
    if "://" in key_or_uri:
        u = urlparse(key_or_uri)
        if u.scheme == "s3" and u.netloc and u.path:
            return key_or_uri
    return f"s3://{bucket}/{key_or_uri.lstrip('/')}"


@router.post("/{workspace_id}/videos", response_model=VideoOut)
def create_video(workspace_id: str, body: VideoCreate, me = Depends(require_user)):
    if not body.s3_key_raw:
        raise HTTPException(400, "s3_key_raw is required")

    with engine.begin() as conn:
        if not _owns_workspace(conn, workspace_id, str(me.id)):
            raise HTTPException(404, "Workspace not found")

        # workspace_code from service (UI shouldn't send it)
        wcode = get_workspace_code_by_id(workspace_id)

        # camera_code: accept if provided, else generate next CAM{n}
        ccode = body.camera_code or _next_camera_code(conn, workspace_id)

        vid = str(uuid.uuid4())
        r = conn.execute(text(f"""
            INSERT INTO videos (
                id, workspace_id, workspace_code, file_name, camera_label,
                camera_code, s3_key_raw, frame_stride, recorded_at, status, updated_at
            ) VALUES (
                :id, :wid, :wcode, :fname, :clabel,
                :ccode, :s3, :stride, :rec, 'uploaded', now()
            )
            RETURNING {_select_cols()}
        """), {
            "id": vid,
            "wid": workspace_id,
            "wcode": wcode,
            "fname": body.file_name,
            "clabel": body.camera_label,
            "ccode": ccode,
            "s3": body.s3_key_raw,
            "stride": body.frame_stride,
            "rec": body.recorded_at,
        }).mappings().first()

    return _row_to_out(r)

@router.patch("/{workspace_id}/videos/{video_id}", response_model=VideoOut)
def edit_video(workspace_id: str, video_id: str, body: VideoPatch, me = Depends(require_user)):
    with engine.begin() as conn:
        if not _owns_workspace(conn, workspace_id, str(me.id)):
            raise HTTPException(404, "Workspace not found")

        st = conn.execute(text("""
            SELECT status FROM videos
             WHERE id = :vid AND workspace_id = :wid
             LIMIT 1
        """), {"vid": video_id, "wid": workspace_id}).fetchone()
        if not st:
            raise HTTPException(404, "Video not found")
        if str(st[0]) != "uploaded":
            raise HTTPException(409, "Only 'uploaded' videos can be edited")

        sets, params = [], {"vid": video_id, "wid": workspace_id}
        if body.file_name is not None:
            sets.append("file_name = :fname"); params["fname"] = body.file_name
        if body.camera_label is not None:
            sets.append("camera_label = :clabel"); params["clabel"] = body.camera_label
        if body.recorded_at is not None:
            sets.append("recorded_at = :rec"); params["rec"] = body.recorded_at

        if not sets:
            r = conn.execute(text(f"""
                SELECT {_select_cols()}
                  FROM videos
                 WHERE id = :vid AND workspace_id = :wid
                 LIMIT 1
            """), params).mappings().first()
            return _row_to_out(r)

        sets.append("updated_at = now()")
        r = conn.execute(text(f"""
            UPDATE videos
               SET {", ".join(sets)}
             WHERE id = :vid AND workspace_id = :wid
         RETURNING {_select_cols()}
        """), params).mappings().first()

    return _row_to_out(r)

@router.delete("/{workspace_id}/videos/{video_id}", status_code=204)
def delete_video(workspace_id: str, video_id: str, me = Depends(require_user)):
    with engine.begin() as conn:
        if not _owns_workspace(conn, workspace_id, str(me.id)):
            raise HTTPException(404, "Workspace not found")

        # delete detections first to avoid orphans
        conn.execute(text("DELETE FROM detections WHERE video_id = :vid"), {"vid": video_id})
        res = conn.execute(text("""
            DELETE FROM videos
             WHERE id = :vid AND workspace_id = :wid
        """), {"vid": video_id, "wid": workspace_id})
        if res.rowcount == 0:
            raise HTTPException(404, "Video not found")
    return Response(status_code=204)

@router.post("/{workspace_id}/videos/{video_id}/enqueue")
def enqueue_video(workspace_id: str, video_id: str, me = Depends(require_user)):
    with engine.begin() as conn:
        if not _owns_workspace(conn, workspace_id, str(me.id)):
            raise HTTPException(404, "Workspace not found")

        # set queued only if currently uploaded (prevents double-click)
        r = conn.execute(text("""
            UPDATE videos
               SET status = 'queued', updated_at = now()
             WHERE id = :vid
               AND workspace_id = :wid
               AND status = 'uploaded'
         RETURNING id, workspace_id, file_name, camera_label, camera_code,
                   s3_key_raw, frame_stride, recorded_at
        """), {"vid": video_id, "wid": workspace_id}).mappings().first()

        if not r:
            raise HTTPException(409, "Video is not in 'uploaded' state")

        wcode = get_workspace_code_by_id(workspace_id) or ""

        payload = {
            "event": "PROCESS_VIDEO",
            "video_id": str(r["id"]),
            "workspace_id": str(workspace_id),
            "workspace_code": wcode,
            "camera_code": r["camera_code"],
            "s3_key_raw": r["s3_key_raw"],
            "frame_stride": int(r["frame_stride"] or 3),
            "recordedAt": (r["recorded_at"].isoformat() if r["recorded_at"] else None),
        }

        res = send_process_video(payload)
        if not res or not res.get("message_id"):
            # revert if enqueue fails
            conn.execute(text("""
                UPDATE videos
                   SET status = 'uploaded', updated_at = now()
                 WHERE id = :vid AND workspace_id = :wid
            """), {"vid": r["id"], "wid": workspace_id})
            raise HTTPException(502, "Failed to enqueue video")

    return {"ok": True, "message_id": res["message_id"], "status": "queued"}

# ------------------ Streaming URL (presigned GET) ------------------

@router.get("/{workspace_id}/videos/{video_id}/url")
def get_video_stream_url(workspace_id: str, video_id: str, me = Depends(require_user)):
    """
    Returns a short-lived presigned GET URL for the video's s3_key_raw so the frontend
    can play it directly in a <video> tag (with Range support, assuming S3 CORS is set).
    """
    # Lazy imports to avoid touching module imports
    import os, boto3
    from app.config import settings
    from sqlalchemy import text

    # Resolve config with safe defaults
    aws_region = getattr(settings, "aws_region", os.getenv("AWS_REGION", "ap-southeast-1"))
    s3_bucket = getattr(settings, "s3_bucket", os.getenv("S3_BUCKET"))
    ttl_sec_env = os.getenv("S3_SIGNED_GET_TTL", "3600")
    try:
        ttl_sec = int(ttl_sec_env)
    except Exception:
        ttl_sec = 3600
    # S3 presigned GET supports up to 7 days (604800); cap if someone sets larger
    if ttl_sec > 604800:
        ttl_sec = 604800

    if not s3_bucket:
        raise HTTPException(500, "S3_BUCKET is not configured")

    with engine.begin() as conn:
        if not _owns_workspace(conn, workspace_id, str(me.id)):
            raise HTTPException(404, "Workspace not found")

        row = conn.execute(text("""
            SELECT s3_key_raw, file_name
              FROM videos
             WHERE id = :vid AND workspace_id = :wid
             LIMIT 1
        """), {"vid": video_id, "wid": workspace_id}).mappings().first()
        if not row:
            raise HTTPException(404, "Video not found")

        key = row["s3_key_raw"]
        if not key:
            raise HTTPException(422, "Video has no s3_key_raw")

    # Create a client and presign
    s3 = boto3.client("s3", region_name=aws_region)
    # Force a sane content type for browsers; override if you ever store non-mp4
    resp_ct = "video/mp4"

    try:
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": s3_bucket,
                "Key": key,
                # These response headers help the browser/player
                "ResponseContentType": resp_ct,
                # Optional: let the browser cache briefly
                "ResponseCacheControl": f"private, max-age={ttl_sec}",
            },
            ExpiresIn=ttl_sec,
        )
    except Exception as e:
        raise HTTPException(500, f"Presign failed: {e}")

    return {"url": url, "content_type": resp_ct, "expires_in": ttl_sec}

# EDIT HERE — safe test at 2025-10-16T02:59:18Z UTC
