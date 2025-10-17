from app.auth.deps import require_user
import os, uuid, boto3
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# ---- TEMP auth (replace later)

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET  = os.getenv("S3_BUCKET") or os.getenv("BUCKET")
S3_EXPIRE  = int(os.getenv("S3_PRESIGN_EXPIRE", "900"))
if not (AWS_REGION and S3_BUCKET): raise RuntimeError("Missing AWS config")
s3 = boto3.client("s3", region_name=AWS_REGION, endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com")

DB_URL = os.getenv("DB_URL")
if not DB_URL: raise RuntimeError("DB_URL not set")
SQLA_URL = DB_URL.replace("postgresql+psycopg2", "postgresql")
engine = create_engine(SQLA_URL, pool_pre_ping=True)

router = APIRouter(prefix="/workspaces", tags=["workspaces-files"])

# ---- Allowed content types for workspace attachments (adjust as you need)
ALLOWED_WS_CT = {
    # images
    "image/png", "image/jpeg", "image/webp",
    # docs
    "text/plain", "application/pdf",
    # videos (if you plan to allow user uploads)
    "video/mp4", "video/quicktime"
}

class PresignIn(BaseModel):
    filename: str
    content_type: str

class PresignOut(BaseModel):
    key: str
    url: str

@router.post("/{workspace_id}/files/presign", response_model=PresignOut)
def presign_workspace_upload(workspace_id: str, body: PresignIn, me=Depends(require_user)):
    # allowed type check
    if body.content_type not in ALLOWED_WS_CT:
        raise HTTPException(415, "Unsupported media type")

    # ensure workspace exists & is active
    with engine.begin() as conn:
        w = conn.execute(
            text("SELECT 1 FROM workspaces WHERE id=:id AND deleted_at IS NULL"),
            {"id": workspace_id},
        ).fetchone()
    if not w:
        raise HTTPException(404, "Workspace not found")

    obj_id = str(uuid.uuid4())
    key = f"workspaces/{workspace_id}/attachments/{obj_id}/{body.filename}"
    url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": S3_BUCKET,
            "Key": key,
            "ContentType": body.content_type,
            "Tagging": f"kind=attachment&workspace_id={workspace_id}&uploader_user_id={me.id}",
        },
        ExpiresIn=S3_EXPIRE,
        HttpMethod="PUT",
    )
    return {"key": key, "url": url}

# Server-side validate object then UPSERT metadata (unique on s3_key)
@router.post("/{workspace_id}/files/commit")
def commit_workspace_file(
    workspace_id: str,
    key: str,
    content_type: str | None = None,
    size_bytes: int | None = None,
    me=Depends(require_user),
):
    if not key.startswith(f"workspaces/{workspace_id}/attachments/"):
        raise HTTPException(400, "Key not in workspace attachments")

    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=key)
    except Exception:
        raise HTTPException(400, "S3 object not found")

    actual_ct = head.get("ContentType")
    actual_sz = int(head.get("ContentLength") or 0)
    if content_type and content_type != actual_ct:
        raise HTTPException(400, f"Content-Type mismatch: {actual_ct}")
    if size_bytes is not None and int(size_bytes) != actual_sz:
        raise HTTPException(400, f"Size mismatch: {actual_sz}")

    with engine.begin() as conn:
        row = conn.execute(
            text("""
                INSERT INTO public.workspace_files (workspace_id, uploader_user_id, s3_key, content_type, size_bytes)
                VALUES (:ws, :uid, :k, :ct, :sz)
                ON CONFLICT (s3_key)
                DO UPDATE SET content_type=EXCLUDED.content_type, size_bytes=EXCLUDED.size_bytes
                RETURNING id
            """),
            {"ws": workspace_id, "uid": me.id, "k": key, "ct": actual_ct, "sz": actual_sz},
        ).fetchone()
        file_id = row[0] if row else conn.execute(
            text("SELECT id FROM public.workspace_files WHERE s3_key=:k"), {"k": key}
        ).scalar()
    return {"ok": True, "id": file_id}

# List files (newest first)
@router.get("/{workspace_id}/files")
def list_workspace_files(workspace_id: str, me=Depends(require_user)):
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, s3_key, content_type, size_bytes, created_at
                FROM public.workspace_files
                WHERE workspace_id=:ws
                ORDER BY created_at DESC
            """),
            {"ws": workspace_id},
        ).mappings().all()
    return rows

# GET URL; optional download=1 for Content-Disposition
@router.get("/{workspace_id}/files/{file_id}/url")
def get_workspace_file_url(workspace_id: str, file_id: str, download: int = 0, me=Depends(require_user)):
    with engine.begin() as conn:
        key = conn.execute(
            text("SELECT s3_key FROM public.workspace_files WHERE id=:id AND workspace_id=:ws"),
            {"id": file_id, "ws": workspace_id},
        ).scalar()
    if not key:
        raise HTTPException(404, "Not found")

    params = {"Bucket": S3_BUCKET, "Key": key}
    if download:
        from os.path import basename
        params["ResponseContentDisposition"] = f'attachment; filename="{basename(key)}"'

    url = s3.generate_presigned_url(
        "get_object", Params=params, ExpiresIn=S3_EXPIRE, HttpMethod="GET"
    )
    return {"key": key, "url": url}

# Delete (S3 + DB)
@router.delete("/{workspace_id}/files/{file_id}")
def delete_workspace_file(workspace_id: str, file_id: str, me=Depends(require_user)):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT s3_key FROM public.workspace_files WHERE id=:id AND workspace_id=:ws"),
            {"id": file_id, "ws": workspace_id},
        ).fetchone()
        if not row:
            raise HTTPException(404, "Not found")
        key = row[0]
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
        conn.execute(text("DELETE FROM public.workspace_files WHERE id=:id"), {"id": file_id})
    return {"ok": True}
