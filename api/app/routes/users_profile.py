from app.auth.deps import require_user
import os, boto3
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# ---- TEMP auth (replace later)

router = APIRouter(prefix="/users/me", tags=["users"])

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET  = os.getenv("S3_BUCKET") or os.getenv("BUCKET")
S3_EXPIRE  = int(os.getenv("S3_PRESIGN_EXPIRE", "900"))
if not (AWS_REGION and S3_BUCKET):
    raise RuntimeError("Missing AWS_REGION or S3 bucket env")
s3 = boto3.client("s3", region_name=AWS_REGION, endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com")

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL not set")
SQLA_URL = DB_URL.replace("postgresql+psycopg2", "postgresql")
engine = create_engine(SQLA_URL, pool_pre_ping=True)

class PresignOut(BaseModel):
    key: str
    url: str

class UrlOut(BaseModel):
    url: str | None

# We force PNG for avatars. (If you want JPEG/WEBP later, make a tiny allowlist check here.)
@router.post("/avatar/presign", response_model=PresignOut)
def presign_avatar_upload(me=Depends(require_user)):
    key = f"users/{me.id}/avatar.png"
    url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": S3_BUCKET,
            "Key": key,
            "ContentType": "image/png",
            "Tagging": f"kind=avatar&owner_user_id={me.id}",
        },
        ExpiresIn=S3_EXPIRE,
        HttpMethod="PUT",
    )
    return {"key": key, "url": url}

# Server-side object validation (type/size) + DB write
@router.post("/avatar/commit")
def commit_avatar(
    key: str,
    content_type: str | None = None,
    size_bytes: int | None = None,
    me=Depends(require_user),
):
    if not key.startswith(f"users/{me.id}/"):
        raise HTTPException(400, "Invalid avatar key")

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
        conn.execute(
            text("UPDATE users SET avatar_s3_key=:k, updated_at=now() WHERE id=:id"),
            {"k": key, "id": me.id},
        )
    return {"ok": True}

# Optional download=1 -> Content-Disposition attachment
@router.get("/avatar/url", response_model=UrlOut)
def get_avatar_url(download: int = 0, me=Depends(require_user)):
    with engine.begin() as conn:
        key = conn.execute(
            text("SELECT avatar_s3_key FROM users WHERE id=:id"),
            {"id": me.id},
        ).scalar()
    if not key:
        return {"url": None}

    params = {"Bucket": S3_BUCKET, "Key": key}
    if download:
        params["ResponseContentDisposition"] = 'attachment; filename="avatar.png"'

    url = s3.generate_presigned_url(
        "get_object", Params=params, ExpiresIn=S3_EXPIRE, HttpMethod="GET"
    )
    return {"url": url}

# Delete avatar (S3 + DB)
@router.delete("/avatar")
def delete_avatar(me=Depends(require_user)):
    with engine.begin() as conn:
        key = conn.execute(
            text("SELECT avatar_s3_key FROM users WHERE id=:id"),
            {"id": me.id},
        ).scalar()
        if key:
            s3.delete_object(Bucket=S3_BUCKET, Key=key)
            conn.execute(
                text("UPDATE users SET avatar_s3_key=NULL, updated_at=now() WHERE id=:id"),
                {"id": me.id},
            )
    return {"ok": True}

@router.get("/workspaces")
def list_my_workspaces(me=Depends(require_user)):
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, code, title, description, created_at
                  FROM workspaces
                 WHERE owner_user_id = :uid
                   AND deleted_at IS NULL
              ORDER BY created_at DESC NULLS LAST
            """),
            {"uid": me.id},
        ).fetchall()
    return [
        {
            "id": str(r[0]),
            "code": r[1],
            "title": r[2],
            "description": r[3],
            "created_at": (r[4].isoformat() if r[4] else None),
        }
        for r in rows
    ]

# --- BEGIN: add GET /users/me base ---
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import HTTPException

class UserMe(BaseModel):
    id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    affiliation_name: Optional[str] = None
    role: str
    is_active: bool
    force_password_reset: bool
    created_at: datetime
    updated_at: datetime
    avatar_s3_key: Optional[str] = None

@router.get("", response_model=UserMe)
def get_me(me=Depends(require_user)):
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT id, email, first_name, last_name, affiliation_name, role, is_active,
                   force_password_reset, created_at, updated_at, avatar_s3_key
            FROM users
            WHERE id = :id
        """), {"id": me.id}).fetchone()
    if not row:
        raise HTTPException(404, "User not found")
    return {
        "id": str(row[0]),
        "email": row[1],
        "first_name": row[2],
        "last_name": row[3],
        "affiliation_name": row[4],
        "role": row[5],
        "is_active": row[6],
        "force_password_reset": row[7],
        "created_at": row[8],
        "updated_at": row[9],
        "avatar_s3_key": row[10],
    }
# --- END: add GET /users/me base ---
