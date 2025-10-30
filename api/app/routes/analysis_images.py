from fastapi import HTTPException
import os, uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from uuid import UUID
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import boto3

from uuid import UUID

def _clean_uuid_for_path(raw: str) -> str:
    s = str(raw or "")
    s = s.lstrip("=")
    s = s.split("?", 1)[0]
    s = s.split("&", 1)[0]
    try:
        return str(UUID(s))
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "invalid_analysis_id", "value": raw})


from app.auth.deps import require_user
from app.config import settings
from app.services.sqs import send_json

router = APIRouter(prefix="/workspaces", tags=["image-analyses"])

# DB + S3 clients
DB_URL = (os.getenv("DATABASE_URL") or os.getenv("DB_URL"))
if not DB_URL:
    raise RuntimeError("Missing DATABASE_URL/DB_URL")
DB_URL = DB_URL.replace("postgresql+psycopg2","postgresql")
engine = create_engine(DB_URL, pool_pre_ping=True)
s3 = boto3.client("s3", region_name=settings.aws_region)

# Config
S3_PREFIX = os.getenv("S3_IMAGE_ANALYSIS_PREFIX", "imageanalysis")
TTL = int(os.getenv("ANALYSIS_PRESIGN_TTL", "604800"))  # 7d for GET urls
Q_BASE = os.getenv("SQS_ANALYSIS_BASELINE_URL")
Q_CMT  = os.getenv("SQS_ANALYSIS_CMT_URL")

class PresignIn(BaseModel):
    filename: str
    content_type: str
    title: Optional[str] = None
    description: Optional[str] = None

class CommitIn(BaseModel):
    key: str
    content_type: str
    size_bytes: int
    title: Optional[str] = None
    description: Optional[str] = None

def _assert_workspace(conn, wid: str, uid: str) -> None:
    r = conn.execute(text("""
      SELECT 1 FROM workspaces WHERE id=:wid AND owner_user_id=:uid AND deleted_at IS NULL
    """), {"wid": wid, "uid": uid}).fetchone()
    if not r:
        raise HTTPException(404, "Workspace not found")

def _next_no(conn, wid: str) -> int:
    row = conn.execute(text("""
      SELECT COALESCE(MAX(analysis_no),0)+1 AS next_no
        FROM image_analyses WHERE workspace_id=:wid
    """), {"wid": wid}).mappings().first()
    return int(row["next_no"])

@router.post("/{workspace_id}/image-analyses/presign")
def presign(workspace_id: str, body: PresignIn, me=Depends(require_user)):
    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))
        next_no = _next_no(conn, workspace_id)
    key = f"{S3_PREFIX}/{workspace_id}/{next_no}/original/{body.filename}"
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.s3_bucket, "Key": key, "ContentType": body.content_type},
        ExpiresIn=3600, HttpMethod="PUT"
    )
    return {"key": key, "url": url, "content_type": body.content_type, "max_bytes": 25*1024*1024}

@router.post("/{workspace_id}/image-analyses/commit")
def commit(workspace_id: str, body: CommitIn, me=Depends(require_user)):
    # --- Security: content-type & size guard ---
    allowed_types = {"image/jpeg", "image/png"}
    if body.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported content_type {body.content_type}")
    if body.size_bytes is None or int(body.size_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (25MB max)")

    head = s3.head_object(Bucket=settings.s3_bucket, Key=body.key)
    actual_len = int(head.get("ContentLength", 0))
    actual_ct = head.get("ContentType", "")

    if actual_len != int(body.size_bytes):
        raise HTTPException(status_code=400, detail="Size mismatch with S3 object")
    if actual_ct != body.content_type:
        raise HTTPException(status_code=400, detail=f"Content-Type mismatch (S3 has {actual_ct})")

    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))
        no = _next_no(conn, workspace_id)
        aid = str(uuid.uuid4())
        conn.execute(text("""
          INSERT INTO image_analyses (id, workspace_id, analysis_no, title, description,
                                      input_image_s3_key, content_type, size_bytes, status)
          VALUES (:id,:wid,:no,:title,:descr,:key,:ct,:size,'uploaded')
        """), dict(id=aid, wid=workspace_id, no=no, title=body.title, descr=body.description,
                   key=body.key, ct=body.content_type, size=body.size_bytes))
    return {"id": (str(aid).split('?')[0].split('&')[0].lstrip('=')), "analysis_no": no, "status": "uploaded"}

@router.post("/{workspace_id}/image-analyses/{analysis_id}/enqueue")
def enqueue(workspace_id: str, analysis_id: str, me=Depends(require_user)):
    analysis_id = _clean_uuid_for_path(analysis_id)
    aid = analysis_id  # normalized string UUID, keep legacy var alive
    with engine.begin() as conn:
        r = conn.execute(text("""
          SELECT analysis_no, input_image_s3_key
            FROM image_analyses
           WHERE id=:id AND workspace_id=:wid
        """), {"id": aid, "wid": workspace_id}).mappings().first()
        if not r:
            raise HTTPException(404, "Analysis not found")
        payload = {
          "event": "ANALYZE_IMAGE",
          "analysis_id": analysis_id,
          "workspace_id": workspace_id,
          "analysis_no": int(r["analysis_no"]),
          "input_image_s3_uri": f"s3://{settings.s3_bucket}/{r['input_image_s3_key']}"
        }
        resp_b = send_json(Q_BASE, dict(payload, model_variant="baseline"))
        resp_c = send_json(Q_CMT,  dict(payload, model_variant="cmt"))
        conn.execute(text("UPDATE image_analyses SET status='queued' WHERE id=:id"), {"id": aid})
    return {"ok": True, "status": "queued", "baseline": resp_b, "cmt": resp_c}

@router.get("/{workspace_id}/image-analyses/{analysis_id}")
def show(workspace_id: str, analysis_id: str, me=Depends(require_user)):
    analysis_id = _clean_uuid_for_path(analysis_id)
    aid = analysis_id  # normalized string UUID, keep legacy var alive
    with engine.begin() as conn:
        a = conn.execute(text("""
          SELECT id, analysis_no, input_image_s3_key, status, error_msg, created_at
            FROM image_analyses
           WHERE id=:id AND workspace_id=:wid
        """), {"id": aid, "wid": workspace_id}).mappings().first()
        if not a:
            raise HTTPException(404, "Analysis not found")

        results = conn.execute(text("""
          SELECT model_variant::text, type, type_conf, make, make_conf, model, model_conf,
       parts, colors, plate_text, plate_conf,
       annotated_image_s3_key, vehicle_image_s3_key, plate_image_s3_key,
       latency_ms, gflops, memory_usage, status, error_msg
  FROM image_analysis_results
 WHERE analysis_id=:id
        """), {"id": aid}).mappings().all()

    orig_url = s3.generate_presigned_url("get_object",
      Params={"Bucket": settings.s3_bucket, "Key": a["input_image_s3_key"]},
      ExpiresIn=TTL)

    out = {
      "id": a["id"],
      "analysis_no": int(a["analysis_no"]),
      "input_image": {"s3_key": a["input_image_s3_key"], "url": orig_url},
      "status": a["status"],
      "error_msg": a["error_msg"],
      "created_at": a["created_at"].isoformat() if a["created_at"] else None,
      "results": {}
    }

    for r in results:
        ann_url = veh_url = plate_url = None
        if r["annotated_image_s3_key"]:
            ann_url = s3.generate_presigned_url("get_object",
                Params={"Bucket": settings.s3_bucket, "Key": r["annotated_image_s3_key"]},
                ExpiresIn=TTL)
        if r.get("vehicle_image_s3_key"):
            veh_url = s3.generate_presigned_url("get_object",
                Params={"Bucket": settings.s3_bucket, "Key": r["vehicle_image_s3_key"]},
                ExpiresIn=TTL)
        if r.get("plate_image_s3_key"):
            plate_url = s3.generate_presigned_url("get_object",
                Params={"Bucket": settings.s3_bucket, "Key": r["plate_image_s3_key"]},
                ExpiresIn=TTL)
        
        out["results"][r["model_variant"]] = {
            "type": r["type"], "type_conf": r["type_conf"],
            "make": r["make"], "make_conf": r["make_conf"],
            "model": r["model"], "model_conf": r["model_conf"],
            "colors": r["colors"], "parts": r["parts"],
            "plate_text": r["plate_text"], "plate_conf": r["plate_conf"],
            "assets": {
                "annotated_image_s3_key": r["annotated_image_s3_key"], "annotated_url": ann_url,
                "vehicle_image_s3_key": r.get("vehicle_image_s3_key"), "vehicle_url": veh_url,
                "plate_image_s3_key": r.get("plate_image_s3_key"), "plate_url": plate_url
            },
            "latency_ms": r["latency_ms"], "gflops": r["gflops"],
            "status": r["status"], "error_msg": r["error_msg"]
        }
    return out

# List image analyses (plural) — simple paginated list
@router.get("/{workspace_id}/image-analyses")
def list_image_analyses(workspace_id: str, limit: int = 20, offset: int = 0, me=Depends(require_user)):
    with engine.begin() as conn:
        r = conn.execute(text("""
          SELECT 1 FROM workspaces
           WHERE id=:wid AND owner_user_id=:uid AND deleted_at IS NULL
        """), {"wid": workspace_id, "uid": str(me.id)}).fetchone()
        if not r:
            raise HTTPException(404, "Workspace not found")

        rows = conn.execute(text("""
          SELECT id, analysis_no, input_image_s3_key, status, created_at
            FROM image_analyses
           WHERE workspace_id = :wid
           ORDER BY created_at DESC
           LIMIT :limit OFFSET :offset
        """), {"wid": workspace_id, "limit": limit, "offset": offset}).mappings().all()

    return {
        "items": [dict(row) for row in rows],
        "limit": limit,
        "offset": offset
    }

# ─────────────────────────────────────────────────────────────────────────────
# Singular path aliases: /image-analysis/…  (leave plural routes intact)
alias = APIRouter(prefix="/workspaces", tags=["image-analysis"])

@alias.post("/{workspace_id}/image-analysis/presign")
def presign_singular(workspace_id: str, body: PresignIn, me=Depends(require_user)):
    return presign(workspace_id, body, me)

@alias.post("/{workspace_id}/image-analysis/commit")
def commit_singular(workspace_id: str, body: CommitIn, me=Depends(require_user)):
    return commit(workspace_id, body, me)

@alias.post("/{workspace_id}/image-analysis/{analysis_id}/enqueue")
def enqueue_singular(workspace_id: str, analysis_id: str, me=Depends(require_user)):
    return enqueue(workspace_id, analysis_id, me)

@alias.get("/{workspace_id}/image-analysis/{analysis_id}")
def show_singular(workspace_id: str, analysis_id: str, me=Depends(require_user)):
    return show(workspace_id, analysis_id, me)

# Singular LIST → reuse the plural list handler
@alias.get("/{workspace_id}/image-analysis")
def list_singular(workspace_id: str, limit: int = 20, offset: int = 0, me=Depends(require_user)):
    return list_image_analyses(workspace_id, limit, offset, me)
# ─────────────────────────────────────────────────────────────────────────────
