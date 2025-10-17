import os, uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import boto3

from app.auth.deps import require_user
from app.config import settings
from app.services.sqs import send_json

router = APIRouter(prefix="/workspaces", tags=["image-analyses"])

# DB + S3 clients
DB_URL = os.environ["DB_URL"].replace("postgresql+psycopg2","postgresql")
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
    # Validate the object actually exists in S3
    try:
        s3.head_object(Bucket=settings.s3_bucket, Key=body.key)
    except Exception:
        raise HTTPException(400, "S3 object not found; ensure you PUT with the same Content-Type")
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
    return {"id": aid, "analysis_no": no, "status": "uploaded"}

@router.post("/{workspace_id}/image-analyses/{analysis_id}/enqueue")
def enqueue(workspace_id: str, analysis_id: str, me=Depends(require_user)):
    with engine.begin() as conn:
        r = conn.execute(text("""
          SELECT analysis_no, input_image_s3_key
            FROM image_analyses
           WHERE id=:id AND workspace_id=:wid
        """), {"id": analysis_id, "wid": workspace_id}).mappings().first()
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
        conn.execute(text("UPDATE image_analyses SET status='queued' WHERE id=:id"), {"id": analysis_id})
    return {"ok": True, "status": "queued", "baseline": resp_b, "cmt": resp_c}

@router.get("/{workspace_id}/image-analyses/{analysis_id}")
def show(workspace_id: str, analysis_id: str, me=Depends(require_user)):
    with engine.begin() as conn:
        a = conn.execute(text("""
          SELECT id, analysis_no, input_image_s3_key, status, error_msg, created_at
            FROM image_analyses
           WHERE id=:id AND workspace_id=:wid
        """), {"id": analysis_id, "wid": workspace_id}).mappings().first()
        if not a:
            raise HTTPException(404, "Analysis not found")

        results = conn.execute(text("""
          SELECT model_variant::text, type, type_conf, make, make_conf, model, model_conf,
                 parts, colors, plate_text, annotated_image_s3_key, latency_ms, gflops, status, error_msg
            FROM image_analysis_results
           WHERE analysis_id=:id
        """), {"id": analysis_id}).mappings().all()

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
        ann_url = None
        if r["annotated_image_s3_key"]:
            ann_url = s3.generate_presigned_url("get_object",
              Params={"Bucket": settings.s3_bucket, "Key": r["annotated_image_s3_key"]},
              ExpiresIn=TTL)
        out["results"][r["model_variant"]] = {
          "type": r["type"], "type_conf": r["type_conf"],
          "make": r["make"], "make_conf": r["make_conf"],
          "model": r["model"], "model_conf": r["model_conf"],
          "parts": r["parts"], "colors": r["colors"], "plate_text": r["plate_text"],
          "annotated_image": {"s3_key": r["annotated_image_s3_key"], "url": ann_url} if ann_url else None,
          "latency_ms": r["latency_ms"], "gflops": r["gflops"],
          "status": r["status"], "error_msg": r["error_msg"]
        }
    return out
