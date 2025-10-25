from fastapi import APIRouter
import os, subprocess, boto3
from sqlalchemy import create_engine, text
router = APIRouter()
def _git_sha():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], cwd="/home/ubuntu/cvitx").decode().strip()
    except Exception:
        return "unknown"
def _db_ping():
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DB_URL/DATABASE_URL not set")
    sql_url = db_url.replace("postgresql+psycopg2","postgresql")
    eng = create_engine(sql_url, pool_pre_ping=True)
    with eng.begin() as conn:
        conn.execute(text("SELECT 1"))
def _s3_ping():
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET not set")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-2"
    s3 = boto3.client("s3", region_name=region)
    s3.head_bucket(Bucket=bucket)
@router.get("/healthz", include_in_schema=False)
def healthz():
    _db_ping(); _s3_ping()
    return {"ok": True, "version": _git_sha()}
