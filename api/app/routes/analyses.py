# File: api/app/routes/analyses.py

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from uuid import UUID
import os, re

def _clean_uuid_for_path(raw: str) -> str:
    s = str(raw or "")
    s = s.lstrip("=")
    s = s.split("?", 1)[0]
    s = s.split("&", 1)[0]
    try:
        return str(UUID(s))
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "invalid_analysis_id", "value": raw})

try:
    import psycopg2  # rely on existing dependency
except Exception as e:
    raise RuntimeError("psycopg2 is required for /analyses route") from e

router = APIRouter()

def _dsn_from_env() -> str:
    url = os.environ.get("DATABASE_URL") or os.environ.get("DB_URL")
    if not url:
        raise RuntimeError("DATABASE_URL/DB_URL not set")
    # Convert sqlalchemy-style driver URL to psycopg2-compatible
    return re.sub(r'\+psycopg2', '', url)

@router.get("/analyses/{analysis_id}")
def consolidated_show(
    request: Request,
    analysis_id: str,
    presign: int = Query(1, ge=0, le=1),
    ttl: int = Query(3600, ge=60, le=86400),
):
    analysis_id = _clean_uuid_for_path(analysis_id)
    dsn = _dsn_from_env()
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT workspace_id FROM public.image_analyses WHERE id=%s", (str(analysis_id),))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="analysis not found")
            wid = row[0]
    # 307 so clients keep method + Authorization header; auth enforced by the target route.
    # Forward presign & ttl; point to the canonical singular path.
    url = f"/workspaces/{wid}/image-analysis/{analysis_id}?presign={presign}&ttl={ttl}"
    return RedirectResponse(url=url, status_code=307)
