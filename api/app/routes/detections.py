from app.utils.presign import presign_get
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
import json
from datetime import datetime

from app.db import engine
from app.security import require_api_key

router = APIRouter(
    prefix="/api/detections",
    tags=["detections"],
    dependencies=[Depends(require_api_key)],
)

class Order(str, Enum):
    detected_at_desc = "detected_at_desc"
    detected_at_asc  = "detected_at_asc"

# ---- Pydantic response models (so OpenAPI shows presigned fields)
class DetectionSummary(BaseModel):
    id: UUID
    display_id: str
    snapshot_url: Optional[str] = None
    plate_url: Optional[str] = None
    plate_text: Optional[str] = None
    type: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    type_conf: Optional[float] = None
    make_conf: Optional[float] = None
    model_conf: Optional[float] = None
    colors: Optional[List[str]] = None
    recorded_at: datetime
    detected_in_ms: int
    detected_at: datetime

class PaginationMeta(BaseModel):
    limit: int
    offset: int
    total: int
    next_offset: Optional[int] = None
    order: Order

class DetectionListResponse(BaseModel):
    items: List[DetectionSummary]
    pagination: PaginationMeta

class DetectionDetail(BaseModel):
    # Summary fields
    id: UUID
    display_id: str
    snapshot_url: Optional[str] = None
    plate_url: Optional[str] = None
    plate_text: Optional[str] = None
    yolo_type: Optional[str] = None
    type: Optional[str] = None
    type_conf: Optional[float] = None
    make: Optional[str] = None
    make_conf: Optional[float] = None
    model: Optional[str] = None
    model_conf: Optional[float] = None
    parts: Optional[List[Dict[str, Any]]] = None
    colors: Optional[List[str]] = None
    recorded_at: datetime
    detected_in_ms: int
    detected_at: datetime
    # Extras
    abstain: Optional[bool] = None
    abstain_level: Optional[str] = None
    abstain_reason: Optional[str] = None
    thresholds: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

SELECT_BASE = """
SELECT
  id, display_id, video_id, workspace_id, track_id,
  snapshot_s3_key, plate_image_s3_key,
  recorded_at, detected_in_ms, detected_at,
  yolo_type, type, type_conf, make, make_conf, model, model_conf,
  parts, colors, plate_text,
  abstain, abstain_level, abstain_reason, thresholds, evidence,
  status, created_at
FROM detections
"""

def _row_to_summary(r) -> DetectionSummary:
    d = dict(r._mapping)
    return DetectionSummary(
        id=d["id"],
        display_id=d["display_id"],
        snapshot_url=presign_get(d["snapshot_s3_key"]) if d.get("snapshot_s3_key") else None,
        plate_url=presign_get(d["plate_image_s3_key"]) if d.get("plate_image_s3_key") else None,
        plate_text=d.get("plate_text"),
        type=d.get("type"),
        make=d.get("make"),
        model=d.get("model"),
        type_conf=d.get("type_conf"),
        make_conf=d.get("make_conf"),
        model_conf=d.get("model_conf"),
        colors=d.get("colors"),
        recorded_at=d["recorded_at"],
        detected_in_ms=d["detected_in_ms"],
        detected_at=d["detected_at"],
    )

def _row_to_detail(r) -> DetectionDetail:
    d = dict(r._mapping)
    return DetectionDetail(
        id=d["id"],
        display_id=d["display_id"],
        snapshot_url=presign_get(d["snapshot_s3_key"]) if d.get("snapshot_s3_key") else None,
        plate_url=presign_get(d["plate_image_s3_key"]) if d.get("plate_image_s3_key") else None,
        plate_text=d.get("plate_text"),
        yolo_type=d.get("yolo_type"),
        type=d.get("type"), type_conf=d.get("type_conf"),
        make=d.get("make"), make_conf=d.get("make_conf"),
        model=d.get("model"), model_conf=d.get("model_conf"),
        parts=d.get("parts"),
        colors=d.get("colors"),
        recorded_at=d["recorded_at"],
        detected_in_ms=d["detected_in_ms"],
        detected_at=d["detected_at"],
        abstain=d.get("abstain"),
        abstain_level=d.get("abstain_level"),
        abstain_reason=d.get("abstain_reason"),
        thresholds=d.get("thresholds"),
        evidence=d.get("evidence"),
        status=d.get("status"),
    )

@router.get("", response_model=DetectionListResponse)
def list_detections(
    workspace_id: Optional[UUID] = Query(None),
    video_id: Optional[UUID] = Query(None),
    # Filters
    plate: Optional[str] = Query(None, description="Case-insensitive contains match on plate_text"),
    color: Optional[str] = Query(None, description="Color label, e.g. BLACK"),
    min_detected_at: Optional[str] = Query(None, description="ISO-8601 lower bound"),
    max_detected_at: Optional[str] = Query(None, description="ISO-8601 upper bound"),
    # Paging + order
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    order: Order = Query(Order.detected_at_desc),
):
    order_sql = "ORDER BY detected_at DESC" if order == Order.detected_at_desc else "ORDER BY detected_at ASC"

    where = []
    params: Dict[str, Any] = {}

    if workspace_id:
        where.append("workspace_id = :wsid"); params["wsid"] = str(workspace_id)
    if video_id:
        where.append("video_id = :vid"); params["vid"] = str(video_id)
    # plate contains (case-insensitive, no symbol stripping)
    if plate:
        where.append("upper(plate_text) LIKE :plate_like")
        params["plate_like"] = f"%{plate.strip().upper()}%"
    if color:
        where.append("colors @> :colorjson"); params["colorjson"] = json.dumps([color.strip().upper()])
    if min_detected_at:
        where.append("detected_at >= :min_dt"); params["min_dt"] = min_detected_at
    if max_detected_at:
        where.append("detected_at <= :max_dt"); params["max_dt"] = max_detected_at

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    with engine.begin() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM detections {where_sql}"), params).scalar_one()
        rows = conn.execute(text(f"""
            {SELECT_BASE}
            {where_sql}
            {order_sql}
            LIMIT :limit OFFSET :offset
        """), {**params, "limit": limit, "offset": offset}).fetchall()

    items = [_row_to_summary(r) for r in rows]
    next_offset = (offset + limit) if (offset + limit) < total else None

    return DetectionListResponse(
        items=items,
        pagination=PaginationMeta(
            limit=limit, offset=offset, total=total, next_offset=next_offset, order=order
        ),
    )

@router.get("/{det_id}", response_model=DetectionDetail)
def get_detection(det_id: UUID):
    sql = SELECT_BASE + " WHERE id = :id"
    with engine.begin() as conn:
        row = conn.execute(text(sql), {"id": str(det_id)}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Detection not found")
    return _row_to_detail(row)
