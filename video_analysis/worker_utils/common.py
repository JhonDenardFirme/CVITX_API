# file: video_analysis/worker_utils/common.py
"""
CVITX · Video Analysis Monorepo — Shared Utilities (yolo_worker & main_worker)

This module centralizes:
• Logging (plain or JSON lines)
• Message validation (PROCESS_VIDEO, PROCESS_VIDEO_DB, SNAPSHOT_READY) via Pydantic v2
• AWS clients (S3, SQS) with region from CONFIG
• DB helpers (psycopg2): get_video_by_id, create_image_analysis, upsert_results
• Key builder for deterministic snapshot S3 keys (bucket-relative)
• Imaging helpers: crop_with_margin, letterbox_to_square, encode_jpeg
• Time helpers: ms_from_frame, detected_at

Design rules:
1) ENV is read ONLY in worker_config.py — import CONFIG here.
2) All defaults and invariants come from the SSOT (worker_config.CONFIG).
3) Functions are deterministic and side-effect free (except I/O ops by design).
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, TypedDict, Union

import boto3
import cv2
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, Json as PgJson
from pydantic import BaseModel, Field, ValidationError, field_validator

from video_analysis.worker_config import (
    CONFIG,
    BUNDLES,
    YOLO_VEHICLE_TYPES,
    RAW_KEY_REGEX,
    SNAPSHOT_URI_REGEX,
    s3_uri,
)

# =============================== Logging =================================== #

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


@lru_cache(maxsize=8)
def get_logger(name: str = "cvitx") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    if CONFIG.get("JSON_LOGS"):
        h.setFormatter(_JsonFormatter())
    else:
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


log = get_logger("cvitx.common")


# ========================= Message Schema Models =========================== #
# Pydantic v2 models for strict validation of message contracts.
# NOTE:
#   • SnapshotReady is the canonical SNAPSHOT_READY schema (single source of truth).
#   • All workers and helper modules should rely on this model via
#     validate_snapshot_ready(...) or by importing SnapshotReady directly.

UUIDStr = str  # keep as string; upstream contracts are string UUIDs


def _assert_regex(pattern: re.Pattern, value: str, field: str) -> str:
    if not pattern.match(value):
        raise ValueError(f"{field} failed regex check")
    return value


class ProcessVideo(BaseModel):
    event: Literal["PROCESS_VIDEO"]
    video_id: UUIDStr
    workspace_id: UUIDStr
    workspace_code: str = Field(min_length=1)
    camera_code: str = Field(min_length=1)
    s3_key_raw: str
    frame_stride: int = Field(ge=1)
    recordedAt: Optional[str] = None

    @field_validator("s3_key_raw")
    @classmethod
    def _raw_key_ok(cls, v: str) -> str:
        return _assert_regex(RAW_KEY_REGEX, v, "s3_key_raw")


class ProcessVideoDB(BaseModel):
    event: Literal["PROCESS_VIDEO_DB"]
    video_id: UUIDStr
    workspace_id: UUIDStr


class SnapshotReady(BaseModel):
    event: Literal["SNAPSHOT_READY"]
    video_id: UUIDStr
    workspace_id: UUIDStr
    workspace_code: str = Field(min_length=1)
    camera_code: str = Field(min_length=1)
    track_id: int = Field(ge=1)
    snapshot_s3_key: str  # full s3:// URI (locked bucket)
    recordedAt: Optional[str] = None
    detectedIn: int = Field(ge=0)  # milliseconds since video start
    detectedAt: Optional[str] = None
    yolo_type: str

    @field_validator("snapshot_s3_key")
    @classmethod
    def _snap_uri_ok(cls, v: str) -> str:
        return _assert_regex(SNAPSHOT_URI_REGEX, v, "snapshot_s3_key")

    @field_validator("yolo_type")
    @classmethod
    def _yolo_type_ok(cls, v: str) -> str:
        if v not in YOLO_VEHICLE_TYPES:
            raise ValueError(f"yolo_type '{v}' not in configured YOLO_VEHICLE_TYPES")
        return v


def validate_process_video(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a PROCESS_VIDEO payload; returns normalized dict (python types)."""
    try:
        return ProcessVideo.model_validate(payload).model_dump(mode="python")
    except ValidationError as e:
        raise ValueError(f"PROCESS_VIDEO validation error: {e}") from e


def validate_process_video_db(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a minimal PROCESS_VIDEO_DB payload."""
    try:
        return ProcessVideoDB.model_validate(payload).model_dump(mode="python")
    except ValidationError as e:
        raise ValueError(f"PROCESS_VIDEO_DB validation error: {e}") from e


def validate_snapshot_ready(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a SNAPSHOT_READY payload."""
    try:
        return SnapshotReady.model_validate(payload).model_dump(mode="python")
    except ValidationError as e:
        raise ValueError(f"SNAPSHOT_READY validation error: {e}") from e


# ================================ AWS Clients ============================== #

@lru_cache(maxsize=1)
def get_s3():
    """Singleton S3 client (region from CONFIG)."""
    return boto3.client("s3", region_name=str(CONFIG["AWS_REGION"]))


@lru_cache(maxsize=1)
def get_sqs():
    """Singleton SQS client (region from CONFIG)."""
    return boto3.client("sqs", region_name=str(CONFIG["AWS_REGION"]))


# ================================ DB Helpers =============================== #

def _psycopg_dsn(url: str) -> str:
    """Convert SQLAlchemy-style URL to psycopg2 DSN if needed."""
    # e.g., postgresql+psycopg2:// → postgresql://
    return re.sub(r"^postgresql\+psycopg2://", "postgresql://", url)


@lru_cache(maxsize=1)
def _db_connect_params() -> Dict[str, str]:
    return {"dsn": _psycopg_dsn(str(CONFIG["DB_URL"]))}


def _connect():
    return psycopg2.connect(**_db_connect_params())


class VideoRow(TypedDict, total=False):
    id: str
    workspace_id: str
    workspace_code: str
    camera_code: str
    s3_key_raw: str
    frame_stride: int
    recorded_at: Optional[str]


def get_video_by_id(video_id: str) -> VideoRow:
    """
    Fetch required video metadata. Returns dict with keys:
    {s3_key_raw, workspace_code, camera_code, frame_stride, recorded_at, id, workspace_id}
    """
    sql = """
    SELECT
      v.id::text,
      v.workspace_id::text,
      v.workspace_code,
      v.camera_code,
      v.s3_key_raw,
      COALESCE(v.frame_stride, %s) AS frame_stride,
      CASE
        WHEN v.recorded_at IS NULL THEN NULL
        ELSE to_char(v.recorded_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
      END AS recorded_at
    FROM videos v
    WHERE v.id::text = %s
    LIMIT 1
    """
    with _connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (int(CONFIG["FRAME_STRIDE_DEFAULT"]), str(video_id)))
        row = cur.fetchone()
        if not row:
            raise LookupError(f"Video {video_id} not found")
        return VideoRow(**row)  # type: ignore[arg-type]


def create_image_analysis(snapshot_s3_key: str, workspace_id: str) -> str:
    """
    Create (or fetch existing) image_analyses row for this snapshot key.
    Returns the analysis UUID (string).
    """
    analysis_id = str(uuid.uuid4())
    sql = """
    INSERT INTO image_analyses (id, workspace_id, snapshot_s3_key, source_kind, status, created_at, updated_at)
    VALUES (%s, %s, %s, 'snapshot', 'queued', NOW(), NOW())
    ON CONFLICT (snapshot_s3_key)
    DO UPDATE SET updated_at = EXCLUDED.updated_at
    RETURNING id::text;
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (analysis_id, workspace_id, snapshot_s3_key))
        got_id = cur.fetchone()[0]
        conn.commit()
        return got_id


def upsert_results(analysis_id: str, variant: str, result: Dict[str, Any]) -> None:
    """
    Upsert a result row into image_analysis_results (UNIQUE(analysis_id, variant)).
    Expected 'result' shape (keys are optional; absent ones become NULL/{}):
      {
        "type":  {"label": str, "conf": float},
        "make":  {"label": str, "conf": float},
        "model": {"label": str, "conf": float},
        "plate": {"text": str, "conf": float},
        "colors": [ {finish|None, base, lightness|None, conf}, ... ],
        "assets": { "annotated_key": str|None, "vehicle_key": str|None, "plate_key": str|None },
        "latency_ms": int,
        "memory_gb": float|None,
        "status": "done"|"error" (default "done")
      }
    """
    t = (result.get("type") or {})
    m = (result.get("make") or {})
    mm = (result.get("model") or {})
    p = (result.get("plate") or {})
    colors = result.get("colors") or []
    assets = result.get("assets") or {}
    latency_ms = int(result.get("latency_ms") or 0)
    memory_gb = result.get("memory_gb")
    status = result.get("status") or "done"

    sql = """
    INSERT INTO image_analysis_results (
      analysis_id, variant,
      type_label,  type_conf,
      make_label,  make_conf,
      model_label, model_conf,
      plate_text,  plate_conf,
      colors,      assets,
      latency_ms,  memory_gb,
      status,      created_at, updated_at
    )
    VALUES (
      %(analysis_id)s, %(variant)s,
      %(type_label)s,  %(type_conf)s,
      %(make_label)s,  %(make_conf)s,
      %(model_label)s, %(model_conf)s,
      %(plate_text)s,  %(plate_conf)s,
      %(colors)s,      %(assets)s,
      %(latency_ms)s,  %(memory_gb)s,
      %(status)s,      NOW(), NOW()
    )
    ON CONFLICT (analysis_id, variant)
    DO UPDATE SET
      type_label  = EXCLUDED.type_label,
      type_conf   = EXCLUDED.type_conf,
      make_label  = EXCLUDED.make_label,
      make_conf   = EXCLUDED.make_conf,
      model_label = EXCLUDED.model_label,
      model_conf  = EXCLUDED.model_conf,
      plate_text  = EXCLUDED.plate_text,
      plate_conf  = EXCLUDED.plate_conf,
      colors      = EXCLUDED.colors,
      assets      = EXCLUDED.assets,
      latency_ms  = EXCLUDED.latency_ms,
      memory_gb   = EXCLUDED.memory_gb,
      status      = EXCLUDED.status,
      updated_at  = NOW();
    """

    params = {
        "analysis_id": str(analysis_id),
        "variant": str(variant),
        "type_label": t.get("label"),
        "type_conf": float(t.get("conf")) if t.get("conf") is not None else None,
        "make_label": m.get("label"),
        "make_conf": float(m.get("conf")) if m.get("conf") is not None else None,
        "model_label": mm.get("label"),
        "model_conf": float(mm.get("conf")) if mm.get("conf") is not None else None,
        "plate_text": p.get("text"),
        "plate_conf": float(p.get("conf")) if p.get("conf") is not None else None,
        "colors": PgJson(colors),
        "assets": PgJson(assets),
        "latency_ms": int(latency_ms),
        "memory_gb": float(memory_gb) if memory_gb is not None else None,
        "status": str(status),
    }

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        conn.commit()


def create_video_analysis(snapshot_s3_key: str, workspace_id: str, video_id: str) -> str:
    """
    Create (or fetch existing) video_analyses row for this snapshot key.
    Returns the analysis UUID (string).

    This is the video-analysis counterpart to create_image_analysis and is
    intentionally scoped to the video_analyses table so that video detections
    never leak into image_analyses.
    """
    analysis_id = str(uuid.uuid4())
    sql = """
    INSERT INTO video_analyses (
      id, workspace_id, video_id, snapshot_s3_key, source_kind, status, created_at, updated_at
    )
    VALUES (%s, %s, %s, %s, 'snapshot', 'processing', NOW(), NOW())
    ON CONFLICT (snapshot_s3_key)
    DO UPDATE SET
      workspace_id = EXCLUDED.workspace_id,
      video_id     = EXCLUDED.video_id,
      updated_at   = EXCLUDED.updated_at
    RETURNING id::text;
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (analysis_id, workspace_id, video_id, snapshot_s3_key))
        got_id = cur.fetchone()[0]
        conn.commit()
        return got_id


def upsert_video_results(analysis_id: str, variant: str, result: Dict[str, Any]) -> None:
    """
    Upsert a result row into video_analysis_results (UNIQUE(analysis_id, variant)).

    Expected 'result' shape (keys are optional; absent ones become NULL/{}):
      {
        "type":  {"label": str, "conf": float},
        "make":  {"label": str, "conf": float},
        "model": {"label": str, "conf": float},
        "plate": {"text": str, "conf": float},
        "colors": [ {finish|None, base, lightness|None, conf}, ... ],
        "assets": { "annotated_key": str|None, "vehicle_key": str|None, "plate_key": str|None },
        "latency_ms": int,
        "memory_gb": float|None,
        "status": "done"|"error" (default "done"),
        "error_msg": str|None
      }
    """
    t = (result.get("type") or {})
    m = (result.get("make") or {})
    mm = (result.get("model") or {})
    p = (result.get("plate") or {})
    colors = result.get("colors") or []
    assets = result.get("assets") or {}
    latency_ms = int(result.get("latency_ms") or 0)
    memory_gb = result.get("memory_gb")
    status = result.get("status") or "done"
    error_msg = result.get("error_msg")

    sql = """
    INSERT INTO video_analysis_results (
      analysis_id, variant,
      type_label,  type_conf,
      make_label,  make_conf,
      model_label, model_conf,
      plate_text,  plate_conf,
      colors,      assets,
      latency_ms,  memory_gb,
      status,      error_msg,
      created_at,  updated_at
    )
    VALUES (
      %(analysis_id)s, %(variant)s,
      %(type_label)s,  %(type_conf)s,
      %(make_label)s,  %(make_conf)s,
      %(model_label)s, %(model_conf)s,
      %(plate_text)s,  %(plate_conf)s,
      %(colors)s,      %(assets)s,
      %(latency_ms)s,  %(memory_gb)s,
      %(status)s,      %(error_msg)s,
      NOW(),           NOW()
    )
    ON CONFLICT (analysis_id, variant)
    DO UPDATE SET
      type_label  = EXCLUDED.type_label,
      type_conf   = EXCLUDED.type_conf,
      make_label  = EXCLUDED.make_label,
      make_conf   = EXCLUDED.make_conf,
      model_label = EXCLUDED.model_label,
      model_conf  = EXCLUDED.model_conf,
      plate_text  = EXCLUDED.plate_text,
      plate_conf  = EXCLUDED.plate_conf,
      colors      = EXCLUDED.colors,
      assets      = EXCLUDED.assets,
      latency_ms  = EXCLUDED.latency_ms,
      memory_gb   = EXCLUDED.memory_gb,
      status      = EXCLUDED.status,
      error_msg   = EXCLUDED.error_msg,
      updated_at  = NOW();
    """

    params = {
        "analysis_id": str(analysis_id),
        "variant": str(variant),
        "type_label": t.get("label"),
        "type_conf": float(t.get("conf")) if t.get("conf") is not None else None,
        "make_label": m.get("label"),
        "make_conf": float(m.get("conf")) if m.get("conf") is not None else None,
        "model_label": mm.get("label"),
        "model_conf": float(mm.get("conf")) if mm.get("conf") is not None else None,
        "plate_text": p.get("text"),
        "plate_conf": float(p.get("conf")) if p.get("conf") is not None else None,
        "colors": PgJson(colors),
        "assets": PgJson(assets),
        "latency_ms": int(latency_ms),
        "memory_gb": float(memory_gb) if memory_gb is not None else None,
        "status": str(status),
        "error_msg": str(error_msg) if error_msg is not None else None,
    }

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        conn.commit()


def set_video_analysis_status(analysis_id: str, status: str, error_msg: Optional[str] = None) -> None:
    """
    Update video_analyses.status and optional error_msg for a given analysis.
    """
    sql = """
    UPDATE video_analyses
    SET status = %(status)s,
        error_msg = %(error_msg)s,
        updated_at = NOW()
    WHERE id::text = %(analysis_id)s;
    """
    params = {
        "analysis_id": str(analysis_id),
        "status": str(status),
        "error_msg": str(error_msg) if error_msg is not None else None,
    }
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        conn.commit()


# ============================ Key / Path Builders =========================== #

def build_snapshot_key(
    workspace_id: str,
    video_id: str,
    workspace_code: str,
    camera_code: str,
    track_id: int,
    offset_ms: int,
) -> str:
    """
    Deterministic bucket-relative snapshot key:
      demo_user/<wid>/<vid>/snapshots/CTX####_CAM#_{track:06d}_{offsetMs:06d}.jpg
    Use s3_uri(key) to convert to full s3:// URI.
    """
    return (
        f"demo_user/{workspace_id}/{video_id}/snapshots/"
        f"{workspace_code}_{camera_code}_{track_id:06d}_{offset_ms:06d}.jpg"
    )


# ================================ Imaging ================================== #

def crop_with_margin(
    frame: np.ndarray,
    tlbr: Tuple[float, float, float, float],
    margin: float = float(CONFIG["SNAPSHOT_MARGIN"]),
    neighbor_iou: float = float(CONFIG["SNAPSHOT_NEIGHBOR_IOU"]),
) -> np.ndarray:
    """
    Crop a region with adaptive margin. Inputs:
      frame: H×W×3 uint8 (BGR)
      tlbr:  (x1, y1, x2, y2) in pixel space
      margin: % of box size added around all sides
      neighbor_iou: if >0.10, we increase margin slightly to avoid truncation
    Returns:
      ROI image (uint8).
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(float, tlbr)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)

    # Adaptive bump if crowded
    bump = 1.15 if neighbor_iou > 0.10 else 1.0
    mx, my = (margin * w * bump), (margin * h * bump)

    cx1 = max(0, int(math.floor(x1 - mx)))
    cy1 = max(0, int(math.floor(y1 - my)))
    cx2 = min(W, int(math.ceil(x2 + mx)))
    cy2 = min(H, int(math.ceil(y2 + my)))

    if cx2 <= cx1 or cy2 <= cy1:
        # Fallback to entire frame (should not happen with sane boxes)
        return frame.copy()
    return frame[cy1:cy2, cx1:cx2].copy()


def letterbox_to_square(img: np.ndarray, size: int = int(CONFIG["SNAPSHOT_SIZE"])) -> np.ndarray:
    """
    Letterbox an image to a square canvas of `size` with aspect preserved (pad with black).
    Returns the new H×W×3 uint8 image (size×size×3).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero dimension")

    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def encode_jpeg(img: np.ndarray, quality: int = int(CONFIG["JPG_QUALITY"])) -> bytes:
    """Encode an image to JPEG bytes with the configured quality."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


# ================================ Time Utils =============================== #

def ms_from_frame(frame_index: int, fps: float) -> int:
    """Convert frame index to elapsed milliseconds (rounded)."""
    if fps <= 0:
        return 0
    return int(round((frame_index / float(fps)) * 1000.0))


def _parse_iso(iso: Optional[str]) -> Optional[datetime]:
    if not iso:
        return None
    # Accept ...Z or with offset
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(iso)
    except Exception:
        return None


def detected_at(recorded_at_iso: Optional[str], detected_ms: int) -> Optional[str]:
    """Compute detectedAt ISO8601 (UTC) from recordedAt + detectedIn (ms)."""
    start = _parse_iso(recorded_at_iso)
    if not start:
        return None
    when = start + timedelta(milliseconds=int(detected_ms))
    # Normalize to UTC Z
    return when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ================================ Exports ================================== #

__all__ = [
    # logging
    "get_logger",
    "log",
    # validators
    "validate_process_video",
    "validate_process_video_db",
    "validate_snapshot_ready",
    # schema models (canonical)
    "SnapshotReady",
    # aws
    "get_s3",
    "get_sqs",
    # db
    "get_video_by_id",
    "create_image_analysis",
    "upsert_results",
    "create_video_analysis",
    "upsert_video_results",
    "set_video_analysis_status",
    # keys
    "build_snapshot_key",
    # imaging
    "crop_with_margin",
    "letterbox_to_square",
    "encode_jpeg",
    # time
    "ms_from_frame",
    "detected_at",
    # passthroughs useful to workers
    "CONFIG",
    "BUNDLES",
    "YOLO_VEHICLE_TYPES",
    "RAW_KEY_REGEX",
    "SNAPSHOT_URI_REGEX",
    "s3_uri",
]

