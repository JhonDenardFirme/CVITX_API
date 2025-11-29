# file: video_analysis/worker_utils/common.py
"""
CVITX · Video Analysis Monorepo — Shared Utilities (yolo_worker & main_worker)

This module centralizes:
• Logging (plain or JSON lines)
• Message validation (PROCESS_VIDEO, PROCESS_VIDEO_DB, SNAPSHOT_READY) via Pydantic v2
• AWS clients (S3, SQS) with region from CONFIG
• DB helpers (psycopg2): get_video_by_id, create_image_analysis, upsert_results,
  create_video_analysis, upsert_video_results, set_video_analysis_status,
  start_video_run, get_video_run, set_video_expected, upsert_video_detection_and_progress
• Key builder for deterministic snapshot S3 keys (bucket-relative)
• Imaging helpers: crop_with_margin, letterbox_to_square, encode_jpeg
• Time helpers: ms_from_frame, detected_at

Design rules:
1) ENV is read ONLY in worker_config.py — import CONFIG here.
2) All defaults and invariants come from the SSOT (worker_config.CONFIG).
3) Functions are deterministic and side-effect free (except I/O ops by design).
"""

from __future__ import annotations

import os
import json
import logging
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict

import boto3
import cv2
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, Json as PgJson
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import create_engine, text

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
    handler = logging.StreamHandler()
    if CONFIG.get("JSON_LOGS"):
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
            )
        )
    logger.addHandler(handler)
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
    """
    Canonical schema for PROCESS_VIDEO messages.

    Matches what /workspaces/{wid}/videos/{vid}/enqueue sends:
      {
        "event": "PROCESS_VIDEO",
        "video_id": "<uuid>",
        "workspace_id": "<uuid>",
        "workspace_code": "CTX####" | null,
        "camera_code": "CAM-TEST-001",
        "s3_key_raw": "demo_user/<wid>/<vid>/raw/file.mp4",
        "frame_stride": 3,
        "recordedAt": "...iso..." | null,
        "variant": "baseline" | "cmt" (default "cmt"),
        "run_id": "<uuid>" | null
      }
    """

    event: Literal["PROCESS_VIDEO"]
    video_id: UUIDStr
    workspace_id: UUIDStr
    # Optional here because older videos may have NULL workspace_code in the DB.
    # Worker MUST handle the "no workspace_code" case by falling back to a DB lookup
    # or a safe placeholder when building display IDs / filenames.
    workspace_code: Optional[str] = None
    camera_code: str = Field(min_length=1)
    s3_key_raw: str
    frame_stride: int = Field(ge=1)
    recordedAt: Optional[str] = None
    # Variant is optional on the wire; default stays "cmt".
    variant: str = "cmt"
    # Run ID is assigned by the worker when it calls start_video_run if missing.
    run_id: Optional[UUIDStr] = None

    @field_validator("s3_key_raw")
    @classmethod
    def _raw_key_ok(cls, v: str) -> str:
        return _assert_regex(RAW_KEY_REGEX, v, "s3_key_raw")


class ProcessVideoDB(BaseModel):
    """
    Minimal PROCESS_VIDEO_DB payload used when the worker rehydrates
    a run purely from DB context (no raw S3 key).
    """

    event: Literal["PROCESS_VIDEO_DB"]
    video_id: UUIDStr
    workspace_id: UUIDStr
    variant: str = "cmt"
    run_id: Optional[UUIDStr] = None


class SnapshotReady(BaseModel):
    """
    Canonical schema for SNAPSHOT_READY messages (YOLO → main worker).

    This is the single source of truth for snapshot notifications:
      {
        "event": "SNAPSHOT_READY",
        "video_id": "<uuid>",
        "workspace_id": "<uuid>",
        "workspace_code": "CTX####",
        "camera_code": "CAM-TEST-001",
        "track_id": 12,
        "snapshot_s3_key": "s3://<bucket>/demo_user/<wid>/<vid>/snapshots/CTX####_CAM-TEST-001_000012_003000.jpg",
        "recordedAt": "...iso..." | null,
        "detectedIn": 3000,
        "detectedAt": "...iso..." | null,
        "yolo_type": "Car",
        "variant": "cmt",
        "run_id": "<uuid>"
      }
    """

    event: Literal["SNAPSHOT_READY"]
    video_id: UUIDStr
    workspace_id: UUIDStr
    workspace_code: str = Field(min_length=1)
    camera_code: str = Field(min_length=1)
    track_id: int = Field(ge=1)
    # Full s3:// URI (locked bucket + canonical prefix), e.g.:
    # s3://<bucket>/<user_prefix>/<workspace_id>/<video_id>/snapshots/CTX####_CAM-01_000123_003000.jpg
    snapshot_s3_key: str
    recordedAt: Optional[str] = None
    detectedIn: int = Field(ge=0)  # milliseconds since video start
    detectedAt: Optional[str] = None
    yolo_type: str
    variant: str = "cmt"
    run_id: UUIDStr

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
    except ValidationError as exc:
        raise ValueError(f"PROCESS_VIDEO validation error: {exc}") from exc


def validate_process_video_db(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a minimal PROCESS_VIDEO_DB payload."""
    try:
        return ProcessVideoDB.model_validate(payload).model_dump(mode="python")
    except ValidationError as exc:
        raise ValueError(f"PROCESS_VIDEO_DB validation error: {exc}") from exc


def validate_snapshot_ready(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a SNAPSHOT_READY payload."""
    try:
        return SnapshotReady.model_validate(payload).model_dump(mode="python")
    except ValidationError as exc:
        raise ValueError(f"SNAPSHOT_READY validation error: {exc}") from exc


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


# --------------------------------------------------------------------
# Video DB engine — shared by YOLO + main workers
# --------------------------------------------------------------------
_VIDEO_DB_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
if not _VIDEO_DB_URL:
    raise RuntimeError("Missing DATABASE_URL/DB_URL for video workers")

# Normalize for SQLAlchemy if needed
_VIDEO_DB_URL = _VIDEO_DB_URL.replace("postgresql+psycopg2", "postgresql")
_video_engine = create_engine(_VIDEO_DB_URL, pool_pre_ping=True)


def _video_conn():
    """Small helper to get a connection context for video_* tables."""
    return _video_engine.begin()


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
    INSERT INTO image_analyses (
      id,
      workspace_id,
      snapshot_s3_key,
      source_kind,
      status,
      created_at,
      updated_at
    )
    VALUES (%s, %s, %s, 'snapshot', 'queued', NOW(), NOW())
    ON CONFLICT (snapshot_s3_key)
    DO UPDATE SET
      updated_at = EXCLUDED.updated_at
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
    type_obj = result.get("type") or {}
    make_obj = result.get("make") or {}
    model_obj = result.get("model") or {}
    plate_obj = result.get("plate") or {}
    colors = result.get("colors") or []
    assets = result.get("assets") or {}
    latency_ms = int(result.get("latency_ms") or 0)
    memory_gb = result.get("memory_gb")
    status = result.get("status") or "done"

    sql = """
    INSERT INTO image_analysis_results (
      analysis_id,
      variant,
      type_label,  type_conf,
      make_label,  make_conf,
      model_label, model_conf,
      plate_text,  plate_conf,
      colors,
      assets,
      latency_ms,
      memory_usage,
      status,
      created_at,
      updated_at
    )
    VALUES (
      %(analysis_id)s,
      %(variant)s,
      %(type_label)s,  %(type_conf)s,
      %(make_label)s,  %(make_conf)s,
      %(model_label)s, %(model_conf)s,
      %(plate_text)s,  %(plate_conf)s,
      %(colors)s,
      %(assets)s,
      %(latency_ms)s,
      %(memory_usage)s,
      %(status)s,
      NOW(),
      NOW()
    )
    ON CONFLICT (analysis_id, variant)
    DO UPDATE SET
      type_label   = EXCLUDED.type_label,
      type_conf    = EXCLUDED.type_conf,
      make_label   = EXCLUDED.make_label,
      make_conf    = EXCLUDED.make_conf,
      model_label  = EXCLUDED.model_label,
      model_conf   = EXCLUDED.model_conf,
      plate_text   = EXCLUDED.plate_text,
      plate_conf   = EXCLUDED.plate_conf,
      colors       = EXCLUDED.colors,
      assets       = EXCLUDED.assets,
      latency_ms   = EXCLUDED.latency_ms,
      memory_usage = EXCLUDED.memory_usage,
      status       = EXCLUDED.status,
      updated_at   = NOW();
    """

    params = {
        "analysis_id": str(analysis_id),
        "variant": str(variant),
        "type_label": type_obj.get("label"),
        "type_conf": float(type_obj.get("conf")) if type_obj.get("conf") is not None else None,
        "make_label": make_obj.get("label"),
        "make_conf": float(make_obj.get("conf")) if make_obj.get("conf") is not None else None,
        "model_label": model_obj.get("label"),
        "model_conf": float(model_obj.get("conf")) if model_obj.get("conf") is not None else None,
        "plate_text": plate_obj.get("text"),
        "plate_conf": float(plate_obj.get("conf")) if plate_obj.get("conf") is not None else None,
        "colors": PgJson(colors),
        "assets": PgJson(assets),
        "latency_ms": int(latency_ms),
        "memory_usage": float(memory_gb) if memory_gb is not None else None,
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
      id,
      workspace_id,
      video_id,
      snapshot_s3_key,
      source_kind,
      status,
      created_at,
      updated_at
    )
    VALUES (
      %s,
      %s,
      %s,
      %s,
      'snapshot',
      'processing',
      NOW(),
      NOW()
    )
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

    Expected 'result' shape (keys are optional; absent ones become NULL/{} or []):
      {
        "type":  {"label": str, "conf": float},
        "make":  {"label": str, "conf": float},
        "model": {"label": str, "conf": float},
        "plate": {"text": str, "conf": float},
        "colors": [ {finish|None, base, lightness|None, conf}, ... ],
        "parts":  [ {name, conf}, ... ],
        "assets": { "annotated_key": str|None, "vehicle_key": str|None, "plate_key": str|None },
        "latency_ms": int,
        "memory_gb": float|None,
        "status": "done"|"error" (default "done"),
        "error_msg": str|None
      }
    """
    type_obj = result.get("type") or {}
    make_obj = result.get("make") or {}
    model_obj = result.get("model") or {}
    plate_obj = result.get("plate") or {}
    colors = result.get("colors") or []
    parts = result.get("parts") or []
    assets = result.get("assets") or {}
    latency_ms = int(result.get("latency_ms") or 0)
    memory_gb = result.get("memory_gb")
    status = result.get("status") or "done"
    error_msg = result.get("error_msg")

    sql = """
    INSERT INTO video_analysis_results (
      analysis_id,
      variant,
      type_label,  type_conf,
      make_label,  make_conf,
      model_label, model_conf,
      plate_text,  plate_conf,
      colors,
      parts,
      assets,
      latency_ms,
      memory_usage,
      status,
      error_msg,
      created_at,
      updated_at
    )
    VALUES (
      %(analysis_id)s,
      %(variant)s,
      %(type_label)s,  %(type_conf)s,
      %(make_label)s,  %(make_conf)s,
      %(model_label)s, %(model_conf)s,
      %(plate_text)s,  %(plate_conf)s,
      %(colors)s,
      %(parts)s,
      %(assets)s,
      %(latency_ms)s,
      %(memory_usage)s,
      %(status)s,
      %(error_msg)s,
      NOW(),
      NOW()
    )
    ON CONFLICT (analysis_id, variant)
    DO UPDATE SET
      type_label   = EXCLUDED.type_label,
      type_conf    = EXCLUDED.type_conf,
      make_label   = EXCLUDED.make_label,
      make_conf    = EXCLUDED.make_conf,
      model_label  = EXCLUDED.model_label,
      model_conf   = EXCLUDED.model_conf,
      plate_text   = EXCLUDED.plate_text,
      plate_conf   = EXCLUDED.plate_conf,
      colors       = EXCLUDED.colors,
      parts        = EXCLUDED.parts,
      assets       = EXCLUDED.assets,
      latency_ms   = EXCLUDED.latency_ms,
      memory_usage = EXCLUDED.memory_usage,
      status       = EXCLUDED.status,
      error_msg    = EXCLUDED.error_msg,
      updated_at   = NOW();
    """

    params = {
        "analysis_id": str(analysis_id),
        "variant": str(variant),
        "type_label": type_obj.get("label"),
        "type_conf": float(type_obj.get("conf")) if type_obj.get("conf") is not None else None,
        "make_label": make_obj.get("label"),
        "make_conf": float(make_obj.get("conf")) if make_obj.get("conf") is not None else None,
        "model_label": model_obj.get("label"),
        "model_conf": float(model_obj.get("conf")) if model_obj.get("conf") is not None else None,
        "plate_text": plate_obj.get("text"),
        "plate_conf": float(plate_obj.get("conf")) if plate_obj.get("conf") is not None else None,
        "colors": PgJson(colors),
        "parts": PgJson(parts),
        "assets": PgJson(assets),
        "latency_ms": int(latency_ms),
        "memory_usage": float(memory_gb) if memory_gb is not None else None,
        "status": str(status),
        "error_msg": str(error_msg) if error_msg is not None else None,
    }

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        conn.commit()


def set_video_analysis_status(analysis_id: str, status: str, error_msg: Optional[str]) -> None:
    """
    Set run-level status for a video analysis.

    Used for fatal errors or explicit cancellation.
    """
    with _video_conn() as conn:
        conn.execute(
            text(
                """
                UPDATE video_analyses
                   SET status          = :status,
                       error_msg       = :error_msg,
                       run_finished_at = CASE
                                           WHEN :status IN ('error', 'done')
                                           THEN COALESCE(run_finished_at, now())
                                           ELSE run_finished_at
                                         END,
                       updated_at      = now()
                 WHERE id = :aid
                """
            ),
            {"aid": analysis_id, "status": status, "error_msg": error_msg},
        )


def start_video_run(workspace_id: str, video_id: str, variant: str, run_id: str) -> str:
    """
    Create a new run container in video_analyses and return its analysis_id.

    • One row per (workspace, video, variant, run_id).
    • Older runs are kept for history; latest run is selected by updated_at.
    """
    analysis_id = str(uuid.uuid4())
    now = datetime.utcnow()

    with _video_conn() as conn:
        conn.execute(
            text(
                """
                INSERT INTO video_analyses (
                    id,
                    workspace_id,
                    video_id,
                    variant,
                    run_id,
                    status,
                    expected_snapshots,
                    processed_snapshots,
                    processed_ok,
                    processed_err,
                    run_started_at,
                    run_finished_at,
                    last_snapshot_at,
                    error_msg,
                    created_at,
                    updated_at
                )
                VALUES (
                    :id,
                    :wid,
                    :vid,
                    :variant,
                    :run_id,
                    'running',
                    0,
                    0,
                    0,
                    0,
                    :now,
                    NULL,
                    NULL,
                    NULL,
                    :now,
                    :now
                )
                """
            ),
            {
                "id": analysis_id,
                "wid": workspace_id,
                "vid": video_id,
                "variant": variant,
                "run_id": run_id,
                "now": now,
            },
        )

    return analysis_id


def get_video_run(workspace_id: str, video_id: str, variant: str) -> Optional[Dict[str, Any]]:
    """
    Return the latest run for (workspace, video, variant) or None.

    Used by main_worker to decide if it should attach to an existing run
    or bootstrap a new run via start_video_run().
    """
    with _video_conn() as conn:
        row = conn.execute(
            text(
                """
                SELECT
                  id,
                  workspace_id,
                  video_id,
                  variant,
                  run_id,
                  status,
                  expected_snapshots,
                  processed_snapshots,
                  processed_ok,
                  processed_err,
                  run_started_at,
                  run_finished_at,
                  last_snapshot_at,
                  error_msg
                FROM video_analyses
                WHERE workspace_id = :wid
                  AND video_id     = :vid
                  AND variant      = :variant
                ORDER BY updated_at DESC,
                         run_started_at DESC NULLS LAST,
                         created_at DESC
                LIMIT 1
                """
            ),
            {"wid": workspace_id, "vid": video_id, "variant": variant},
        ).mappings().first()

    if not row:
        return None

    return {
        "analysis_id": str(row["id"]),
        "workspace_id": str(row["workspace_id"]),
        "video_id": str(row["video_id"]),
        "variant": row["variant"],
        "run_id": str(row["run_id"]) if row["run_id"] else None,
        "status": row["status"],
        "expected_snapshots": row["expected_snapshots"],
        "processed_snapshots": row["processed_snapshots"],
        "processed_ok": row["processed_ok"],
        "processed_err": row["processed_err"],
        "run_started_at": row["run_started_at"],
        "run_finished_at": row["run_finished_at"],
        "last_snapshot_at": row["last_snapshot_at"],
        "error_msg": row["error_msg"],
    }


def set_video_expected(analysis_id: str, expected: int) -> None:
    """
    Set expected_snapshots for a run container.

    Called once by the YOLO worker after it has enumerated how many snapshots it
    will emit for this (video, variant, run_id).
    """
    with _video_conn() as conn:
        conn.execute(
            text(
                """
                UPDATE video_analyses
                   SET expected_snapshots = :expected,
                       updated_at         = now()
                 WHERE id = :aid
                """
            ),
            {"aid": analysis_id, "expected": int(expected)},
        )


def upsert_video_detection_and_progress(
    *,
    analysis_id: str,
    run_id: str,
    track_id: int,
    snapshot_s3_key: str,
    yolo_type: Optional[str],
    detected_in_ms: int,
    detected_at: Optional[datetime],
    result: Dict[str, Any],
) -> None:
    """
    Insert or update a detection row + bump run counters in video_analyses.

    Expected result shape (keys optional, absent → NULL/[]):
      {
        "type":  {"label": str, "conf": float},
        "make":  {"label": str, "conf": float},
        "model": {"label": str, "conf": float},
        "plate": {"text": str, "conf": float},
        "colors": [ {base, finish|None, lightness|None, conf}, ... ],
        "parts":  [ {name, conf}, ... ],
        "assets": {
          "annotated_image_s3_key": str|None,
          "vehicle_image_s3_key":   str|None,
          "plate_image_s3_key":     str|None
        },
        "latency_ms": int,
        "memory_gb": float|None,
        "status": "done"|"error",
        "error_msg": str|None
      }
    """
    type_obj = result.get("type") or {}
    make_obj = result.get("make") or {}
    model_obj = result.get("model") or {}
    plate_obj = result.get("plate") or {}
    colors = result.get("colors") or []
    parts = result.get("parts") or []
    assets = result.get("assets") or {}
    latency_ms = int(result.get("latency_ms") or 0)
    memory_gb = result.get("memory_gb")
    status = result.get("status") or "done"
    error_msg = result.get("error_msg")

    # Treat anything not explicitly error as OK for counters.
    is_ok = (status != "error") and (error_msg is None)

    sql_det = """
    INSERT INTO video_detections (
      analysis_id,
      run_id,
      track_id,
      snapshot_s3_key,
      yolo_type,
      detected_in_ms,
      detected_at,
      type_label,  type_conf,
      make_label,  make_conf,
      model_label, model_conf,
      plate_text,  plate_conf,
      colors,
      parts,
      assets,
      latency_ms,
      memory_usage,
      status,
      error_msg,
      created_at,
      updated_at
    )
    VALUES (
      :analysis_id,
      :run_id,
      :track_id,
      :snapshot_s3_key,
      :yolo_type,
      :detected_in_ms,
      :detected_at,
      :type_label,  :type_conf,
      :make_label,  :make_conf,
      :model_label, :model_conf,
      :plate_text,  :plate_conf,
      :colors,
      :parts,
      :assets,
      :latency_ms,
      :memory_usage,
      :status,
      :error_msg,
      now(),
      now()
    )
    ON CONFLICT (analysis_id, run_id, track_id)
    DO UPDATE SET
      snapshot_s3_key = EXCLUDED.snapshot_s3_key,
      yolo_type       = EXCLUDED.yolo_type,
      detected_in_ms  = EXCLUDED.detected_in_ms,
      detected_at     = EXCLUDED.detected_at,
      type_label      = EXCLUDED.type_label,
      type_conf       = EXCLUDED.type_conf,
      make_label      = EXCLUDED.make_label,
      make_conf       = EXCLUDED.make_conf,
      model_label     = EXCLUDED.model_label,
      model_conf      = EXCLUDED.model_conf,
      plate_text      = EXCLUDED.plate_text,
      plate_conf      = EXCLUDED.plate_conf,
      colors          = EXCLUDED.colors,
      parts           = EXCLUDED.parts,
      assets          = EXCLUDED.assets,
      latency_ms      = EXCLUDED.latency_ms,
      memory_usage    = EXCLUDED.memory_usage,
      status          = EXCLUDED.status,
      error_msg       = EXCLUDED.error_msg,
      updated_at      = now();
    """

    with _video_conn() as conn:
        # Upsert detection row
        conn.execute(
            text(sql_det),
            {
                "analysis_id": analysis_id,
                "run_id": run_id,
                "track_id": int(track_id),
                "snapshot_s3_key": snapshot_s3_key,
                "yolo_type": yolo_type,
                "detected_in_ms": int(detected_in_ms),
                "detected_at": detected_at,
                "type_label": type_obj.get("label"),
                "type_conf": float(type_obj.get("conf")) if type_obj.get("conf") is not None else None,
                "make_label": make_obj.get("label"),
                "make_conf": float(make_obj.get("conf")) if make_obj.get("conf") is not None else None,
                "model_label": model_obj.get("label"),
                "model_conf": float(model_obj.get("conf")) if model_obj.get("conf") is not None else None,
                "plate_text": plate_obj.get("text"),
                "plate_conf": float(plate_obj.get("conf")) if plate_obj.get("conf") is not None else None,
                "colors": PgJson(colors),
                "parts": PgJson(parts),
                "assets": PgJson(assets),
                "latency_ms": latency_ms,
                "memory_usage": float(memory_gb) if memory_gb is not None else None,
                "status": status,
                "error_msg": error_msg,
            },
        )

        # Bump per-run counters
        conn.execute(
            text(
                """
                UPDATE video_analyses
                   SET processed_snapshots = processed_snapshots + 1,
                       processed_ok        = processed_ok + CASE WHEN :ok THEN 1 ELSE 0 END,
                       processed_err       = processed_err + CASE WHEN :ok THEN 0 ELSE 1 END,
                       last_snapshot_at    = COALESCE(:detected_at, last_snapshot_at),
                       updated_at          = now()
                 WHERE id = :aid
                """
            ),
            {"aid": analysis_id, "ok": is_ok, "detected_at": detected_at},
        )


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


def letterbox_to_square(
    img: np.ndarray,
    size: int = int(CONFIG["SNAPSHOT_SIZE"]),
) -> np.ndarray:
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
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def encode_jpeg(
    img: np.ndarray,
    quality: int = int(CONFIG["JPG_QUALITY"]),
) -> bytes:
    """Encode an image to JPEG bytes with the configured quality."""
    ok, buf = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
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
    "start_video_run",
    "get_video_run",
    "set_video_expected",
    "upsert_video_detection_and_progress",
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

