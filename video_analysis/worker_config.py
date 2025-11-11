"""
CVITX · Video Analysis Monorepo
Deterministic, centralized runtime config for BOTH workers:

• yolo_worker  — consumes PROCESS_VIDEO/PROCESS_VIDEO_DB → writes 640×640 JPEG snapshots → emits SNAPSHOT_READY
• main_worker  — consumes SNAPSHOT_READY → runs inference (single or Baseline+CMT) → writes DB rows + artifacts

Design:
- ENV-FIRST with safe, locked dev defaults (SSOT) so the pipeline boots on a fresh EC2 without editing code.
- One import point for both workers (CONFIG + BUNDLES). No other file should read os.environ directly.
- Read-only CONFIG at runtime (mappingproxy) to prevent accidental mutation.

Sample data shapes (for implementers):
- Snapshot image: 640×640×3 uint8 (JPEG ~30–200 KB)
- Model input: [1, 3, 640, 640] float32
- YOLO det per frame: D×6 → [x1,y1,x2,y2,conf,cls]
- Colors (FBL): list[{"finish": null|Matte|Metallic, "base": TitleCase, "lightness": null|Light, "conf": 0..1}]

================================================================================
⚠️ AWS SETUP REQUIRED (once per environment)
   • S3 bucket must exist: s3://cvitx-uploads-dev-jdfirme
   • SQS queues must exist:
       - Video tasks    : https://sqs.ap-southeast-2.amazonaws.com/118730128890/cvitx-video-tasks
       - Snapshot tasks : https://sqs.ap-southeast-2.amazonaws.com/118730128890/cvitx-snapshot-tasks
     with DLQs + RedrivePolicy (maxReceiveCount≈5), VisibilityTimeout≈300s, LongPolling=10s
   • IAM roles:
       - YOLO worker: s3:GetObject on /raw/*, s3:PutObject on /snapshots/*; SQS Receive/Delete(ChangeVisibility) on video-tasks; SQS Send on snapshot-tasks
       - Main worker: s3:GetObject on /snapshots/*; SQS Receive/Delete(ChangeVisibility) on snapshot-tasks
   • Security: do NOT expose presigned URLs or credentials in client apps. DB is private (RDS SG only).
================================================================================
"""

from __future__ import annotations

import os
import re
from types import MappingProxyType
from typing import Dict, Final, Tuple

# ---------------------------- Helpers (internal) ---------------------------- #

def _env(name: str, default: str) -> str:
    """Read environment variable with a deterministic default (no None)."""
    val = os.getenv(name)
    return default if val is None or val == "" else val

def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val not in (None, "") else default
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val not in (None, "") else default
    except ValueError:
        return default

def _redact_dsn(dsn: str) -> str:
    # postgresql+psycopg2://user:password@host:5432/db -> redact password segment
    return re.sub(r"(://[^:]+:)[^@]+@", r"\1****@", dsn)

# --------------------------- Frozen Taxonomies/Regex ------------------------ #
# ✅ Canonical 8-class, no-space taxonomy (exact order is SSOT)
YOLO_VEHICLE_TYPES: Final[Tuple[str, ...]] = (
   "Car", "SUV", "Pickup", "Van", "Utility Vehicle", "Motorcycle",
    "Bicycle", "E-Bike", "Pedicab", "Tricycle", "Jeepney",
    "E-Jeepney", "Bus", "CarouselBus", "LightTruck",
    "ContainerTruck", "SpecialVehicle"
)

# Deterministic mappings if needed by workers
TYPE_TO_ID: Final[Dict[str, int]] = MappingProxyType({name: idx for idx, name in enumerate(YOLO_VEHICLE_TYPES)})

# Common, safe container of allowed video extensions (lowercase; regex is case-insensitive)
VIDEO_EXTS: Final[Tuple[str, ...]] = ("mp4", "mov", "m4v", "avi", "mkv", "webm")

# Raw video KEY (not full URI).
# Contract-aligned: schema expects .mp4 only for raw videos (case-insensitive).
RAW_KEY_REGEX: Final = re.compile(
    r"^demo_user/[0-9a-f-]{36}/[0-9a-f-]{36}/raw/.+\.mp4$",
    flags=re.IGNORECASE,
)

# Snapshot full s3:// URI (bucket locked to dev bucket per SSOT)
SNAPSHOT_URI_REGEX: Final = re.compile(
    r"^s3://cvitx-uploads-dev-jdfirme/demo_user/[0-9a-f-]{36}/[0-9a-f-]{36}/snapshots/CTX\d{4,}_CAM\d+_\d{6}_\d{6}\.jpg$"
)

# ------------------------------ Config (ENV-first) -------------------------- #
# Note: Defaults below are the canonical DEV values from the SSOT.
#       Override via env only if you know what you’re doing.

_AWS_REGION = _env("AWS_REGION", "ap-southeast-2")
_S3_BUCKET  = _env("S3_BUCKET",  "cvitx-uploads-dev-jdfirme")

# Queues (Standard)
_SQS_VIDEO_QUEUE_URL    = _env("SQS_VIDEO_QUEUE_URL",
                               "https://sqs.ap-southeast-2.amazonaws.com/118730128890/cvitx-video-tasks")
_SQS_SNAPSHOT_QUEUE_URL = _env("SQS_SNAPSHOT_QUEUE_URL",
                               "https://sqs.ap-southeast-2.amazonaws.com/118730128890/cvitx-snapshot-tasks")

# Database DSN (dev) — can be overridden by DB_URL env. Redacted in summaries.
_DB_URL = _env(
    "DB_URL",
    # Dev RDS from parent roadmap (explicitly stored as canonical)
    "postgresql+psycopg2://cvitx_admin:cvitx-pg-password@cvitx-dev-pg.crawocq82lpx.ap-southeast-2.rds.amazonaws.com:5432/cvitx",
)

# Worker knobs (identical in both workers; used where applicable)
_FRAME_STRIDE_DEFAULT = _env_int("FRAME_STRIDE_DEFAULT", 3)

# Snapshot invariants
_SNAPSHOT_SIZE           = _env_int("SNAPSHOT_SIZE", 640)         # final square side (px)
_SNAPSHOT_MARGIN         = _env_float("SNAPSHOT_MARGIN", 0.15)    # adaptive context
_SNAPSHOT_NEIGHBOR_IOU   = _env_float("SNAPSHOT_NEIGHBOR_IOU", 0.03)
_JPG_QUALITY             = _env_int("JPG_QUALITY", 95)

# YOLO inference knobs
_YOLO_IMGSZ = _env_int("YOLO_IMGSZ", 640)
_YOLO_CONF  = _env_float("YOLO_CONF", 0.35)
_YOLO_IOU   = _env_float("YOLO_IOU", 0.45)

# SQS hygiene
_SQS_VIS_TIMEOUT       = _env_int("SQS_VIS_TIMEOUT", 300)  # seconds
_SQS_HEARTBEAT_SEC     = _env_int("SQS_HEARTBEAT_SEC", 60) # seconds
_RECEIVE_WAIT_TIME_SEC = _env_int("RECEIVE_WAIT_TIME_SEC", 10)

# Optional feature flags for main_worker
_ENABLE_COLOR = _env_int("ENABLE_COLOR", 1)
_ENABLE_PLATE = _env_int("ENABLE_PLATE", 1)

# Logging style
_JSON_LOGS = bool(_env_int("PROGRESS_LOG_JSON", 0))

CONFIG: Dict[str, object] = MappingProxyType(
    {
        # Cloud
        "AWS_REGION": _AWS_REGION,
        "S3_BUCKET": _S3_BUCKET,
        "SQS_VIDEO_QUEUE_URL": _SQS_VIDEO_QUEUE_URL,
        "SQS_SNAPSHOT_QUEUE_URL": _SQS_SNAPSHOT_QUEUE_URL,
        "DB_URL": _DB_URL,

        # Invariants / knobs
        "FRAME_STRIDE_DEFAULT": _FRAME_STRIDE_DEFAULT,
        "SNAPSHOT_SIZE": _SNAPSHOT_SIZE,
        "SNAPSHOT_MARGIN": _SNAPSHOT_MARGIN,
        "SNAPSHOT_NEIGHBOR_IOU": _SNAPSHOT_NEIGHBOR_IOU,
        "JPG_QUALITY": _JPG_QUALITY,

        "YOLO_IMGSZ": _YOLO_IMGSZ,
        "YOLO_CONF": _YOLO_CONF,
        "YOLO_IOU": _YOLO_IOU,

        "SQS_VIS_TIMEOUT": _SQS_VIS_TIMEOUT,
        "SQS_HEARTBEAT_SEC": _SQS_HEARTBEAT_SEC,
        "RECEIVE_WAIT_TIME_SEC": _RECEIVE_WAIT_TIME_SEC,

        # Features (main_worker)
        "ENABLE_COLOR": _ENABLE_COLOR,
        "ENABLE_PLATE": _ENABLE_PLATE,

        # Logging
        "JSON_LOGS": _JSON_LOGS,

        # Frozen taxonomies/regex (for validators)
        "YOLO_VEHICLE_TYPES": YOLO_VEHICLE_TYPES,
        "TYPE_TO_ID": TYPE_TO_ID,
        "VIDEO_EXTS": VIDEO_EXTS,
        "RAW_KEY_REGEX": RAW_KEY_REGEX,
        "SNAPSHOT_URI_REGEX": SNAPSHOT_URI_REGEX,
    }
)

# ------------------------------- Bundles (paths) ---------------------------- #
# Keep local-relative defaults for zero-conf; allow env override.

BUNDLES: Dict[str, str] = MappingProxyType(
    {
        # YOLO + tracker (yolo_worker)
        "YOLO_WEIGHTS": _env("YOLO_WEIGHTS", "yolo_worker/bundle/yolov8m.pt"),
        "DEEPSORT":     _env("DEEPSORT",     "yolo_worker/bundle/deepsort.engine"),

        # Main model bundles (main_worker)
        # Use either a single 'MAIN' or the split Baseline/CMT (preferred)
        "BASELINE": _env("BASELINE_BUNDLE_PATH", "main_worker/bundle/baseline.pt"),
        "CMT":      _env("CMT_BUNDLE_PATH",      "main_worker/bundle/cmt.pt"),
        # Optional single-model mode:
        "MAIN":     _env("MAIN_BUNDLE_PATH",     "main_worker/bundle/main.pt"),
    }
)

# -------------------------- Derived helpers (public) ------------------------ #

def s3_uri(key: str) -> str:
    """Build s3:// URI from a bucket-relative key."""
    return f"s3://{CONFIG['S3_BUCKET']}/{key}"

def build_snapshot_s3_uri(
    workspace_id: str,
    video_id: str,
    workspace_code: str,
    camera_code: str,
    track_id: int,
    offset_ms: int,
) -> str:
    """
    Deterministic snapshot s3:// URI, locked to:
    s3://<BUCKET>/demo_user/<wid>/<vid>/snapshots/CTX####_CAM#_{track:06d}_{offsetMs:06d}.jpg
    """
    key = (
        f"demo_user/{workspace_id}/{video_id}/snapshots/"
        f"{workspace_code}_{camera_code}_{track_id:06d}_{offset_ms:06d}.jpg"
    )
    return s3_uri(key)

def config_summary() -> str:
    """Human-friendly one-liner (safe to log)."""
    return (
        "CFG{region=%s, bucket=%s, videoQ=%s, snapQ=%s, db=%s, "
        "imgsz=%d, yolo(conf=%.2f,iou=%.2f), snap(640, m=%.2f, nIoU=%.2f), "
        "sqs(vis=%ds,hb=%ds), color=%s, plate=%s, yolo_types=%d}"
        % (
            CONFIG["AWS_REGION"],
            CONFIG["S3_BUCKET"],
            CONFIG["SQS_VIDEO_QUEUE_URL"],
            CONFIG["SQS_SNAPSHOT_QUEUE_URL"],
            _redact_dsn(CONFIG["DB_URL"]),  # redact password
            CONFIG["YOLO_IMGSZ"],
            CONFIG["YOLO_CONF"],
            CONFIG["YOLO_IOU"],
            CONFIG["SNAPSHOT_MARGIN"],
            CONFIG["SNAPSHOT_NEIGHBOR_IOU"],
            CONFIG["SQS_VIS_TIMEOUT"],
            CONFIG["SQS_HEARTBEAT_SEC"],
            "on" if CONFIG["ENABLE_COLOR"] else "off",
            "on" if CONFIG["ENABLE_PLATE"] else "off",
            len(YOLO_VEHICLE_TYPES),
        )
    )

def assert_config_sane() -> None:
    """Fail fast on obvious misconfigurations."""
    # Region/queue alignment
    if "ap-southeast-2" not in str(CONFIG["SQS_VIDEO_QUEUE_URL"]):
        raise ValueError("SQS_VIDEO_QUEUE_URL must be in ap-southeast-2 per SSOT.")
    if "ap-southeast-2" not in str(CONFIG["SQS_SNAPSHOT_QUEUE_URL"]):
        raise ValueError("SQS_SNAPSHOT_QUEUE_URL must be in ap-southeast-2 per SSOT.")

    # Snapshot invariants
    if CONFIG["SNAPSHOT_SIZE"] != 640:
        raise ValueError("SNAPSHOT_SIZE must be exactly 640 per SSOT.")
    if not (0.0 <= float(CONFIG["SNAPSHOT_MARGIN"]) <= 0.5):
        raise ValueError("SNAPSHOT_MARGIN out of sane range [0.0, 0.5].")
    if not (0.0 <= float(CONFIG["SNAPSHOT_NEIGHBOR_IOU"]) <= 0.5):
        raise ValueError("SNAPSHOT_NEIGHBOR_IOU out of sane range [0.0, 0.5].")

    # YOLO invariants
    if CONFIG["YOLO_IMGSZ"] != 640:
        raise ValueError("YOLO_IMGSZ must be 640 to match model and snapshot size.")
    if not (0.0 < float(CONFIG["YOLO_CONF"]) <= 1.0):
        raise ValueError("YOLO_CONF must be in (0, 1].")
    if not (0.0 < float(CONFIG["YOLO_IOU"]) <= 1.0):
        raise ValueError("YOLO_IOU must be in (0, 1].")

    # Types sanity — exact 8-class, no-space sequence (SSOT)
    _expected = (
        "Car",
        "SUV",
        "Van",
        "LightTruck",
        "Utility",
        "Motorcycle",
        "CarouselBus",
        "E-Jeepney",
    )
    if tuple(YOLO_VEHICLE_TYPES) != _expected:
        raise ValueError("YOLO_VEHICLE_TYPES must be exactly 8 classes in this order: Car,SUV,Van,LightTruck,Utility,Motorcycle,CarouselBus,E-Jeepney")

# Eager sanity check on import (explicit by design; fail fast)
assert_config_sane()

# Public exports
__all__ = [
    "CONFIG",
    "BUNDLES",
    "YOLO_VEHICLE_TYPES",
    "TYPE_TO_ID",
    "VIDEO_EXTS",
    "RAW_KEY_REGEX",
    "SNAPSHOT_URI_REGEX",
    "s3_uri",
    "build_snapshot_s3_uri",
    "config_summary",
    "assert_config_sane",
]
