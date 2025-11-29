# file: video_analysis/main_worker/utils/contracts.py
# -*- coding: utf-8 -*-
"""
Contract helpers for Video Analysis (SNAPSHOT_READY + optional ANALYZE_IMAGE for parity).
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, ValidationError
from video_analysis.worker_utils.common import SnapshotReady


class ContractError(Exception):
    pass


# ----------------------------- ANALYZE_IMAGE (compat) -----------------------------


def parse_analyze_image_message(body: str) -> Dict[str, Any]:
    try:
        d = json.loads(body)
    except Exception as e:
        raise ContractError(f"Invalid JSON: {e}")
    if "input_image_s3_uri" not in d and "s3_uri" in d:
        d["input_image_s3_uri"] = d["s3_uri"]
    base_required = ["event", "analysis_id", "workspace_id", "model_variant"]
    missing_base = [k for k in base_required if k not in d]
    if missing_base:
        raise ContractError(f"Missing fields: {missing_base}")
    if d["event"] != "ANALYZE_IMAGE":
        raise ContractError(f"Unexpected event: {d['event']}")
    if not ("input_image_s3_uri" in d or "input_s3_key" in d):
        raise ContractError(
            "Missing input source: need one of ['input_image_s3_uri'|'s3_uri'|'input_s3_key']."
        )
    return d


# ----------------------------- SNAPSHOT_READY (strict, canonical-backed) -----------------------------


def parse_snapshot_ready(body: str) -> Dict[str, Any]:
    """
    Strict SNAPSHOT_READY parser.

    • Decodes JSON body.
    • Validates against the canonical SnapshotReady Pydantic model
      (same one used by workers via validate_snapshot_ready).
    • Raises ContractError on any shape/regex mismatch.
    """
    try:
        raw = json.loads(body)
    except Exception as e:
        raise ContractError(f"Invalid JSON: {e}")

    try:
        snap = SnapshotReady.model_validate(raw)
    except ValidationError as e:
        # Normalize Pydantic error into our contract-level error type
        raise ContractError(f"SNAPSHOT_READY validation error: {e}") from e

    # Return a plain dict with Python-native types (no Pydantic objects)
    return snap.model_dump(mode="python")


# ----------------------------- Optional models -----------------------------


class ColorFBL(BaseModel):
    finish: Optional[str] = None
    base: Optional[str] = None
    lightness: Optional[str] = None
    conf: float = 0.0


class AnalysisMetrics(BaseModel):
    latency_ms: float
    gflops: Optional[float] = None
    mem_gb: Optional[float] = Field(default=None, alias="memory_gb")
    memory_usage: Optional[float] = None  # ratio 0..1
    device: Optional[str] = None
    trained: Optional[bool] = None

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True

    @root_validator(pre=True)
    def _coalesce_memory_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "mem_gb" not in values and "memory_gb" in values:
            values["mem_gb"] = values.get("memory_gb")
        return values


class ResultAssets(BaseModel):
    annotated_image_s3_key: Optional[str] = None
    vehicle_image_s3_key: Optional[str] = None
    plate_image_s3_key: Optional[str] = None
    annotated_url: Optional[str] = None
    vehicle_url: Optional[str] = None
    plate_url: Optional[str] = None


class EngineResult(BaseModel):
    type: Optional[str] = None
    type_conf: Optional[float] = None
    make: Optional[str] = None
    make_conf: Optional[float] = None
    model: Optional[str] = None
    model_conf: Optional[float] = None
    parts: Optional[Dict[str, Any]] = None
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    below_threshold: Dict[str, Any] = Field(default_factory=dict)
    evidence: Optional[Dict[str, Any]] = None
    plate_text: Optional[str] = None
    plate_conf: Optional[float] = None
    plate_box: Optional[List[float]] = None
    plate_candidates: Optional[List[Dict[str, Any]]] = None
    colors: List[ColorFBL] = Field(default_factory=list)
    metrics: Optional[AnalysisMetrics] = None
    assets: Optional[ResultAssets] = None
    status: Optional[str] = None
    error_msg: Optional[str] = None


# Helpers


def coerce_fbl_colors(payload: Dict[str, Any]) -> List[ColorFBL]:
    raw = payload.get("colors", [])
    fbl_list: List[ColorFBL] = []

    def _as_fbl(item: Any) -> Optional[ColorFBL]:
        if not isinstance(item, dict):
            return None
        if any(k in item for k in ("finish", "base", "lightness", "conf")):
            return ColorFBL(
                **{
                    "finish": item.get("finish"),
                    "base": item.get("base"),
                    "lightness": item.get("lightness"),
                    "conf": float(item.get("conf", 0.0) or 0.0),
                }
            )
        if "hex" in item or "p" in item:
            return ColorFBL(
                finish=None,
                base=None,
                lightness=None,
                conf=float(item.get("p") or 0.0),
            )
        return None

    for it in raw if isinstance(raw, list) else []:
        f = _as_fbl(it)
        if f:
            fbl_list.append(f)
    return fbl_list


__all__ = [
    "ContractError",
    "parse_analyze_image_message",
    "parse_snapshot_ready",
    "ColorFBL",
    "AnalysisMetrics",
    "ResultAssets",
    "EngineResult",
    "coerce_fbl_colors",
]

