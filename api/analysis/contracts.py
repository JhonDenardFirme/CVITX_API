# api/analysis/contracts.py
# -*- coding: utf-8 -*-
"""
Contract helpers for Image Analysis.

⚠ Pipeline stability promise:
- We DO NOT change the existing SQS message parsing signature or behavior that your
  workers rely on. `parse_analyze_image_message(body: str) -> Dict[str, Any]` remains.
- We only *broaden* compatibility to accept both legacy URI and new S3 key messages.

Additions:
- Optional Pydantic models to validate/shape engine outputs before DB writes.
  These cover FBL colors, metrics (latency/gflops/memory), and result assets.

Phase-9 Notes (SSOT):
- SQS messages may include either legacy 's3_uri'/'input_image_s3_uri' or modern 'input_s3_key'.
- Result colors MUST follow FBL schema: [{finish, base, lightness, conf}], 1–3 entries.
- Result metrics commonly include latency_ms, gflops, memory (GB), and usage ratio.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

# ----------------------------- Pipeline-safe parser -----------------------------
class ContractError(Exception):
    pass


def parse_analyze_image_message(body: str) -> Dict[str, Any]:
    """
    Parse SQS message for ANALYZE_IMAGE.

    Back-compat:
      - Accepts either 'input_image_s3_uri' (preferred legacy) or 's3_uri'.
      - Also accepts modern 'input_s3_key' (Phase-9 enqueue path).
    Requirement:
      - 'event' must be 'ANALYZE_IMAGE'.
      - 'analysis_id', 'workspace_id', and 'model_variant' are required.
      - At least one of {'input_image_s3_uri'|'s3_uri'|'input_s3_key'} must be present.

    Returns a dict that preserves original fields and may include both
    'input_image_s3_uri' (if provided) and 'input_s3_key' (if provided).
    """
    try:
        d = json.loads(body)
    except Exception as e:
        raise ContractError(f"Invalid JSON: {e}")

    # Normalize legacy alias
    if "input_image_s3_uri" not in d and "s3_uri" in d:
        d["input_image_s3_uri"] = d["s3_uri"]

    # Basic required keys (pipeline-critical)
    base_required = ["event", "analysis_id", "workspace_id", "model_variant"]
    missing_base = [k for k in base_required if k not in d]
    if missing_base:
        raise ContractError(f"Missing fields: {missing_base}")

    if d["event"] != "ANALYZE_IMAGE":
        raise ContractError(f"Unexpected event: {d['event']}")

    # Source location: accept either URI or S3 key (Phase-9 path)
    has_uri = ("input_image_s3_uri" in d)
    has_key = ("input_s3_key" in d)
    if not (has_uri or has_key):
        raise ContractError("Missing input source: need one of ['input_image_s3_uri'|'s3_uri'|'input_s3_key'].")

    return d


# ----------------------------- Optional result models -----------------------------
# These models are OPTIONAL sugar for validating/structuring engine outputs
# before persisting to DB. They do not change worker behavior.

from pydantic import BaseModel, Field, root_validator


class ColorFBL(BaseModel):
    finish: Optional[str] = None
    base: Optional[str] = None
    lightness: Optional[str] = None
    conf: float = 0.0


class AnalysisMetrics(BaseModel):
    latency_ms: float
    gflops: Optional[float] = None

    # Accept either mem_gb or memory_gb from producers; unify into mem_gb
    mem_gb: Optional[float] = Field(default=None, alias="memory_gb")
    memory_usage: Optional[float] = None  # ratio in [0,1]
    device: Optional[str] = None
    trained: Optional[bool] = None

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True

    @root_validator(pre=True)
    def _coalesce_memory_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Prefer explicit mem_gb; else map memory_gb -> mem_gb
        if "mem_gb" not in values and "memory_gb" in values:
            values["mem_gb"] = values.get("memory_gb")
        return values


class ResultAssets(BaseModel):
    annotated_image_s3_key: Optional[str] = None
    vehicle_image_s3_key: Optional[str] = None
    plate_image_s3_key: Optional[str] = None

    # When presign=1 is used at read-time (API), URLs may be injected transiently.
    annotated_url: Optional[str] = None
    vehicle_url: Optional[str] = None
    plate_url: Optional[str] = None


class EngineResult(BaseModel):
    # Core recognitions (add/omit freely — model remains backward compatible)
    type: Optional[str] = None
    type_conf: Optional[float] = None
    make: Optional[str] = None
    make_conf: Optional[float] = None
    model: Optional[str] = None
    model_conf: Optional[float] = None

    # Parts/evidence/thresholds (opaque dicts are fine to keep contract loose)
    parts: Optional[Dict[str, Any]] = None
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    below_threshold: Dict[str, Any] = Field(default_factory=dict)
    evidence: Optional[Dict[str, Any]] = None

    # Plates (optional feature)
    plate_text: Optional[str] = None
    plate_conf: Optional[float] = None
    plate_box: Optional[List[float]] = None
    plate_candidates: Optional[List[Dict[str, Any]]] = None

    # Colors — MUST be FBL schema if present
    colors: List[ColorFBL] = Field(default_factory=list)

    # Metrics & assets
    metrics: Optional[AnalysisMetrics] = None
    assets: Optional[ResultAssets] = None

    # Status, errors (for worker rows or API-computed)
    status: Optional[str] = None
    error_msg: Optional[str] = None


# ----------------------------- Helpers (optional) -----------------------------
def coerce_fbl_colors(payload: Dict[str, Any]) -> List[ColorFBL]:
    """
    Helper to ensure FBL color list on mixed/legacy inputs.
    - If 'colors' already looks like FBL, it is parsed directly.
    - If legacy shapes exist (e.g., hex-only), map them into FBL with nulls where unknown.

    Returns a validated list[ColorFBL]. Safe to use before DB persistence.
    """
    raw = payload.get("colors", [])
    fbl_list: List[ColorFBL] = []

    def _as_fbl(item: Any) -> Optional[ColorFBL]:
        if not isinstance(item, dict):
            return None
        if any(k in item for k in ("finish", "base", "lightness", "conf")):
            return ColorFBL(**{
                "finish": item.get("finish"),
                "base": item.get("base"),
                "lightness": item.get("lightness"),
                "conf": float(item.get("conf", 0.0) or 0.0),
            })
        # Legacy example: {"hex":"#aabbcc","p":0.62}
        if "hex" in item or "p" in item:
            return ColorFBL(finish=None, base=None, lightness=None, conf=float(item.get("p") or 0.0))
        return None

    for it in raw if isinstance(raw, list) else []:
        f = _as_fbl(it)
        if f:
            fbl_list.append(f)

    return fbl_list


__all__ = [
    # pipeline-critical
    "ContractError",
    "parse_analyze_image_message",
    # optional models (API-side validation)
    "ColorFBL",
    "AnalysisMetrics",
    "ResultAssets",
    "EngineResult",
    # optional helpers
    "coerce_fbl_colors",
]
