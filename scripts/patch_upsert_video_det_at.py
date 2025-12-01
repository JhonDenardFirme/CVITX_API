#!/usr/bin/env python
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
target = ROOT / "video_analysis" / "worker_utils" / "common.py"

FUNC_START = "def upsert_video_detection_and_progress(\n"
ANCHOR_AFTER = "# ============================ Key / Path Builders =========================== #\n"

NEW_FUNC = """def upsert_video_detection_and_progress(
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
    \"""
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
    \"""
    # Normalize detected_at (accept ISO strings or datetime/None)
    # We want to:
    #   • Accept either an ISO8601 string (from SNAPSHOT_READY.detectedAt)
    #     or a datetime/None passed directly by callers.
    #   • Always have a well-defined local det_at for both the INSERT/UPDATE
    #     and the per-run counters (last_snapshot_at).
    det_at: Optional[datetime]
    if isinstance(detected_at, str):
        # Accept "...Z" or with explicit offset; _parse_iso handles both and
        # returns a timezone-aware datetime or None on failure.
        det_at = _parse_iso(detected_at)
    else:
        # Already a datetime or None — use as-is.
        det_at = detected_at

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

    sql_det = \"""
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
    \"""

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
                "detected_at": det_at,
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
                \"""
                UPDATE video_analyses
                   SET processed_snapshots = processed_snapshots + 1,
                       processed_ok        = processed_ok + CASE WHEN :ok THEN 1 ELSE 0 END,
                       processed_err       = processed_err + CASE WHEN :ok THEN 0 ELSE 1 END,
                       last_snapshot_at    = COALESCE(:detected_at, last_snapshot_at),
                       updated_at          = now()
                 WHERE id = :aid
                \"""
            ),
            {"aid": analysis_id, "ok": is_ok, "detected_at": det_at},
        )
"""

def main() -> int:
    text = target.read_text(encoding="utf-8")

    if FUNC_START not in text:
        print("ERROR: upsert_video_detection_and_progress() not found", file=sys.stderr)
        return 1
    if ANCHOR_AFTER not in text:
        print("ERROR: Key/Path Builders anchor not found", file=sys.stderr)
        return 1

    before, rest = text.split(FUNC_START, 1)
    # rest currently starts at function body; we want to cut up to just before Key/Path Builders
    if ANCHOR_AFTER not in rest:
        print("ERROR: anchor after function not found inside tail", file=sys.stderr)
        return 1

    func_and_tail, after = rest.split(ANCHOR_AFTER, 1)
    new_text = before + FUNC_START + NEW_FUNC + "\n\n" + ANCHOR_AFTER + after

    target.write_text(new_text, encoding="utf-8")
    print("Patched:", target)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
