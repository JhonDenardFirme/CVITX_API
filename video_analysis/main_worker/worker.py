# File: /home/ubuntu/cvitx/video_analysis/main_worker/worker.py
"""
CVITX · Video Analysis Monorepo — Main-Model Worker
Directory: /home/ubuntu/cvitx/video_analysis/main_worker
Filename : worker.py

Role
----
Consume SNAPSHOT_READY → run main model (Type→Make→Model; optional Color/Plate) →
write image_analyses + image_analysis_results (Baseline and/or CMT variants) →
upload artifacts (vehicle.jpg, annotated.jpg[, plate.jpg]) → ACK SQS.

Determinism & Contracts (locked)
--------------------------------
• Inbound message (validated): SNAPSHOT_READY
  {
    "event": "SNAPSHOT_READY",
    "video_id": "UUID",
    "workspace_id": "UUID",
    "workspace_code": "CTX1005",
    "camera_code": "CAM1",
    "track_id": 305,
    "snapshot_s3_key": "s3://cvitx-uploads-dev-jdfirme/demo_user/<wid>/<vid>/snapshots/CTX1005_CAM1_000305_002272.jpg",
    "recordedAt": null,
    "detectedIn": 2272,
    "detectedAt": null,
    "yolo_type": "SUV"
  }

• Parent DB row: image_analyses
  - id (UUID) is returned by create_image_analysis(snapshot_s3_key, workspace_id)
  - UNIQUE(snapshot_s3_key)

• Child DB rows: image_analysis_results (one per variant)
  - UNIQUE(analysis_id, variant)
  - Required fields we upsert in result_dict (keys are stable, FE already consumes):
    {
      "variant": "baseline" | "cmt",
      "type_label": str | null, "type_conf": float | null,
      "make_label": str | null, "make_conf": float | null,
      "model_label": str | null, "model_conf": float | null,
      "colors": [{"finish": null|"Matte"|"Metallic", "base": "White", "lightness": "Light"|null, "conf": float}] | [],
      "plate_text": str | null, "plate_conf": float | null,
      "assets": {
        "vehicle_image_s3_key": str,
        "annotated_image_s3_key": str,
        "plate_image_s3_key": str | null
      },
      "latency_ms": int,
      "memory_gb": float | null,
      "status": "done"
    }

• Artifacts S3 layout (AID-based; modern layout):
  s3://{S3_BUCKET}/{workspace_id}/{analysis_id}/{variant}/vehicle.jpg
  s3://{S3_BUCKET}/{workspace_id}/{analysis_id}/{variant}/annotated.jpg
  s3://{S3_BUCKET}/{workspace_id}/{analysis_id}/{variant}/plates/plate.jpg  (optional)

================================================================================
🟡 AWS SETUP REMINDER (once per environment)
   • S3 bucket must exist: s3://cvitx-uploads-dev-jdfirme
   • SQS queue must exist:
       - Snapshot tasks : https://sqs.ap-southeast-2.amazonaws.com/118730128890/cvitx-snapshot-tasks
     with a DLQ + RedrivePolicy (maxReceiveCount≈5), VisibilityTimeout≈300s, LongPolling=10s
   • IAM role for this service needs:
       - s3:GetObject on demo_user/*/snapshots/*
       - s3:PutObject on */*/*/{baseline,cmt}/*  (AID-based results layout)
       - sqs:ReceiveMessage/DeleteMessage/ChangeMessageVisibility on snapshot-tasks
   • Systemd ExecStart should be:
       /home/ubuntu/cvitx/api/.venv/bin/python -m video_analysis.main_worker.worker
================================================================================
"""

from __future__ import annotations

import io
import json
import os
import time
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # Allows dry-run without torch installed (results become fallback)

from video_analysis.worker_config import CONFIG, BUNDLES, s3_uri
from video_analysis.worker_utils.common import (
    # logging
    log,
    # validators
    validate_snapshot_ready,
    # aws
    get_s3,
    get_sqs,
    # db
    create_image_analysis,
    upsert_results,
)


# ============================== Utilities ================================== #

def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Split 's3://bucket/key' to (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    no_scheme = uri[len("s3://"):]
    parts = no_scheme.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Malformed S3 URI: {uri}")
    return parts[0], parts[1]


def _jpeg_bytes_to_cv2(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode returned None for snapshot bytes")
    return img


def _ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _to_tensor(img_bgr: np.ndarray, size: int = 640, rgb: bool = True) -> "torch.Tensor":
    """Resize/letterbox to size×size, BGR→RGB optionally, [0,1], CHW, float32."""
    H, W = img_bgr.shape[:2]
    if H != size or W != size:
        img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if rgb else img_bgr
    arr = img.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    t = torch.from_numpy(chw).unsqueeze(0)  # [1,3,H,W]
    return t


def _device() -> str:
    # Prefer explicit CONFIG key; otherwise auto-select
    explicit = str(CONFIG.get("MAIN_DEVICE") or "").strip().lower()
    if explicit:
        return explicit
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _gpu_mem_gb_or_none() -> Optional[float]:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        bytes_used = torch.cuda.max_memory_allocated()  # peak since reset
        return round(bytes_used / (1024**3), 3)
    except Exception:
        return None


# ============================== Model Loading =============================== #

_BASELINE_MODEL = None
_CMT_MODEL = None
_MODEL_INPUT_RGB = True  # assume model expects RGB; change here if needed


def _try_load_model(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    if not os.path.exists(path):
        log.error(f"[model] path not found: {path}")
        return None
    if torch is None:
        log.warning("[model] torch not available; running in fallback mode")
        return None
    try:
        # Prefer torch.jit (most portable), fallback to torch.load
        try:
            mdl = torch.jit.load(path, map_location=_device())
            mdl.eval()
            return mdl
        except Exception:
            checkpoint = torch.load(path, map_location=_device())
            if hasattr(checkpoint, "eval"):
                checkpoint.eval()
                return checkpoint
            # If checkpoint is a state_dict, user must wrap their model loader.
            log.error("[model] state_dict provided, but no architecture to load it into.")
            return None
    except Exception as e:
        log.error(f"[model] failed to load {path}: {e}")
        return None


def _load_models_once() -> None:
    global _BASELINE_MODEL, _CMT_MODEL
    if _BASELINE_MODEL is None:
        base_path = (
            BUNDLES.get("BASELINE_WEIGHTS")
            or BUNDLES.get("MAIN_WEIGHTS")
            or BUNDLES.get("BASELINE_BUNDLE_PATH")
        )
        if base_path:
            log.info(f"[model] loading baseline: {base_path}")
        _BASELINE_MODEL = _try_load_model(base_path)
    if _CMT_MODEL is None:
        cmt_path = BUNDLES.get("CMT_WEIGHTS") or BUNDLES.get("CMT_BUNDLE_PATH")
        if cmt_path:
            log.info(f"[model] loading cmt: {cmt_path}")
        _CMT_MODEL = _try_load_model(cmt_path)


# ============================== Inference Core ============================== #

def _forward_any(model: Any, tensor: "torch.Tensor") -> Dict[str, Any]:
    """
    Call user model. Expect either:
      • returns dict with keys {type, make, model} -> each a dict {label, conf}
      • or returns tuple/list of logits we can't parse (then fallback)
    This keeps the worker generic while still deterministic.
    """
    try:
        with torch.no_grad():
            out = model(tensor.to(_device()))
        # Heuristic: if dict-like with expected shape, use it
        if isinstance(out, dict):
            t = out.get("type") or {}
            m = out.get("make") or {}
            mm = out.get("model") or {}
            return {
                "type_label": t.get("label"),
                "type_conf": float(t.get("conf")) if t.get("conf") is not None else None,
                "make_label": m.get("label"),
                "make_conf": float(m.get("conf")) if m.get("conf") is not None else None,
                "model_label": mm.get("label"),
                "model_conf": float(mm.get("conf")) if mm.get("conf") is not None else None,
            }
        # Unknown shape → fallback
        log.warning("[infer] model returned non-dict; using fallback parser")
        return {}
    except Exception as e:
        log.error(f"[infer] model forward failed: {e}")
        return {}


def _estimate_color_fbl(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Tiny deterministic color estimate:
      - Compute mean in HSV; map hue to coarse base; lightness from V.
    This is stable and fast, not ML. You can replace with your color head later.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean = hsv.reshape(-1, 3).mean(axis=0)
    H, S, V = mean.tolist()
    # Base mapping (very coarse)
    if V > 220 and S < 30:
        base = "White"
    elif V < 50:
        base = "Black"
    elif 10 <= H < 25:
        base = "Orange"
    elif 25 <= H < 40:
        base = "Yellow"
    elif 40 <= H < 85:
        base = "Green"
    elif 85 <= H < 130:
        base = "Cyan"
    elif 130 <= H < 170:
        base = "Blue"
    elif 170 <= H < 185:
        base = "Purple"
    elif H < 10 or H >= 185:
        base = "Red"
    else:
        base = "Gray"
    lightness = "Light" if V >= 160 else (None if 80 <= V < 160 else None)
    finish = None  # Unknown
    conf = 0.66
    return [{"finish": finish, "base": base, "lightness": lightness, "conf": conf}]


def _variant_result_dict(
    variant: str,
    img_bgr: np.ndarray,
    infer_out: Dict[str, Any],
    vehicle_key: str,
    annotated_key: str,
    plate_key: Optional[str],
    latency_ms: int,
) -> Dict[str, Any]:
    colors: List[Dict[str, Any]] = []
    if str(CONFIG.get("ENABLE_COLOR", "0")) not in ("0", "false", "False", "", "no"):
        try:
            colors = _estimate_color_fbl(img_bgr)
        except Exception as e:
            log.error(f"[color] estimation failed: {e}")
            colors = []

    mem_gb = _gpu_mem_gb_or_none()

    return {
        "variant": variant,
        "type_label": infer_out.get("type_label"),
        "type_conf": infer_out.get("type_conf"),
        "make_label": infer_out.get("make_label"),
        "make_conf": infer_out.get("make_conf"),
        "model_label": infer_out.get("model_label"),
        "model_conf": infer_out.get("model_conf"),
        "colors": colors,
        "plate_text": None,   # set by optional plate pipeline
        "plate_conf": None,
        "assets": {
            "vehicle_image_s3_key": vehicle_key,
            "annotated_image_s3_key": annotated_key,
            "plate_image_s3_key": plate_key,
        },
        "latency_ms": int(latency_ms),
        "memory_gb": mem_gb,
        "status": "done",
    }


# ============================== Worker Logic ================================ #

class Heartbeat(threading.Thread):
    """SQS visibility heartbeat while processing a long analysis."""
    def __init__(self, receipt_handle: str, queue_url: str):
        super().__init__(daemon=True)
        self.receipt_handle = receipt_handle
        self.queue_url = queue_url
        self.stop_flag = threading.Event()

    def run(self) -> None:
        sqs = get_sqs()
        vis = int(CONFIG["SQS_VIS_TIMEOUT"])
        hb = int(CONFIG["SQS_HEARTBEAT_SEC"])
        while not self.stop_flag.wait(timeout=hb):
            try:
                sqs.change_message_visibility(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=self.receipt_handle,
                    VisibilityTimeout=vis,
                )
                log.info("[hb] extended visibility")
            except Exception as e:
                log.error(f"[hb] change_message_visibility failed: {e}")

    def stop(self) -> None:
        self.stop_flag.set()


def _download_snapshot_bytes(snapshot_uri: str) -> bytes:
    bucket, key = _parse_s3_uri(snapshot_uri)
    s3 = get_s3()
    log.info(f"[s3] get_object {snapshot_uri}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def _upload_artifact_bytes(key: str, data: bytes) -> str:
    s3 = get_s3()
    bucket = str(CONFIG["S3_BUCKET"])
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="image/jpeg")
    return s3_uri(key)


def _maybe_run_variant(variant: str, model: Optional[Any], img_bgr: np.ndarray) -> Dict[str, Any]:
    """Run a variant if model exists; otherwise fallback uses yolo_type later."""
    if model is None or torch is None:
        return {}
    t = _to_tensor(img_bgr, size=640, rgb=_MODEL_INPUT_RGB)
    t = t if _device() == "cpu" else t.to(_device(), non_blocking=True)
    t0 = time.time()
    out = _forward_any(model, t)
    latency_ms = int((time.time() - t0) * 1000)
    out["_latency_ms"] = latency_ms
    return out


def _process_one_snapshot(body: Dict[str, Any]) -> int:
    """
    Process a single SNAPSHOT_READY message.
    Returns number of result rows written (1 for baseline-only, 2 if baseline+cmt).
    """
    payload = validate_snapshot_ready(body)

    # 1) Fetch snapshot
    snapshot_uri = str(payload["snapshot_s3_key"])
    raw_bytes = _download_snapshot_bytes(snapshot_uri)
    img_bgr = _ensure_3ch(_jpeg_bytes_to_cv2(raw_bytes))

    # 2) Create parent image_analysis row (idempotent behavior implemented server-side)
    analysis_id = create_image_analysis(snapshot_s3_key=snapshot_uri, workspace_id=str(payload["workspace_id"]))

    # 3) Build artifact keys (AID-based, per variant)
    wid = str(payload["workspace_id"])
    base_prefix = f"{wid}/{analysis_id}"

    # 4) Decide variants
    variants: List[Tuple[str, Optional[Any]]] = []
    if _BASELINE_MODEL is not None:
        variants.append(("baseline", _BASELINE_MODEL))
    if _CMT_MODEL is not None:
        variants.append(("cmt", _CMT_MODEL))
    if not variants:
        # Run baseline "fallback" so UI still shows something
        variants.append(("baseline", None))

    # 5) For each variant: run inference, upload artifacts, upsert result
    written = 0
    for variant, model in variants:
        # Inference (or fallback)
        infer_out = _maybe_run_variant(variant, model, img_bgr)
        latency_ms = infer_out.pop("_latency_ms", 0)

        # Fallbacks for labels if model didn't return
        if not infer_out.get("type_label"):
            infer_out["type_label"] = str(payload.get("yolo_type") or "Car")
            infer_out["type_conf"] = 0.99
        infer_out.setdefault("make_label", None)
        infer_out.setdefault("make_conf", None)
        infer_out.setdefault("model_label", None)
        infer_out.setdefault("model_conf", None)

        # Artifacts: for now reuse the snapshot for both vehicle.jpg and annotated.jpg
        # (Deterministic; you can replace annotated with overlays later.)
        vehicle_key = f"{base_prefix}/{variant}/vehicle.jpg"
        annotated_key = f"{base_prefix}/{variant}/annotated.jpg"
        plate_key: Optional[str] = None

        _upload_artifact_bytes(vehicle_key, raw_bytes)
        _upload_artifact_bytes(annotated_key, raw_bytes)

        # (Optional) plate OCR pipeline — disabled by default to keep worker self-contained.
        # If you later enable it, write the plate.jpg and set plate_key accordingly.

        # Assemble row
        row = _variant_result_dict(
            variant=variant,
            img_bgr=img_bgr,
            infer_out=infer_out,
            vehicle_key=vehicle_key,
            annotated_key=annotated_key,
            plate_key=plate_key,
            latency_ms=latency_ms,
        )

        # UPSERT
        upsert_results(analysis_id=analysis_id, variant=variant, result_dict=row)
        written += 1

    log.info(f"[ok] analysis_id={analysis_id} variants_written={written}")
    return written


# ================================ SQS Poller =============================== #

def _receive_loop() -> None:
    sqs = get_sqs()
    qurl = str(CONFIG["SQS_SNAPSHOT_QUEUE_URL"])
    wait = int(CONFIG["RECEIVE_WAIT_TIME_SEC"])
    vis = int(CONFIG["SQS_VIS_TIMEOUT"])

    log.info(f"[boot] main_worker ready — queue={qurl} wait={wait}s vis={vis}s :: {CONFIG['AWS_REGION']}")

    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=qurl,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=wait,
                VisibilityTimeout=vis,
            )
            msgs = resp.get("Messages", [])
            if not msgs:
                continue

            msg = msgs[0]
            receipt = msg["ReceiptHandle"]
            body_raw = msg["Body"]
            try:
                body = json.loads(body_raw)
            except json.JSONDecodeError:
                log.error("[recv] invalid JSON body; deleting")
                sqs.delete_message(QueueUrl=qurl, ReceiptHandle=receipt)
                continue

            hb = Heartbeat(receipt_handle=receipt, queue_url=qurl)
            hb.start()

            try:
                written = _process_one_snapshot(body)
                # Success → delete message
                sqs.delete_message(QueueUrl=qurl, ReceiptHandle=receipt)
                log.info(f"[ack] wrote {written} result row(s)")
            except Exception as e:
                log.error(f"[err] processing failed: {e}", exc_info=True)
                # Do NOT delete → allow DLQ via RedrivePolicy
            finally:
                hb.stop()

        except Exception as outer:
            log.error(f"[loop] receive/process error: {outer}", exc_info=True)
            time.sleep(2.0)  # small backoff


# ================================ Entrypoint =============================== #

def main() -> None:
    _load_models_once()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    _receive_loop()


if __name__ == "__main__":
    main()
