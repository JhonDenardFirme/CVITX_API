# file: video_analysis/main_worker/worker.py
# -*- coding: utf-8 -*-
"""
CVITX · Video Analysis — main_worker (CMT)
Consumes SNAPSHOT_READY messages, runs the real CMT engine, writes DB rows and artifacts.

Key properties:
• Uses the same engine path as image workers: from api.analysis import engine as E
• Model bundle loading is ENV-driven: CMT_BUNDLE_PATH points to a DIRECTORY
  with label_maps.json + *.pt (no hard-coded bundle paths in code)
• Real S3 image bytes → E.run_inference(...) → write results & artifacts
• Preserves the video_analyses / video_analysis_results schema and logic
• Normalizes DB URL scheme if needed (postgresql+psycopg2:// → postgresql://)
"""

import io, json, time, traceback
from typing import Any, Dict, Tuple, Optional
from urllib.parse import urlparse

from PIL import Image

from video_analysis.worker_utils.common import (
    CONFIG,
    get_logger,
    get_s3,
    get_sqs,
    validate_snapshot_ready,
    create_video_analysis,
    upsert_video_results,
    set_video_analysis_status,
)
from video_analysis.worker_config import config_summary

# ----------------------------------------------------------------
# (AUTHORIZED CHANGE) CONFIG → ENV bridge for engine behavior parity
# ----------------------------------------------------------------
import os as _os

# Snapshot/image size
if "SNAPSHOT_SIZE" in CONFIG:
    _os.environ.setdefault("IMG_SIZE", str(int(CONFIG["SNAPSHOT_SIZE"])))

# Feature toggles
_os.environ.setdefault(
    "ENABLE_COLOR", "1" if CONFIG.get("ENABLE_COLOR", True) else "0"
)
_os.environ.setdefault(
    "ENABLE_PLATE", "1" if CONFIG.get("ENABLE_PLATE", True) else "0"
)

# On-disk bundle directory (REQUIRED for trained weights)
_os.environ.setdefault(
    "CMT_BUNDLE_PATH",
    "/home/ubuntu/cvitx/video_analysis/main_worker/bundle/cmt_dir",
)

# Optional thresholds/temperatures (keep consistent with CONFIG if applicable)
_os.environ.setdefault("TAU_TYPE", "0.70")
_os.environ.setdefault("TAU_MAKE", "0.70")
_os.environ.setdefault("TAU_MODEL", "0.70")
_os.environ.setdefault("TEMP_TYPE", "1.00")
_os.environ.setdefault("TEMP_MAKE", "1.00")
_os.environ.setdefault("TEMP_MODEL", "1.00")

# ---- logging / aws clients --------------------------------------
log = get_logger("cvitx.video.main")

REGION = CONFIG["AWS_REGION"]
BUCKET = CONFIG["S3_BUCKET"]
s3 = get_s3()
sqs = get_sqs()
Q_SNAPSHOT = CONFIG["SQS_SNAPSHOT_QUEUE_URL"]

# ---- s3 helpers -------------------------------------------------


def _norm_uri(s: str) -> str:
    return s if s.startswith("s3://") else f"s3://{s}"


def _bucket_key(uri_or_key: str) -> Tuple[str, str]:
    p = urlparse(_norm_uri(uri_or_key))
    return p.netloc, p.path.lstrip("/")


def _jpeg_bytes(img: Image.Image, q: int = 95) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    return buf.getvalue()


# ---- engine (ENV-driven, parity with image worker) --------------
# IMPORTANT: use the same engine as image workers; it reads CMT_BUNDLE_PATH (DIRECTORY)
from api.analysis import engine as E


# ---- artifacts --------------------------------------------------


def _save_artifacts(
    aid: str, wid: str, pil: Image.Image, dets: Dict[str, Any]
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    prefix = f"{wid}/{aid}/main/"
    try:
        try:
            # Reuse image worker helpers when present
            from api.workers.image_worker_baseline import (  # type: ignore
                _draw_anno,
                _crop,
            )
        except Exception:
            _draw_anno = lambda im, d: im  # no-op fallback

            def _crop(im: Image.Image, box):
                if not box:
                    return None
                x1, y1, x2, y2 = map(int, map(round, box))
                x1 = max(0, min(x1, im.width - 1))
                y1 = max(0, min(y1, im.height - 1))
                x2 = max(x1 + 1, min(x2, im.width))
                y2 = max(y1 + 1, min(y2, im.height))
                return im.crop((x1, y1, x2, y2))

        anno = _draw_anno(pil, dets)
        s3.put_object(
            Bucket=BUCKET,
            Key=prefix + "annotated.jpg",
            Body=_jpeg_bytes(anno),
            ContentType="image/jpeg",
        )
        out["annotated_image_s3_key"] = prefix + "annotated.jpg"

        if dets.get("veh_box"):
            veh = _crop(pil, dets["veh_box"])
            if veh:
                s3.put_object(
                    Bucket=BUCKET,
                    Key=prefix + "vehicle.jpg",
                    Body=_jpeg_bytes(veh),
                    ContentType="image/jpeg",
                )
                out["vehicle_image_s3_key"] = prefix + "vehicle.jpg"

        if dets.get("plate_box"):
            plc = _crop(pil, dets["plate_box"])
            if plc:
                s3.put_object(
                    Bucket=BUCKET,
                    Key=prefix + "plate.jpg",
                    Body=_jpeg_bytes(plc),
                    ContentType="image/jpeg",
                )
                out["plate_image_s3_key"] = prefix + "plate.jpg"
    except Exception as e:
        log.warning("artifact_failed: %s", e)
    return out


# ---- main processing --------------------------------------------


def _process_one(msg_body: Dict[str, Any], receipt: str) -> None:
    """
    Process a single SNAPSHOT_READY message.

    • On success:
        - Creates/updates video_analyses + video_analysis_results (variant="main")
        - Marks parent analysis status = 'done'
        - Deletes the SQS message
    • On failure (after aid is known):
        - Marks parent analysis status = 'error' with error_msg
        - Re-queues the SQS message with a short visibility timeout
    """
    aid: Optional[str] = None

    try:
        # 1) Validate SNAPSHOT_READY via shared Pydantic schema
        snap = validate_snapshot_ready(msg_body)
        wid = snap["workspace_id"]
        vid = snap["video_id"]
        bkt, key = _bucket_key(snap["snapshot_s3_key"])

        # 2) Create analysis row tied to video (or reuse existing snapshot)
        aid = create_video_analysis(f"s3://{bkt}/{key}", wid, vid)

        # 3) Download image and run inference (CMT as main)
        obj = s3.get_object(Bucket=bkt, Key=key)
        img_bytes = obj["Body"].read()

        # Engine: warm or on-demand load of env-driven CMT bundle (DIRECTORY)
        try:
            if hasattr(E, "load_model"):
                E.load_model("cmt")
        except Exception as e:
            log.warning(
                "[warm] load failed (continuing, will retry if needed): %s", e
            )

        dets, timings, metrics = E.run_inference(
            img_bytes,
            variant="cmt",
            analysis_id=f"vid_{aid}",
        )

        # 4) Optional artifacts
        assets: Dict[str, str] = {}
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            assets = _save_artifacts(aid, wid, pil, dets)
        except Exception as e:
            log.warning("artifact_io: %s", e)

        # 5) Normalize payload for shared upsert helper (FBL colors already list)
        colors_fbl = dets.get("colors") or []
        result: Dict[str, Any] = {
            "type": {"label": dets.get("type"), "conf": dets.get("type_conf")},
            "make": {"label": dets.get("make"), "conf": dets.get("make_conf")},
            "model": {"label": dets.get("model"), "conf": dets.get("model_conf")},
            "plate": {
                "text": dets.get("plate_text"),
                "conf": dets.get("plate_conf"),
            },
            "colors": colors_fbl[:3],
            "assets": assets,
            "latency_ms": int(
                (metrics.get("latency_ms") or timings.get("total") or 0.0)
            ),
            "memory_gb": metrics.get("mem_gb") or metrics.get("memory_gb"),
            "status": "done",
            "error_msg": None,
        }

        # 6) UPSERT results and mark parent done
        upsert_video_results(aid, "main", result)
        set_video_analysis_status(aid, "done", None)

        log.info(
            "[upsert] aid=%s variant=main type=%s make=%s model=%s | %s",
            aid,
            result["type"]["label"],
            result["make"]["label"],
            result["model"]["label"],
            config_summary(),
        )

        # 7) Acknowledge SQS message only after DB writes succeed
        sqs.delete_message(QueueUrl=Q_SNAPSHOT, ReceiptHandle=receipt)

    except Exception as e:
        log.error("[error] process failed: %s", e)
        log.debug(traceback.format_exc())

        # If we know the analysis id, mark it as error
        if aid is not None:
            try:
                set_video_analysis_status(aid, "error", str(e)[:500])
            except Exception as db_err:
                log.error("[error] failed to mark analysis as error: %s", db_err)

        # Re-queue the message for another attempt (or DLQ after retries)
        try:
            sqs.change_message_visibility(
                QueueUrl=Q_SNAPSHOT,
                ReceiptHandle=receipt,
                VisibilityTimeout=10,
            )
        except Exception as sqs_err:
            log.error(
                "[error] failed to change message visibility after error: %s",
                sqs_err,
            )

        # No re-raise: SQS + DB status have been handled here.
        return


# ---- daemon loop ------------------------------------------------


def main() -> None:
    log.info(
        "[boot] video main-model worker starting… region=%s bucket=%s queue=%s | %s",
        REGION,
        BUCKET,
        Q_SNAPSHOT,
        config_summary(),
    )

    # Preload bundle once (best-effort, env-driven directory)
    try:
        if hasattr(E, "load_model"):
            E.load_model("cmt")
            log.info("[warm] CMT model loaded (env-driven bundle).")
    except Exception as e:
        log.warning("[warm] initial load failed (lazy on-demand): %s", e)

    if not Q_SNAPSHOT:
        log.error("No SQS_SNAPSHOT_QUEUE_URL configured in code.")
        return

    while True:
        try:
            r = sqs.receive_message(
                QueueUrl=Q_SNAPSHOT,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=CONFIG["RECEIVE_WAIT_TIME_SEC"],
                VisibilityTimeout=CONFIG["SQS_VIS_TIMEOUT"],
                AttributeNames=["All"],
            )
            msgs = r.get("Messages", [])
            if not msgs:
                continue

            m = msgs[0]
            receipt = m["ReceiptHandle"]
            body = (
                json.loads(m["Body"])
                if isinstance(m.get("Body"), str)
                else m.get("Body", {})
            )

            try:
                _process_one(body, receipt)
            except Exception as e:
                # Failsafe: _process_one is supposed to handle SQS + DB status.
                # This block only catches truly unexpected errors.
                log.error("[loop] unexpected error in _process_one: %s", e)
                log.debug(traceback.format_exc())
                try:
                    sqs.change_message_visibility(
                        QueueUrl=Q_SNAPSHOT,
                        ReceiptHandle=receipt,
                        VisibilityTimeout=10,
                    )
                except Exception as sqs_err:
                    log.error(
                        "[loop] failed to change message visibility after exception: %s",
                        sqs_err,
                    )
                continue

            # No delete_message here; _process_one handles it after successful DB writes.

        except KeyboardInterrupt:
            log.warning("shutdown")
            break
        except Exception as e:
            log.error("[loop] error: %s", e)
            log.debug(traceback.format_exc())
            time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("interrupted by user")
    except Exception as e:
        log.error("[fatal] %s", e)
        log.debug(traceback.format_exc())
        raise

