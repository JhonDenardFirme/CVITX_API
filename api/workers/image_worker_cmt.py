
# api/workers/image_worker_cmt.py

"""
CVITX · CMT Worker (independent)
- Pulls ANALYZE_IMAGE jobs from SQS (CMT queue)
- Runs engine inference (variant='cmt')
- Writes crops/annotation to S3 under <workspace>/<analysis_id>/cmt/
- UPSERTs results into image_analysis_results
- Updates parent image_analyses.status when both variants present

IMPORTANT: Pipeline-safe. No schema changes. Colors are FBL-only.
"""

from __future__ import annotations

import os, io, json, time, traceback
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

import boto3
import psycopg2, psycopg2.extras
from PIL import Image, ImageDraw

# --- engine & contracts -------------------------------------------------------
from api.analysis.contracts import parse_analyze_image_message
from api.analysis import engine as E

# --- constants ----------------------------------------------------------------
VARIANT = "cmt"
SVC     = "cvitx-worker"

# --- envs ---------------------------------------------------------------------
AWS_REGION  = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-1"
SQS_URL     = os.getenv("SQS_ANALYSIS_CMT_URL")  # REQUIRED for loop (CMT queue)
VIS_TIMEOUT = int(os.getenv("SQS_VIS_TIMEOUT", "120"))
WAIT_SECS   = int(os.getenv("SQS_WAIT_SECS", "20"))

ENABLE_ANNOTATION = os.getenv("ENABLE_ANNOTATION", "1") not in ("0", "false", "False")
DEST_BUCKET = os.getenv("RESULTS_S3_BUCKET") or os.getenv("S3_BUCKET")  # default to input bucket if unset
JPEG_Q      = int(os.getenv("JPEG_QUALITY", "90"))

# DB: either DATABASE_URL or discrete vars
DB_URL  = (os.getenv("DATABASE_URL") or os.getenv("DB_URL"))
if DB_URL:
    DB_URL = DB_URL.replace("postgresql+psycopg2", "postgresql")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD") or os.getenv("DB_PASS")

# --- clients ------------------------------------------------------------------
boto3.setup_default_session(region_name=AWS_REGION)
_s3  = boto3.client("s3")
_sqs = boto3.client("sqs")

# --- logging (JSON lines) -----------------------------------------------------
def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def log(level: str, event: str, **kw):
    rec = {"ts": _now_iso(), "level": level, "svc": SVC, "variant": VARIANT, "event": event}
    rec.update(kw)
    print(json.dumps(rec, separators=(",", ":"), ensure_ascii=False), flush=True)

# --- db helpers ----------------------------------------------------------------
def _connect():
    if DB_URL:
        return psycopg2.connect(DB_URL)
    if not (DB_HOST and DB_NAME and DB_USER and DB_PASS):
        raise RuntimeError("DB connection envs missing (DATABASE_URL or DB_HOST/DB_NAME/DB_USER/DB_PASSWORD).")
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS,
        connect_timeout=5, application_name=f"cvitx_{VARIANT}"
    )

UPSERT_SQL = """
INSERT INTO image_analysis_results AS r (
    analysis_id, model_variant,
    type, type_conf, make, make_conf, model, model_conf,
    parts, colors, plate_text, plate_conf,
    annotated_image_s3_key, vehicle_image_s3_key, plate_image_s3_key,
    latency_ms, gflops, memory_usage, updated_at,
    colors_fbl, colors_overall_conf
)
VALUES (
    %(analysis_id)s, %(variant)s,
    %(type)s, %(type_conf)s, %(make)s, %(make_conf)s, %(model)s, %(model_conf)s,
    %(parts_jsonb)s, %(colors_jsonb)s, %(plate_text)s, %(plate_conf)s,
    %(annotated_key)s, %(vehicle_key)s, %(plate_key)s,
    %(latency_ms)s, %(gflops)s, %(memory_usage)s, NOW(),
    %(colors_fbl_jsonb)s, %(colors_overall_conf)s
)
ON CONFLICT (analysis_id, model_variant) DO UPDATE SET
    type=EXCLUDED.type, type_conf=EXCLUDED.type_conf,
    make=EXCLUDED.make, make_conf=EXCLUDED.make_conf,
    model=EXCLUDED.model, model_conf=EXCLUDED.model_conf,
    parts=EXCLUDED.parts, colors=EXCLUDED.colors,
    plate_text=EXCLUDED.plate_text, plate_conf=EXCLUDED.plate_conf,
    annotated_image_s3_key=EXCLUDED.annotated_image_s3_key,
    vehicle_image_s3_key=EXCLUDED.vehicle_image_s3_key,
    plate_image_s3_key=EXCLUDED.plate_image_s3_key,
    latency_ms=EXCLUDED.latency_ms, gflops=EXCLUDED.gflops,
    memory_usage=EXCLUDED.memory_usage,
    updated_at=NOW(),
    colors_fbl=EXCLUDED.colors_fbl,
    colors_overall_conf=EXCLUDED.colors_overall_conf;
"""

def _update_parent_status(conn, analysis_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE image_analyses a SET status = CASE
                WHEN (SELECT COUNT(*) FROM image_analysis_results r
                      WHERE r.analysis_id=a.id AND r.model_variant IN ('baseline','cmt')) >= 2
                THEN 'done' ELSE 'processing' END
            WHERE a.id = %s;
        """, (analysis_id,))

# --- s3 helpers ----------------------------------------------------------------
def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri or not uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")
    rest = uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key

def _s3_get_bytes(bucket: str, key: str) -> bytes:
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def _jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_Q)
    return buf.getvalue()

def _s3_put_image(bucket: str, key: str, img: Image.Image):
    body = _jpeg_bytes(img.convert("RGB"))
    _s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="image/jpeg", ACL="private")
    return key

# --- drawing / crops -----------------------------------------------------------
def _draw_anno(img: Image.Image, dets: Dict[str, Any]) -> Image.Image:
    """
    Draw vehicle + plate + per-part boxes.
    veh_box, plate_box are in ORIGINAL coords (engine maps back).
    _debug_parts_sq in 640x640 square coords; map back via _debug_pad_scale (pad, scale).
    """
    im = img.copy().convert("RGB")
    dr = ImageDraw.Draw(im)

    # Title
    label = []
    if dets.get("type")  is not None: label.append(str(dets["type"]))
    if dets.get("make")  is not None: label.append(str(dets["make"]))
    if dets.get("model") is not None: label.append(str(dets["model"]))
    title = " / ".join(label) or "vehicle"

    def _rect_xyxy(b, color="red", width=3):
        if not b: return None
        x1,y1,x2,y2 = map(int, map(round, b))
        x1 = max(0, min(x1, im.width  - 1))
        y1 = max(0, min(y1, im.height - 1))
        x2 = max(x1+1, min(x2, im.width))
        y2 = max(y1+1, min(y2, im.height))
        dr.rectangle([x1,y1,x2,y2], outline=color, width=width)
        return (x1,y1,x2,y2)

    # Vehicle & Plate
    _rect_xyxy(dets.get("veh_box"),   "red",    3)
    _rect_xyxy(dets.get("plate_box"), "yellow", 3)

    # Parts: square → original
    parts   = dets.get("_debug_parts_sq") or []
    padinfo = dets.get("_debug_pad_scale") or {}
    pad     = padinfo.get("pad") or [0.0, 0.0]
    scale   = float(padinfo.get("scale") or 1.0)
    px, py  = float(pad[0]), float(pad[1])
    s       = scale if scale else 1.0

    def _sq_to_orig(x, y):
        xo = (float(x) - px) / max(1e-12, s)
        yo = (float(y) - py) / max(1e-12, s)
        X = max(0, min(int(round(xo)), im.width  - 1))
        Y = max(0, min(int(round(yo)), im.height - 1))
        return X, Y

    for p in parts:
        box = p.get("box_sq") or []
        if len(box) == 4:
            x1o, y1o = _sq_to_orig(box[0], box[1])
            x2o, y2o = _sq_to_orig(box[2], box[3])
            if x2o <= x1o: x2o = min(im.width  - 1, x1o + 2)
            if y2o <= y1o: y2o = min(im.height - 1, y1o + 2)
            _rect_xyxy((x1o, y1o, x2o, y2o), "cyan", 2)
            nm = str(p.get("name", "part"))
            cf = float(p.get("conf", 0.0))
            dr.text((x1o, y1o + 2), f"{nm}:{cf:.2f}", fill="cyan")

    dr.text((10,10), title, fill="white", stroke_width=2, stroke_fill="black")
    return im

def _crop(img: Image.Image, box: Any) -> Optional[Image.Image]:
    if not box: return None
    x1,y1,x2,y2 = map(int, map(round, box))
    x1 = max(0, min(x1, img.width-1))
    y1 = max(0, min(y1, img.height-1))
    x2 = max(x1+1, min(x2, img.width))
    y2 = max(y1+1, min(y2, img.height))
    return img.crop((x1,y1,x2,y2))

# --- model warm cache ----------------------------------------------------------
def _warm_model():
    try:
        if hasattr(E, "load_model"):
            E.load_model(VARIANT)
            log("INFO", "warm_loaded", note="engine.load_model cached")
            return
    except Exception as e:
        log("WARN", "warm_load_failed", note=str(e))
    # Fallback: tiny dummy to init kernels
    from PIL import Image as _PILImage
    img = _PILImage.new("RGB", (64, 64), (100, 100, 100))
    b = io.BytesIO(); img.save(b, format="JPEG"); data = b.getvalue()
    try:
        _ = E.run_inference(data, variant=VARIANT, analysis_id="worker_warmup")
        log("INFO", "warm_dummy_done")
    except Exception as e:
        log("WARN", "warm_dummy_failed", note=str(e))

# --- core processing -----------------------------------------------------------
def _overall_fbl_conf(fbl: Any) -> float:
    try:
        if not fbl: return 0.0
        return float(max((float(x.get("conf", 0.0)) for x in fbl), default=0.0))
    except Exception:
        return 0.0

def _process_one_message(body: str, receipt_handle: str):
    # Parse message (supports input_image_s3_uri and legacy s3_uri via contracts)
    msg = parse_analyze_image_message(body)
    ws  = msg["workspace_id"]
    aid = msg["analysis_id"]
    src_bucket, src_key = _parse_s3_uri(msg["input_image_s3_uri"])

    # Skip/ack bad payloads early (empty key)
    if not src_key:
        try:
            if SQS_URL and receipt_handle:
                _sqs.delete_message(QueueUrl=SQS_URL, ReceiptHandle=receipt_handle)
        except Exception:
            pass
        log("WARN", "bad_message", note="empty_s3_key", analysis_id=aid)
        return

    # bytes → inference
    t0 = time.time()
    img_bytes = _s3_get_bytes(src_bucket, src_key)
    dets, timings, metrics = E.run_inference(
        img_bytes, variant=VARIANT, analysis_id=f"job_{aid}_{VARIANT}"
    )

    # destination bucket/key prefix
    dest_bucket = DEST_BUCKET or src_bucket
    prefix = f"{ws}/{aid}/{VARIANT}/"

    # crops/annotation
    keys = {"annotated": None, "vehicle": None, "plate": None}
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if ENABLE_ANNOTATION:
            anno = _draw_anno(pil, dets)
            keys["annotated"] = _s3_put_image(dest_bucket, prefix + "annotated.jpg", anno)
        if dets.get("veh_box"):
            veh = _crop(pil, dets["veh_box"])
            if veh: keys["vehicle"] = _s3_put_image(dest_bucket, prefix + "vehicle.jpg", veh)
        if dets.get("plate_box"):
            plc = _crop(pil, dets["plate_box"])
            if plc: keys["plate"] = _s3_put_image(dest_bucket, prefix + "plate.jpg", plc)
    except Exception as e:
        log("WARN", "s3_write_crops_failed", analysis_id=aid, note=str(e))

    # Prepare DB payload (schema-safe)
    colors_fbl = dets.get("colors") or []     # FBL-only in canon
    overall_cf = _overall_fbl_conf(colors_fbl)

    payload = {
        "analysis_id": aid,
        "variant": VARIANT,
        "type": dets.get("type"),
        "type_conf": dets.get("type_conf"),
        "make": dets.get("make"),
        "make_conf": dets.get("make_conf"),
        "model": dets.get("model"),
        "model_conf": dets.get("model_conf"),
        "parts_jsonb": json.dumps(dets.get("parts") or []),
        "colors_jsonb": json.dumps(colors_fbl),  # modern colors = FBL
        "plate_text": dets.get("plate_text") or None,
        "plate_conf": dets.get("plate_conf"),
        "annotated_key": keys["annotated"],
        "vehicle_key":   keys["vehicle"],
        "plate_key":     keys["plate"],
        "latency_ms": float(metrics.get("latency_ms") or timings.get("total") or 0.0),
        "gflops": metrics.get("gflops"),
        # DB column is 'memory_usage' (ratio in [0,1]); mem_gb not stored in this schema
        "memory_usage": metrics.get("memory_usage"),
        # FBL mirrors (gap closer per canon)
        "colors_fbl_jsonb": json.dumps(colors_fbl[:3]),
        "colors_overall_conf": float(overall_cf),
    }

    # DB upsert + parent update
    with _connect() as conn:
        conn.autocommit = True
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(UPSERT_SQL, payload)
        _update_parent_status(conn, aid)

    dt_ms = int((time.time() - t0) * 1000)
    log("INFO", "upsert",
        analysis_id=aid,
        latency_ms=metrics.get("latency_ms"),
        end_to_end_ms=dt_ms,
        memory_usage=metrics.get("memory_usage"),
        mem_gb=metrics.get("mem_gb"),  # logged for observability
        gflops=metrics.get("gflops"),
        s3={k: v for k, v in keys.items() if v},
        summary={"type": dets.get("type"), "make": dets.get("make"), "model": dets.get("model")}
    )

    # ack message
    if SQS_URL and receipt_handle:
        _sqs.delete_message(QueueUrl=SQS_URL, ReceiptHandle=receipt_handle)
        log("INFO", "sqs_delete", analysis_id=aid)

# --- main loop -----------------------------------------------------------------
def main():
    log("INFO", "start", note="worker starting")
    _warm_model()
    if not SQS_URL:
        log("ERROR", "no_queue_url", note="Set SQS_ANALYSIS_CMT_URL to enable the loop.")
        return
    while True:
        try:
            resp = _sqs.receive_message(
                QueueUrl=SQS_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=WAIT_SECS,
                VisibilityTimeout=VIS_TIMEOUT,
            )
            msgs = resp.get("Messages", [])
            if not msgs:
                continue
            m = msgs[0]
            _process_one_message(m["Body"], m["ReceiptHandle"])
        except KeyboardInterrupt:
            log("WARN", "shutdown")
            break
        except Exception as e:
            log("ERROR", "loop_error", err=str(e), trace=traceback.format_exc())
            time.sleep(2)

if __name__ == "__main__":
    main()

# --- Legacy wrapper note (not executed) ----------------------------------------
# PREV V1 used a thin wrapper that mutated baseline globals (base.VARIANT/QUEUE).
# With this independent worker, that wrapper can be retired after rollout.
# Also avoid import-time color injection blocks (undefined names) — push that logic into the engine.


