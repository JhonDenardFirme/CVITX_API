from api.analysis.contracts import parse_analyze_image_message
import os, io, json, time, traceback, base64, decimal
from datetime import datetime, timezone
import uuid
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont

import boto3
import botocore
import psycopg2, psycopg2.extras

# --- engine & contracts -------------------------------------------------------
from api.analysis import engine as E
try:
    pass
except Exception:
    # Fallback parser (lenient): expects either {"s3_uri": "...", "workspace_id": "...", "analysis_id": ...}
    def __legacy__parse_analyze_image_message(body: str) -> Dict[str, Any]:
        d = json.loads(body)
        # allow simple forms or nested {"src":{"s3_uri":...}}
        s3_uri = d.get("s3_uri") or (d.get("src", {}) or {}).get("s3_uri")
        if not s3_uri or not isinstance(s3_uri, str) or not s3_uri.startswith("s3://"):
            raise ValueError("Message missing s3_uri")
        return {
            "workspace_id": d.get("workspace_id") or d.get("workspace") or "ws-dev",
            "analysis_id": d.get("analysis_id") or d.get("analysis_no") or d.get("id") or 0,
            "s3_uri": s3_uri,
        }

# --- constants ----------------------------------------------------------------
VARIANT = "baseline"
SVC     = "cvitx-worker"

# --- envs ---------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-2"
SQS_URL    = os.getenv("SQS_ANALYSIS_BASELINE_URL")  # REQUIRED for prod loop
VIS_TIMEOUT = int(os.getenv("SQS_VIS_TIMEOUT", "120"))
WAIT_SECS   = int(os.getenv("SQS_WAIT_SECS", "20"))
ENABLE_ANNOTATION = os.getenv("ENABLE_ANNOTATION", "1") not in ("0", "false", "False")
DEST_BUCKET = os.getenv("RESULTS_S3_BUCKET") or os.getenv("S3_BUCKET")  # optional: default to input bucket if unset
JPEG_Q = int(os.getenv("JPEG_QUALITY", "90"))

# DB: either DATABASE_URL or discrete vars
DB_URL   = (os.getenv("DATABASE_URL") or os.getenv("DB_URL"))
if DB_URL:
    DB_URL = DB_URL.replace("postgresql+psycopg2","postgresql")
DB_HOST  = os.getenv("DB_HOST")
DB_PORT  = os.getenv("DB_PORT", "5432")
DB_NAME  = os.getenv("DB_NAME")
DB_USER  = os.getenv("DB_USER")
DB_PASS  = os.getenv("DB_PASSWORD") or os.getenv("DB_PASS")

# --- clients ------------------------------------------------------------------
boto3.setup_default_session(region_name=AWS_REGION)
_s3  = boto3.client("s3")
_sqs = boto3.client("sqs")

# --- logging (JSON lines, consistent with engine) ------------------------------
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
    if not (DB_HOST and DB_NAME and DB_USER):
        raise RuntimeError("DB connection envs missing (DATABASE_URL or DB_HOST/DB_NAME/DB_USER/DB_PASSWORD).")
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS,
        connect_timeout=5, application_name=f"cvitx_{VARIANT}"
    )


# --- id resolution -------------------------------------------------------------
def _resolve_analysis_id(aid_raw, ws):
    s = str(aid_raw).strip()
    # UUID path
    try:
        return str(uuid.UUID(s))
    except Exception:
        pass
    # analysis_no integer path
    try:
        no = int(s)
    except Exception:
        raise ValueError('analysis_id must be a UUID or integer analysis_no')
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id FROM public.image_analyses WHERE workspace_id=%s AND analysis_no=%s LIMIT 1', (ws, no))
            row = cur.fetchone()
            if not row:
                raise ValueError(f'analysis_no {no} not found for workspace {ws}')
            return str(row[0])

UPSERT_SQL = """
INSERT INTO image_analysis_results AS r (
    analysis_id,
    model_variant,
    workspace_id,
    type,
    type_conf,
    make,
    make_conf,
    model,
    model_conf,
    parts,
    colors,
    plate_text,
    plate_conf,
    annotated_image_s3_key,
    vehicle_image_s3_key,
    plate_image_s3_key,
    latency_ms,
    gflops,
    memory_usage,
    updated_at
)
VALUES (
    %(analysis_id)s,
    %(variant)s,
    %(workspace_id)s,
    %(type)s,
    %(type_conf)s,
    %(make)s,
    %(make_conf)s,
    %(model)s,
    %(model_conf)s,
    %(parts_jsonb)s,
    %(colors_jsonb)s,
    %(plate_text)s,
    %(plate_conf)s,
    %(annotated_key)s,
    %(vehicle_key)s,
    %(plate_key)s,
    %(latency_ms)s,
    %(gflops)s,
    %(mem_gb)s,
    NOW()
)
ON CONFLICT (analysis_id, model_variant)
DO UPDATE SET
    type=EXCLUDED.type,
    type_conf=EXCLUDED.type_conf,
    make=EXCLUDED.make,
    make_conf=EXCLUDED.make_conf,
    model=EXCLUDED.model,
    model_conf=EXCLUDED.model_conf,
    parts=EXCLUDED.parts,
    colors=EXCLUDED.colors,
    plate_text=EXCLUDED.plate_text,
    plate_conf=EXCLUDED.plate_conf,
    annotated_image_s3_key=EXCLUDED.annotated_image_s3_key,
    vehicle_image_s3_key=EXCLUDED.vehicle_image_s3_key,
    plate_image_s3_key=EXCLUDED.plate_image_s3_key,
    latency_ms=EXCLUDED.latency_ms,
    gflops=EXCLUDED.gflops,
    memory_usage=EXCLUDED.memory_usage,
    updated_at=NOW();
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
    assert uri.startswith("s3://")
    rest = uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key

def _s3_get_bytes(bucket: str, key: str) -> bytes:
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def _jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=JPEG_Q); return buf.getvalue()

def _s3_put_image(bucket: str, key: str, img: Image.Image):
    body = _jpeg_bytes(img)
    _s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="image/jpeg", ACL="private")
    return key

# --- drawing / crops -----------------------------------------------------------
def _draw_anno(img: Image.Image, dets: Dict[str, Any]) -> Image.Image:
    im = img.copy().convert("RGB")
    dr = ImageDraw.Draw(im)
    # basic label
    label = []
    if dets.get("type") is not None:  label.append(str(dets["type"]))
    if dets.get("make") is not None:  label.append(str(dets["make"]))
    if dets.get("model") is not None: label.append(str(dets["model"]))
    label = " / ".join(label) or "vehicle"
    # boxes
    def _rect(b, color):
        if not b: return
        x1,y1,x2,y2 = map(int, b)
        dr.rectangle([x1,y1,x2,y2], outline=color, width=3)
    _rect(dets.get("veh_box"),   "red")
    _rect(dets.get("plate_box"), "yellow")
    # title
    dr.text((10,10), label, fill="white", stroke_width=2, stroke_fill="black")
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
    if hasattr(E, "load_model"):
        E.load_model(VARIANT)
        log("INFO","warm_loaded", note="engine.load_model cached", variant=VARIANT)
        return
    # Fallback: a tiny dummy call just to initialize shaders/kernels
    import io
    from PIL import Image as _PILImage
    img = _PILImage.new("RGB",(64,64),(100,100,100))
    b = io.BytesIO(); img.save(b, format="JPEG"); data = b.getvalue()
    _ = E.run_inference(data, variant=VARIANT, analysis_id="worker_warmup")

# --- core processing -----------------------------------------------------------
def _process_one_message(body: str, receipt_handle: str):
    msg = parse_analyze_image_message(body)
    ws  = str(msg['workspace_id']).strip()
    try:
        aid = _resolve_analysis_id(msg['analysis_id'], ws)
    except Exception as e:
        log('WARN','bad_message', note=str(e))
        try:
            if 'SQS_URL' in globals() and SQS_URL and receipt_handle:
                _sqs.delete_message(QueueUrl=SQS_URL, ReceiptHandle=receipt_handle)
        except Exception:
            pass
        return
    src_bucket, src_key = _parse_s3_uri(msg["input_image_s3_uri"])

    # bytes → inference
    t0 = time.time()
    if not src_key:

        # Skip bad payloads (e.g., empty S3 key); delete message and continue

        try:

            _sqs.delete_message(QueueUrl=(SQS_URL if 'SQS_URL' in globals() else QUEUE_URL), ReceiptHandle=receipt_handle)

        except Exception:

            pass

        log("WARN","bad_message", note="empty_s3_key")

        return

    img_bytes = _s3_get_bytes(src_bucket, src_key)
    dets, timings, metrics = E.run_inference(img_bytes, variant=VARIANT, analysis_id=f"job_{aid}_{VARIANT}")

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

    # db upsert
    payload = {
        "workspace_id": ws,
        "analysis_id": aid, "variant": VARIANT,
        "type": dets.get("type"), "type_conf": dets.get("type_conf"),
        "make": dets.get("make"), "make_conf": dets.get("make_conf"),
        "model": dets.get("model"), "model_conf": dets.get("model_conf"),
        "parts_jsonb": json.dumps(dets.get("parts") or []),
        "colors_jsonb": json.dumps(dets.get("colors") or []),
        "plate_text": dets.get("plate_text") or None,
        "plate_conf": dets.get("plate_conf"),
        "annotated_key": keys["annotated"],
        "vehicle_key":   keys["vehicle"],
        "plate_key":     keys["plate"],
        "latency_ms": int(float(metrics.get("latency_ms") or timings.get("total") or 0.0)),
        "gflops": metrics.get("gflops"),
        "mem_gb": metrics.get("mem_gb"),
    }

    with _connect() as conn:
        conn.autocommit = True
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(UPSERT_SQL, payload)
            _update_parent_status(conn, aid)

    dt_ms = int((time.time() - t0) * 1000)
    log("INFO", "upsert", analysis_id=aid, latency_ms=metrics.get("latency_ms"), end_to_end_ms=dt_ms,
        s3={k:v for k,v in keys.items() if v}, summary={
            "type": dets.get("type"), "make": dets.get("make"), "model": dets.get("model")
        })

    # ack message
    if SQS_URL and receipt_handle:
        _sqs.delete_message(QueueUrl=SQS_URL, ReceiptHandle=receipt_handle)
        log("INFO","sqs_delete", analysis_id=aid)

# --- main loop -----------------------------------------------------------------
def main():
    log("INFO","start", note="worker starting")
    _warm_model()
    if not SQS_URL:
        log("ERROR","no_queue_url", note="Set SQS_ANALYSIS_BASELINE_URL to enable the loop.")
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
            log("WARN","shutdown")
            break
        except Exception as e:
            log("ERROR","loop_error", err=str(e), trace=traceback.format_exc())
            time.sleep(2)

if __name__ == "__main__":
    main()
