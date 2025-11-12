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

import io, json, time, uuid, logging, traceback
from typing import Any, Dict, Tuple, Optional
from urllib.parse import urlparse

import boto3
import psycopg2, psycopg2.extras
from PIL import Image

from video_analysis.worker_config import CONFIG, config_summary
from video_analysis.main_worker.utils.contracts import parse_snapshot_ready

# ----------------------------------------------------------------
# (AUTHORIZED CHANGE) CONFIG → ENV bridge for engine behavior parity
# ----------------------------------------------------------------
import os as _os

# Snapshot/image size
if "SNAPSHOT_SIZE" in CONFIG:
    _os.environ.setdefault("IMG_SIZE", str(int(CONFIG["SNAPSHOT_SIZE"])) )

# Feature toggles
_os.environ.setdefault("ENABLE_COLOR", "1" if CONFIG.get("ENABLE_COLOR", True) else "0")
_os.environ.setdefault("ENABLE_PLATE", "1" if CONFIG.get("ENABLE_PLATE", True) else "0")

# On-disk bundle directory (REQUIRED for trained weights)
_os.environ.setdefault("CMT_BUNDLE_PATH", "/home/ubuntu/cvitx/video_analysis/main_worker/bundle/cmt_dir")

# Optional thresholds/temperatures (keep consistent with CONFIG if applicable)
_os.environ.setdefault("TAU_TYPE", "0.70")
_os.environ.setdefault("TAU_MAKE", "0.70")
_os.environ.setdefault("TAU_MODEL", "0.70")
_os.environ.setdefault("TEMP_TYPE", "1.00")
_os.environ.setdefault("TEMP_MAKE", "1.00")
_os.environ.setdefault("TEMP_MODEL", "1.00")

# ---- logging ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cvitx.video.main")

# ---- aws clients ------------------------------------------------
REGION = CONFIG["AWS_REGION"]
BUCKET = CONFIG["S3_BUCKET"]
s3  = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)
Q_SNAPSHOT = CONFIG["SQS_SNAPSHOT_QUEUE_URL"]

# ---- db ---------------------------------------------------------
# Normalize DB URL for psycopg2 if it uses sqlalchemy-style scheme
DB_URL = CONFIG["DB_URL"]
if DB_URL and DB_URL.startswith("postgresql+psycopg2://"):
    DB_URL = DB_URL.replace("postgresql+psycopg2://", "postgresql://", 1)

def _connect():
    if not DB_URL:
        raise RuntimeError("DB_URL is required for video worker.")
    return psycopg2.connect(DB_URL)

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

# ---- sql (video lane, independent) ------------------------------
CREATE_SQL = """
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS video_analyses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID NOT NULL,
  video_id UUID NOT NULL,
  snapshot_s3_key TEXT UNIQUE NOT NULL,
  source_kind TEXT NOT NULL DEFAULT 'snapshot',
  status TEXT NOT NULL DEFAULT 'processing',   -- 'processing' | 'done' | 'error'
  error_msg TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS video_analysis_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id UUID NOT NULL REFERENCES video_analyses(id) ON DELETE CASCADE,
  variant TEXT NOT NULL,                        -- 'main' (CMT)
  type_label TEXT, type_conf DOUBLE PRECISION,
  make_label TEXT, make_conf DOUBLE PRECISION,
  model_label TEXT, model_conf DOUBLE PRECISION,
  plate_text TEXT, plate_conf DOUBLE PRECISION,
  colors JSONB,                                 -- FBL array
  assets JSONB,                                 -- {vehicle_image_s3_key, annotated_image_s3_key, plate_image_s3_key}
  latency_ms INTEGER,
  memory_gb DOUBLE PRECISION,
  status TEXT NOT NULL DEFAULT 'done',
  error_msg TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (analysis_id, variant)
);

CREATE INDEX IF NOT EXISTS ix_video_analyses_ws_vid
  ON video_analyses (workspace_id, video_id);

CREATE UNIQUE INDEX IF NOT EXISTS ux_video_analyses_snapshot
  ON video_analyses (snapshot_s3_key);

CREATE INDEX IF NOT EXISTS ix_video_results_analysis_variant
  ON video_analysis_results (analysis_id, variant);
"""

UPSERT_SQL = """
INSERT INTO video_analysis_results AS r (
  id, analysis_id, variant,
  type_label, type_conf, make_label, make_conf, model_label, model_conf,
  plate_text, plate_conf, colors, assets,
  latency_ms, memory_gb, status, error_msg, updated_at
) VALUES (
  %(id)s, %(analysis_id)s, %(variant)s,
  %(type_label)s, %(type_conf)s, %(make_label)s, %(make_conf)s, %(model_label)s, %(model_conf)s,
  %(plate_text)s, %(plate_conf)s, %(colors)s, %(assets)s,
  %(latency_ms)s, %(memory_gb)s, %(status)s, %(error_msg)s, now()
)
ON CONFLICT (analysis_id, variant) DO UPDATE SET
  type_label=EXCLUDED.type_label, type_conf=EXCLUDED.type_conf,
  make_label=EXCLUDED.make_label, make_conf=EXCLUDED.make_conf,
  model_label=EXCLUDED.model_label, model_conf=EXCLUDED.model_conf,
  plate_text=EXCLUDED.plate_text, plate_conf=EXCLUDED.plate_conf,
  colors=EXCLUDED.colors, assets=EXCLUDED.assets,
  latency_ms=EXCLUDED.latency_ms, memory_gb=EXCLUDED.memory_gb,
  status=EXCLUDED.status, error_msg=EXCLUDED.error_msg, updated_at=now();
"""

# ---- bootstrap --------------------------------------------------

def _ensure_tables():
    with _connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # (AUTHORIZED CHANGE) be resilient if pgcrypto creation is not permitted
            try:
                cur.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')
            except Exception:
                log.warning("pgcrypto_extension_skipped_or_forbidden")
            cur.execute(CREATE_SQL.replace('CREATE EXTENSION IF NOT EXISTS "pgcrypto";\n', ''))

def _create_analysis(workspace_id: str, video_id: str, snapshot_uri: str) -> str:
    with _connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO video_analyses (workspace_id, video_id, snapshot_s3_key, status) VALUES (%s,%s,%s,'processing') RETURNING id",
                (workspace_id, video_id, snapshot_uri)
            )
            return str(cur.fetchone()[0])

def _mark_status(analysis_id: str, status: str, err: Optional[str] = None):
    with _connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE video_analyses SET status=%s, error_msg=%s, updated_at=now() WHERE id=%s",
                (status, err, analysis_id)
            )

# ---- artifacts --------------------------------------------------

def _save_artifacts(aid: str, wid: str, pil: Image.Image, dets: Dict[str, Any]) -> Dict[str,str]:
    out = {}
    prefix = f"{wid}/{aid}/main/"
    try:
        try:
            # Reuse image worker helpers when present
            from api.workers.image_worker_baseline import _draw_anno, _crop  # type: ignore
        except Exception:
            _draw_anno = lambda im, d: im  # no-op fallback
            def _crop(im, box):
                if not box: return None
                x1,y1,x2,y2 = map(int, map(round, box))
                x1 = max(0, min(x1, im.width-1))
                y1 = max(0, min(y1, im.height-1))
                x2 = max(x1+1, min(x2, im.width))
                y2 = max(y1+1, min(y2, im.height))
                return im.crop((x1,y1,x2,y2))

        anno = _draw_anno(pil, dets)
        s3.put_object(Bucket=BUCKET, Key=prefix+"annotated.jpg", Body=_jpeg_bytes(anno), ContentType="image/jpeg")
        out["annotated_image_s3_key"] = prefix+"annotated.jpg"

        if dets.get("veh_box"):
            veh = _crop(pil, dets["veh_box"])
            if veh:
                s3.put_object(Bucket=BUCKET, Key=prefix+"vehicle.jpg", Body=_jpeg_bytes(veh), ContentType="image/jpeg")
                out["vehicle_image_s3_key"] = prefix+"vehicle.jpg"

        if dets.get("plate_box"):
            plc = _crop(pil, dets["plate_box"])
            if plc:
                s3.put_object(Bucket=BUCKET, Key=prefix+"plate.jpg", Body=_jpeg_bytes(plc), ContentType="image/jpeg")
                out["plate_image_s3_key"] = prefix+"plate.jpg"
    except Exception as e:
        log.warning("artifact_failed: %s", e)
    return out

# ---- main processing --------------------------------------------

def _process_one(msg_body: Dict[str, Any]):
    # 1) Validate SNAPSHOT_READY
    snap = parse_snapshot_ready(json.dumps(msg_body))
    wid = snap["workspace_id"]
    vid = snap["video_id"]
    bkt, key = _bucket_key(snap["snapshot_s3_key"])

    # 2) Create analysis row tied to video
    aid = _create_analysis(wid, vid, f"s3://{bkt}/{key}")

    # 3) Download image and run inference (CMT as main)
    obj = s3.get_object(Bucket=bkt, Key=key)
    img_bytes = obj["Body"].read()

    # Engine: warm or on-demand load of env-driven CMT bundle (DIRECTORY)
    try:
        if hasattr(E, "load_model"):
            E.load_model("cmt")
    except Exception as e:
        log.warning("[warm] load failed (continuing, will retry if needed): %s", e)

    dets, timings, metrics = E.run_inference(img_bytes, variant="cmt", analysis_id=f"vid_{aid}")

    # 4) Optional artifacts
    assets = {}
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        assets = _save_artifacts(aid, wid, pil, dets)
    except Exception as e:
        log.warning("artifact_io: %s", e)

    # 5) Normalize payload (FBL colors already list)
    colors_fbl = dets.get("colors") or []
    payload = {
        "id": str(uuid.uuid4()),
        "analysis_id": aid,
        "variant": "main",
        "type_label": dets.get("type"),
        "type_conf": dets.get("type_conf"),
        "make_label": dets.get("make"),
        "make_conf": dets.get("make_conf"),
        "model_label": dets.get("model"),
        "model_conf": dets.get("model_conf"),
        "plate_text": dets.get("plate_text"),
        "plate_conf": dets.get("plate_conf"),
        "colors": json.dumps(colors_fbl[:3]),
        "assets": json.dumps(assets),
        "latency_ms": int((metrics.get("latency_ms") or timings.get("total") or 0.0)),
        "memory_gb": metrics.get("mem_gb") or metrics.get("memory_gb"),
        "status": "done",
        "error_msg": None,
    }

    # 6) UPSERT results and mark parent done
    with _connect() as conn:
        conn.autocommit = True
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(UPSERT_SQL, payload)
        _mark_status(aid, "done")

    log.info("[upsert] aid=%s variant=main type=%s make=%s model=%s | %s",
             aid, payload["type_label"], payload["make_label"], payload["model_label"], config_summary())

# ---- daemon loop ------------------------------------------------

def main():
    log.info("[boot] video main-model worker starting… region=%s bucket=%s queue=%s | %s",
             REGION, BUCKET, Q_SNAPSHOT, config_summary())
    _ensure_tables()

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
            body = json.loads(m["Body"]) if isinstance(m.get("Body"), str) else m.get("Body", {})
            try:
                _process_one(body)
            except Exception as e:
                log.error("[error] process failed: %s", e)
                log.debug(traceback.format_exc())
                sqs.change_message_visibility(QueueUrl=Q_SNAPSHOT, ReceiptHandle=receipt, VisibilityTimeout=10)
                continue
            sqs.delete_message(QueueUrl=Q_SNAPSHOT, ReceiptHandle=receipt)
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
