import os, io, json, time, uuid, logging, traceback
import numpy as np
from PIL import Image
import boto3

from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import engine
from app.s3util import split_s3, get_bytes, put_png_bytes

# ---- Env & clients
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
AWS_REGION = os.environ["AWS_REGION"]
S3_BUCKET = os.environ["S3_BUCKET"]
SQS_SNAPSHOT_QUEUE_URL = os.environ["SQS_SNAPSHOT_QUEUE_URL"]

CONF_MAKE  = float(os.environ.get("CONF_MAKE", "0.70"))
CONF_MODEL = float(os.environ.get("CONF_MODEL", "0.70"))
PLATE_W    = int(os.environ.get("PLATE_WIDTH", "640"))
PLATE_H    = int(os.environ.get("PLATE_HEIGHT", "300"))
COLOR_SECONDARY_MIN = float(os.environ.get("COLOR_SECONDARY_MIN_FRACTION", "0.20"))

sqs = boto3.client("sqs", region_name=AWS_REGION)

# ---- DB reflection (we already created the table via SQL migration)
md = MetaData()
detections = Table("detections", md, autoload_with=engine)

# ---- Color palette (simple RGB nearest)
PALETTE = {
    "WHITE":  np.array([245,245,245], dtype=np.float32),
    "BLACK":  np.array([ 10, 10, 10], dtype=np.float32),
    "SILVER": np.array([192,192,192], dtype=np.float32),
    "GRAY":   np.array([128,128,128], dtype=np.float32),
    "RED":    np.array([200, 30, 30], dtype=np.float32),
    "BLUE":   np.array([ 30, 60,200], dtype=np.float32),
    "GREEN":  np.array([ 30,160, 60], dtype=np.float32),
    "YELLOW": np.array([240,220, 70], dtype=np.float32),
    "ORANGE": np.array([230,140, 40], dtype=np.float32),
    "BROWN":  np.array([120, 80, 50], dtype=np.float32),
}
PALETTE_KEYS = np.stack(list(PALETTE.values()))
PALETTE_NAMES = list(PALETTE.keys())

def map_to_palette(color_vec):
    d = ((PALETTE_KEYS - color_vec) ** 2).sum(axis=1)
    return PALETTE_NAMES[int(np.argmin(d))]

def dominant_palette_labels(img_rgb: np.ndarray, min_second_fraction=0.20):
    # downscale for speed
    im = Image.fromarray(img_rgb).resize((256,256))
    arr = np.asarray(im).reshape(-1,3).astype(np.float32)

    # naive 2-means initialization
    mean = arr.mean(axis=0); d = ((arr - mean) ** 2).sum(axis=1)
    i = int(np.argmax(d)); center1 = arr[i]
    d2 = ((arr - center1) ** 2).sum(axis=1)
    j = int(np.argmax(d2)); center2 = arr[j]
    for _ in range(6):
        dist1 = ((arr - center1) ** 2).sum(axis=1)
        dist2 = ((arr - center2) ** 2).sum(axis=1)
        mask = dist1 < dist2
        if mask.sum() == 0 or mask.sum() == arr.shape[0]:
            break
        center1 = arr[mask].mean(axis=0)
        center2 = arr[~mask].mean(axis=0)
    n1 = int(mask.sum()); n2 = int((~mask).sum())
    fr1 = n1 / (n1 + n2) if (n1+n2) else 1.0
    fr2 = 1.0 - fr1
    labels = [map_to_palette(center1)]
    if fr2 >= min_second_fraction:
        labels.append(map_to_palette(center2))
    return labels

# ---- Inference stubs (replace later with real models)
def run_mobilevit(img_rgb: np.ndarray):
    # TODO: load torch model once and return real predictions
    return "SUV", 0.97, "Toyota", 0.94, "Fortuner", 0.91, [
        {"name":"Front_Grille","conf":0.93},
        {"name":"Left_Headlight","conf":0.90},
    ]

def detect_plate_bbox(img_rgb: np.ndarray):
    # stub: central lower band-ish
    H, W, _ = img_rgb.shape
    box_w, box_h = int(W * 0.5), int(H * 0.18)
    x1 = (W - box_w) // 2; y1 = int(H * 0.62)
    x2 = x1 + box_w;       y2 = min(y1 + box_h, H-1)
    return [x1, y1, x2, y2]

def run_ocr_uppercase(png_bytes: bytes):
    # TODO: plug PlateRecognizer or other OCR
    return None, "failed"

def crop_png(img_rgb: np.ndarray, bbox, size_hw):
    (w,h) = (size_hw[1], size_hw[0])
    x1,y1,x2,y2 = map(int, bbox)
    H,W,_ = img_rgb.shape
    x1 = max(0,x1); y1 = max(0,y1); x2 = min(W-1,x2); y2 = min(H-1,y2)
    crop = Image.fromarray(img_rgb[y1:y2, x1:x2])
    crop = crop.resize((w,h))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()

# ---- UPSERT helper
def upsert_detection(row: dict):
    ins = pg_insert(detections).values(**row)
    # Do not update id on conflict
    update_cols = {k: ins.excluded[k] for k in row.keys() if k != "id"}
    stmt = ins.on_conflict_do_update(
        index_elements=["video_id", "track_id"],
        set_=update_cols
    )
    with engine.begin() as conn:
        conn.execute(stmt)

# ---- Core handler
def handle_snapshot_ready(msg: dict):
    vid = uuid.UUID(msg["video_id"])
    wsid = uuid.UUID(msg["workspace_id"])
    ws_code = msg["workspace_code"]
    cam_code = msg["camera_code"]
    track_id = int(msg["track_id"])
    offset_ms = int(msg["detectedIn"])
    display_id = f"{ws_code}-{cam_code}-{track_id}"

    # 1) Load snapshot (message carries s3://… per Day-2)
    b = get_bytes(msg["snapshot_s3_key"])
    img = Image.open(io.BytesIO(b)).convert("RGB")
    img_np = np.asarray(img)

    # 2) MobileViT (stub)
    type_, type_conf, make, make_conf, model, model_conf, parts = run_mobilevit(img_np)
    abstain = False; abstain_level = None; reason = None; THRESHOLDS = {"make_model_min_conf": min(CONF_MAKE, CONF_MODEL)}
    if model_conf is None or model_conf < CONF_MODEL:
        model = None; abstain = True; abstain_level = "model"; reason = "low_confidence"
    if make_conf is None or make_conf < CONF_MAKE:
        make = None; model = None; abstain = True; abstain_level = "make"; reason = "low_confidence"

    # 3) Plate detect → crop (640x300) → upload → OCR
    bbox = detect_plate_bbox(img_np)
    if bbox:
        plate_png = crop_png(img_np, bbox, (PLATE_H, PLATE_W))
        # NOTE: keep Day-2 path tree + zero padding on ids/timestamps
        # We receive track_id already as int; it's already padded in the filename for snapshots
        # For consistency, we will not re-pad here; filenames in /plates can follow the same zero padding
        # by formatting below.
        track_str = f"{track_id:06d}"
        offset_str = f"{offset_ms:06d}"
        plate_key_rel = f"demo_user/{wsid}/{vid}/plates/{ws_code}_{cam_code}_{track_str}_{offset_str}.png"
        plate_s3_url = put_png_bytes(plate_key_rel, plate_png)
        _, plate_key_only = split_s3(plate_s3_url)
        ocr_pre = ["upscale","grayscale","contrast","binarize"]
        plate_text, ocr_status = run_ocr_uppercase(plate_png)
        plate_text = plate_text.upper() if plate_text else None
    else:
        plate_s3_url = None
        ocr_pre = []
        plate_text, ocr_status = None, "skipped"

    # 4) Colors
    colors = dominant_palette_labels(img_np, min_second_fraction=COLOR_SECONDARY_MIN)

    # 5) Normalize snapshot_s3_key to KEY-ONLY for DB storage
    _, snap_key = split_s3(msg["snapshot_s3_key"])

    row = {
        "id": uuid.uuid4(),
        "display_id": display_id,
        "video_id": vid,
        "workspace_id": wsid,
        "track_id": track_id,
        "snapshot_s3_key": snap_key,
        "plate_image_s3_key": plate_key_only,  # may be None or s3://...
        "recorded_at": msg["recordedAt"],
        "detected_in_ms": offset_ms,
        "detected_at": msg["detectedAt"],

        "yolo_type": msg.get("yolo_type"),
        "type": type_, "type_conf": float(type_conf or 0.0),
        "make": make,  "make_conf": float(make_conf or 0.0),
        "model": model,"model_conf": float(model_conf or 0.0),

        "parts": parts,
        "colors": colors,
        "plate_text": plate_text,

        "abstain": abstain,
        "abstain_level": abstain_level,
        "abstain_reason": reason,
    "thresholds": THRESHOLDS,
        "thresholds": {"make": CONF_MAKE, "model": CONF_MODEL},
        "evidence": {
            "top_parts": [p["name"] for p in (parts or [])[:2]],
            "weights": {p["name"]: round(float(p["conf"]), 3) for p in (parts or [])},
            "ocr": {"provider":"PlateRecognizer","preprocessing":ocr_pre,"status":ocr_status}
        },
        "status": "ready"
    }
    upsert_detection(row)
    logging.info("[worker2] upserted detection: %s (%s)", display_id, vid)

def run():
    logging.info("[worker2] polling: %s", SQS_SNAPSHOT_QUEUE_URL)
    while True:
        rs = sqs.receive_message(
            QueueUrl=SQS_SNAPSHOT_QUEUE_URL,
            MaxNumberOfMessages=5,
            WaitTimeSeconds=20,
            VisibilityTimeout=120
        )
        msgs = rs.get("Messages", [])
        if not msgs:
            continue
        for m in msgs:
            rh = m["ReceiptHandle"]
            try:
                body = m["Body"]
                payload = json.loads(body) if isinstance(body, str) else body
                if isinstance(payload, dict) and payload.get("event") == "SNAPSHOT_READY":
                    handle_snapshot_ready(payload)
                # delete only after successful DB commit
                sqs.delete_message(QueueUrl=SQS_SNAPSHOT_QUEUE_URL, ReceiptHandle=rh)
            except Exception as e:
                logging.error("[worker2] ERROR: %s\n%s", e, traceback.format_exc())
                # leave message for retry / DLQ
                continue

if __name__ == "__main__":
    run()
