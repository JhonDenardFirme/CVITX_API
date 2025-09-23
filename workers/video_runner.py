import os, json, cv2, boto3, traceback
import re, boto3
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ["ULTRALYTICS_DEVICE"] = "cpu"

S3 = boto3.client("s3", region_name=os.getenv("AWS_REGION","ap-southeast-2"))

def _emit_existing_snapshots(payload: dict):
    """
    List s3://{S3_BUCKET}/demo_user/{workspace_id}/{video_id}/snapshots/
    and emit one SNAPSHOT_READY per object.
    """
    bucket = os.getenv("S3_BUCKET") or os.getenv("BUCKET")
    if not bucket:
        return
    ws_id = str(payload.get("workspace_id",""))
    vid = str(payload.get("video_id",""))
    ws_code = str(payload.get("workspace_code",""))
    cam_code_hint = str(payload.get("camera_code",""))
    recorded_at = payload.get("recordedAt")
    prefix = f"demo_user/{ws_id}/{vid}/snapshots/"
    paginator = S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key","")
            # Expect <ws_code>_<cam_code>_<track>_<offset>.jpg
            m = re.search(r'/([^/_]+)_([^/_]+)_(\d{6})_(\d{6})\.jpg$', key)
            track_id = int(m.group(3)) if m else 0
            offset_ms = int(m.group(4)) if m else None
            cam_code = cam_code_hint or (m.group(2) if m else "")
            msg = {
                "event": "SNAPSHOT_READY",
                "video_id": vid,
                "workspace_id": ws_id,
                "workspace_code": ws_code,
                "camera_code": cam_code,
                "track_id": track_id,
                "snapshot_s3_key": key,
                "recordedAt": recorded_at,
                "detectedIn": offset_ms,
                "detectedAt": iso_add_ms(recorded_at, offset_ms) if (recorded_at and offset_ms is not None) else None,
                "yolo_type": "VEHICLE"
            }
            emit_snapshot_ready(msg)
from sqlalchemy import create_engine, text
from datetime import datetime, timezone
from dotenv import load_dotenv

# load the API .env so both services share knobs
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "api", ".env"))

AWS_REGION             = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET              = os.getenv("S3_BUCKET")
SQS_VIDEO_QUEUE_URL    = os.getenv("SQS_VIDEO_QUEUE_URL")
SQS_SNAPSHOT_QUEUE_URL = os.getenv("SQS_SNAPSHOT_QUEUE_URL")
DB_URL                 = os.getenv("DB_URL")
FRAME_STRIDE           = int(os.getenv("FRAME_STRIDE","3"))

# psycopg2 DSN (strip +psycopg2 for SQLAlchemy URL if present)
if DB_URL and DB_URL.startswith("postgresql+psycopg2://"):
    DB_URL = DB_URL.replace("postgresql+psycopg2://","postgresql://",1)

sqs = boto3.client("sqs", region_name=AWS_REGION)
s3  = boto3.client("s3",  region_name=AWS_REGION)
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

from .yolo import YoloDetector, CLASS_NAMES
from .tracker import DeepSortTracker
from .snapshotper import BestFrameKeeper, build_snapshot_key, save_and_upload_snapshot
from .timecode import ms_from_frame, iso_add_ms

def utcnow_iso(): return datetime.now(timezone.utc).isoformat()

def set_status(video_id: str, status: str, err: str | None = None):
    with engine.begin() as conn:
        if err:
            conn.execute(
                text("UPDATE videos SET status=:s, updated_at=NOW(), error_msg=:e WHERE id=:id"),
                {"s": status, "e": err[:500], "id": video_id},
            )
        else:
            # Clear any previous error on success/progress
            conn.execute(
                text("UPDATE videos SET status=:s, updated_at=NOW(), error_msg=NULL WHERE id=:id"),
                {"s": status, "id": video_id},
            )

def download_to_tmp(s3_key: str, path: str):
    s3.download_file(S3_BUCKET, s3_key, path)
    return path

def emit_snapshot_ready(payload: dict):
    sqs.send_message(QueueUrl=SQS_SNAPSHOT_QUEUE_URL, MessageBody=json.dumps(payload))

def handle_process_video(body: dict):
    vid         = body["video_id"]
    ws_id       = body["workspace_id"]
    ws_code     = body["workspace_code"]
    cam_code    = body["camera_code"]
    s3_key_raw  = body["s3_key_raw"]
    frame_stride= int(body.get("frame_stride", FRAME_STRIDE))
    recorded_at = body.get("recordedAt")

    set_status(vid, "processing")

    local = f"/tmp/{vid}.mp4"
    download_to_tmp(s3_key_raw, local)

    det = YoloDetector()
    trk = DeepSortTracker()

    cap = cv2.VideoCapture(local)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0: fps = 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    best = BestFrameKeeper(W, H)

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        if frame_idx % frame_stride != 0: continue

        dets = det.infer(frame)
        tracks = trk.update(dets, frame)
        for t in tracks:
            best.consider(tid=t["id"], frame_idx=frame_idx, bbox=t["tlbr"], conf=t["conf"], cls_id=t["cls"])

    cap.release()
    cap = cv2.VideoCapture(local)

    for tid, rec in best.items_ready():
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec.frame_idx)
        ok, frame = cap.read()
        if not ok: continue

        offset_ms = ms_from_frame(rec.frame_idx, fps)
        key_rel = build_snapshot_key(
            user_id="demo_user",
            workspace_id=ws_id, video_id=vid,
            workspace_code=ws_code, camera_code=cam_code,
            track_id=tid, offset_ms=offset_ms
        )
        save_and_upload_snapshot(frame, rec.bbox, s3, S3_BUCKET, key_rel)

        # full s3:// URI in message (matches your contract)
        snapshot_uri = f"s3://{S3_BUCKET}/{key_rel}"
        yolo_type = CLASS_NAMES[rec.cls_id] if 0 <= rec.cls_id < len(CLASS_NAMES) else "VEHICLE"

        emit_snapshot_ready({
            "event": "SNAPSHOT_READY",
            "video_id": vid,
            "workspace_id": ws_id,
            "workspace_code": ws_code,
            "camera_code": cam_code,
            "track_id": tid,
            "snapshot_s3_key": snapshot_uri,
            "recordedAt": recorded_at,
            "detectedIn": offset_ms,
            "detectedAt": iso_add_ms(recorded_at, offset_ms) if recorded_at else None,
            "yolo_type": yolo_type
        })

    cap.release()
    set_status(vid, "done")

def run():
    print("[worker_video] polling:", SQS_VIDEO_QUEUE_URL, "at", utcnow_iso())
    while True:
        rs = sqs.receive_message(
            QueueUrl=SQS_VIDEO_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=300
        )
        for m in rs.get("Messages", []):
            rh = m["ReceiptHandle"]
            try:
                body = json.loads(m["Body"])
                payload = body if isinstance(body, dict) else json.loads(body)
                if payload.get("event") != "PROCESS_VIDEO":
                    sqs.delete_message(QueueUrl=SQS_VIDEO_QUEUE_URL, ReceiptHandle=rh)
                    continue
                handle_process_video(payload)
                sqs.delete_message(QueueUrl=SQS_VIDEO_QUEUE_URL, ReceiptHandle=rh)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print("[worker_video] ERROR:", err)
                print(traceback.format_exc())
                # leave message for retry / DLQ
                set_status(payload.get("video_id","00000000-0000-0000-0000-000000000000"), "error", err=err)

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("bye")
