import os, json, cv2, boto3, traceback, time, threading, math, re, argparse
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
from collections import defaultdict

# Optional CPU override
if os.getenv("FORCE_CPU","0") == "1":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["ULTRALYTICS_DEVICE"] = "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [worker_video] %(message)s")

# Load API .env for shared knobs
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "api", ".env"))

AWS_REGION             = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET              = os.getenv("S3_BUCKET") or os.getenv("BUCKET")
SQS_VIDEO_QUEUE_URL    = os.getenv("SQS_VIDEO_QUEUE_URL")
SQS_SNAPSHOT_QUEUE_URL = os.getenv("SQS_SNAPSHOT_QUEUE_URL")
DB_URL                 = os.getenv("DB_URL")
FRAME_STRIDE           = int(os.getenv("FRAME_STRIDE", "3"))

# Progress/format/heartbeat
PROG_INT   = int(os.getenv("PROGRESS_LOG_INTERVAL_SEC", "10"))
PROG_JSON  = bool(int(os.getenv("PROGRESS_LOG_JSON", "0")))
VIS_HEARTBEAT_SEC = int(os.getenv("SQS_VIS_HEARTBEAT_SEC", "0"))
VIS_TIMEOUT       = int(os.getenv("SQS_VIS_TIMEOUT", "300"))
DB_STATUS_HEARTBEAT = bool(int(os.getenv("DB_STATUS_HEARTBEAT","0")))

# DB normalize
if DB_URL and DB_URL.startswith("postgresql+psycopg2://"):
    DB_URL = DB_URL.replace("postgresql+psycopg2://","postgresql://",1)

sqs = boto3.client("sqs", region_name=AWS_REGION)
s3  = boto3.client("s3",  region_name=AWS_REGION)
engine = create_engine(DB_URL, pool_pre_ping=True, future=True) if DB_URL else None

from workers.yolo import YoloDetector, CLASS_NAMES
from workers.tracker import DeepSortTracker
from workers.snapshotper import (
    BestFrameKeeper, build_snapshot_key, save_and_upload_snapshot, choose_margin_for_neighbors
)
from workers.timecode import ms_from_frame, iso_add_ms

def utcnow_iso(): return datetime.now(timezone.utc).isoformat()

def set_status(video_id: str, status: str, err: str | None = None):
    if not engine: return
    with engine.begin() as conn:
        if err:
            conn.execute(text("UPDATE videos SET status=:s, updated_at=NOW(), error_msg=:e WHERE id=:id"),
                         {"s": status, "e": err[:500], "id": video_id})
        else:
            conn.execute(text("UPDATE videos SET status=:s, updated_at=NOW(), error_msg=NULL WHERE id=:id"),
                         {"s": status, "id": video_id})

def emit_snapshot_ready(payload: dict):
    sqs.send_message(QueueUrl=SQS_SNAPSHOT_QUEUE_URL, MessageBody=json.dumps(payload))

def _log_progress(pass_id:int, payload:dict, video_id: str | None = None):
    if DB_STATUS_HEARTBEAT and video_id:
        try: set_status(video_id, "processing")
        except: pass
    if PROG_JSON:
        logging.info("[progress] " + json.dumps({"pass":pass_id, **payload}))
    else:
        parts = [f"[progress] pass={pass_id}"] + [f"{k}={v}" for k,v in payload.items()]
        logging.info(" ".join(parts))

def _final_summary(summary: dict):
    if PROG_JSON:
        logging.info("[summary] " + json.dumps(summary))
    else:
        logging.info("[summary] " + ", ".join([f"{k}={v}" for k,v in summary.items()]))

def _compute_sharpness(frame_bgr, tlbr):
    # Laplacian variance on a downscaled grayscale crop (fast, robust)
    x1,y1,x2,y2 = map(int, tlbr)
    x1=max(0,x1); y1=max(0,y1); x2=max(x1+1,x2); y2=max(y1+1,y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0: return 0.0
    h,w = crop.shape[:2]
    # Normalize size for fair variance comparison
    scale = 128.0 / max(1.0, float(min(h,w)))
    if scale < 1.0:
        crop = cv2.resize(crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var()) if lap.size else 0.0
    return var

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
    # Ensure previous temp is gone
    try:
        if os.path.exists(local): os.remove(local)
    except: pass

    # Download
    s3.download_file(S3_BUCKET, s3_key_raw, local)

    det = YoloDetector()
    trk = DeepSortTracker()

    cap = cv2.VideoCapture(local)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0: fps = 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    iters_total = int(math.ceil((total_frames or 0) / max(1, frame_stride))) if total_frames else 0

    best = BestFrameKeeper(W, H)
    frame_to_boxes: dict[int, list[tuple[int, tuple[int,int,int,int]]]] = defaultdict(list)

    # Metrics
    m_frames_read = 0
    m_iters = 0
    m_dets_total = 0
    m_tracks_total = 0
    t1_start = time.time()
    last_log = t1_start

    # -------- Pass 1 --------
    frame_idx = -1
    while True:
        try:
            ok, frame = cap.read()
            if not ok: break
            m_frames_read += 1
            frame_idx += 1
            if frame_idx % frame_stride != 0: continue

            m_iters += 1
            dets = det.infer(frame)
            m_dets_total += len(dets)
            tracks = trk.update(dets, frame)

            # record boxes for neighbor suppression later
            for t in tracks:
                tlbr = tuple(map(int, t["tlbr"]))
                frame_to_boxes[frame_idx].append((int(t["id"]), tlbr))

            # per-track consider with sharpness estimate
            for t in tracks:
                sharp = _compute_sharpness(frame, t["tlbr"])
                best.consider(tid=t["id"], frame_idx=frame_idx, bbox=t["tlbr"], conf=t["conf"], cls_id=t["cls"], sharp=sharp)
            m_tracks_total += len(tracks)

            now = time.time()
            if now - last_log >= PROG_INT:
                elapsed = now - t1_start
                iter_sec = round(m_iters / elapsed, 2) if elapsed > 0 else 0.0
                fps_approx = round((m_iters * frame_stride) / elapsed, 2) if elapsed > 0 else 0.0
                eta = 0
                if iters_total and iter_sec > 0:
                    eta = int(round((iters_total - m_iters) / max(iter_sec, 1e-6)))
                _log_progress(1, dict(iters=m_iters, iters_total=iters_total, iter_sec=iter_sec,
                                      fps_approx=fps_approx, dets_last=len(dets), ETA=f"{eta}s"),
                              video_id=vid)
                last_log = now
        except Exception as e:
            logging.warning("pass1 frame=%d error: %s", frame_idx, e)
            continue

    cap.release()

    # -------- Pass 2 --------
    ready = best.items_ready()
    total_ready = len(ready)
    emit_count = 0
    t2_start = time.time()
    last_log2 = t2_start

    if total_ready == 0:
        set_status(vid, "done")
        _final_summary(dict(video_id=vid, frames=m_frames_read, iters=m_iters, dets=m_dets_total,
                            tracks=m_tracks_total, ready=0, emitted=0, pass1_sec=int(time.time()-t1_start),
                            pass2_sec=0, total_sec=int(time.time()-t1_start)))
        try: os.remove(local)
        except: pass
        return

    cap = cv2.VideoCapture(local)
    for tid, rec in ready:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, rec.frame_idx)
            ok, frame = cap.read()
            if not ok: continue

            neighbors = [tlbr for (other_tid, tlbr) in frame_to_boxes.get(rec.frame_idx, []) if other_tid != tid]
            margin = choose_margin_for_neighbors(rec.bbox, neighbors, W, H)

            offset_ms = ms_from_frame(rec.frame_idx, fps)
            key_rel = build_snapshot_key(
                user_id="demo_user",
                workspace_id=ws_id, video_id=vid,
                workspace_code=ws_code, camera_code=cam_code,
                track_id=tid, offset_ms=offset_ms
            )
            save_and_upload_snapshot(frame, rec.bbox, s3, S3_BUCKET, key_rel, margin=margin)

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
            emit_count += 1

            now = time.time()
            if now - last_log2 >= PROG_INT:
                elapsed = now - t2_start
                snaps_sec = round(emit_count / elapsed, 2) if elapsed > 0 else 0.0
                _log_progress(2, dict(emitted=emit_count, total=total_ready, snapshots_sec=snaps_sec,
                                      elapsed=f"{int(elapsed)}s"), video_id=vid)
                last_log2 = now
        except Exception as e:
            logging.warning("pass2 tid=%s error: %s", str(tid), e)
            continue

    cap.release()
    set_status(vid, "done")

    total_sec = int(time.time() - t1_start)
    pass2_sec = int(time.time() - t2_start)
    _final_summary(dict(video_id=vid, frames=m_frames_read, iters=m_iters, dets=m_dets_total,
                        tracks=m_tracks_total, ready=total_ready, emitted=emit_count,
                        pass1_sec=total_sec-pass2_sec, pass2_sec=pass2_sec, total_sec=total_sec))

    try: os.remove(local)
    except: pass

def _hb_loop(stop_evt: threading.Event, rh: str):
    while not stop_evt.wait(timeout=max(1, VIS_HEARTBEAT_SEC)):
        try:
            sqs.change_message_visibility(
                QueueUrl=SQS_VIDEO_QUEUE_URL,
                ReceiptHandle=rh,
                VisibilityTimeout=VIS_TIMEOUT
            )
            logging.info("[heartbeat] extended visibility to %ss", VIS_TIMEOUT)
        except Exception as e:
            logging.warning("[heartbeat] failed: %s", e)

def poll_loop():
    logging.info("polling: %s at %s", SQS_VIDEO_QUEUE_URL, utcnow_iso())
    while True:
        rs = sqs.receive_message(
            QueueUrl=SQS_VIDEO_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=VIS_TIMEOUT if VIS_HEARTBEAT_SEC <= 0 else max(VIS_TIMEOUT, 120)
        )
        for m in rs.get("Messages", []):
            rh = m["ReceiptHandle"]
            payload = None
            stop_evt = threading.Event()
            hb_thread = None
            try:
                body = json.loads(m["Body"])
                payload = body if isinstance(body, dict) else json.loads(body)
                if payload.get("event") != "PROCESS_VIDEO":
                    sqs.delete_message(QueueUrl=SQS_VIDEO_QUEUE_URL, ReceiptHandle=rh)
                    continue
                if VIS_HEARTBEAT_SEC > 0:
                    hb_thread = threading.Thread(target=_hb_loop, args=(stop_evt, rh), daemon=True)
                    hb_thread.start()
                handle_process_video(payload)
                sqs.delete_message(QueueUrl=SQS_VIDEO_QUEUE_URL, ReceiptHandle=rh)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                logging.error("ERROR: %s", err)
                logging.error(traceback.format_exc())
                if payload and isinstance(payload, dict):
                    try: set_status(payload.get("video_id","00000000-0000-0000-0000-000000000000"), "error", err=err)
                    except: pass
            finally:
                if hb_thread:
                    stop_evt.set()
                    hb_thread.join(timeout=3)

def main():
    ap = argparse.ArgumentParser(description="CVITX Modular Video Worker")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--poll", action="store_true", help="Poll SQS for PROCESS_VIDEO messages")
    g.add_argument("--process-payload", type=str, help="Inline JSON payload for PROCESS_VIDEO")
    g.add_argument("--process-payload-file", type=str, help="Path to JSON payload for PROCESS_VIDEO")
    args = ap.parse_args()

    if args.poll:
        poll_loop(); return
    if args.process_payload:
        payload = json.loads(args.process_payload); handle_process_video(payload); return
    if args.process_payload_file:
        with open(args.process_payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f); handle_process_video(payload); return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("bye")
