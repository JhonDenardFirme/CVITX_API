CVITX · Video Analysis Monorepo (YOLO → SNAPSHOT_READY → Main-Model)
Source of truth. This repo contains exactly two workers wired end-to-end:
yolo_worker — consumes PROCESS_VIDEO, reads the raw MP4 from S3, runs YOLO+DeepSORT, writes 640×640 JPEG snapshots to S3, and emits SNAPSHOT_READY (one per tracked vehicle).


main_worker — consumes SNAPSHOT_READY, fetches the snapshot, runs your Main Model (Type→Make→Model; optional Color/Plate, and optionally Baseline vs CMT), and writes results into Postgres.


Everything below is deterministic and matches the unified setup roadmap you approved.

0) What you get (1-page quickstart)
Install (inside your API venv)
cd /home/ubuntu/cvitx/video_analysis
. /home/ubuntu/cvitx/api/.venv/bin/activate
pip install -r requirements.txt

Run locally (one terminal per worker)
# YOLO → snapshots + SNAPSHOT_READY
python -m video_analysis.yolo_worker.run_poller

# Main-Model → inference + DB writes
python -m video_analysis.main_worker.run_poller

Systemd (production)
sudo systemctl daemon-reload
sudo systemctl enable --now cvitx-yolo-video cvitx-main-model
journalctl -u cvitx-yolo-video -f
journalctl -u cvitx-main-model -f

Deterministic constants (no .env required by default)
All constants live in worker_config.py in one place. Default mode is hard-coded so nothing depends on missing env files. You can flip a single switch there if you want optional env overrides later.

1) End-to-end diagram (exact hand-offs)
Client ──(Presign PUT)──▶ S3: /raw/<file>.mp4
   │
   └─▶ API: register video (DB: videos.status='uploaded')
       └─▶ SQS: cvitx-video-tasks  (message: PROCESS_VIDEO)
            │
            ▼
        YOLO Worker
        - Read raw MP4 from S3
        - YOLO+DeepSORT track
        - Best frame per track
        - Crop+letterbox → 640×640 JPEG
        - Write snapshot → S3: /snapshots/CTX####_CAM#_{track6}_{offset6}.jpg
        - Emit SNAPSHOT_READY → SQS: cvitx-snapshot-tasks
            │
            ▼
        Main-Model Worker
        - Fetch snapshot from S3
        - Run Main Model (Type/Make/Model; optional Color/Plate; optional Baseline vs CMT)
        - Write DB:
            image_analyses (parent, 1 per snapshot)
            image_analysis_results (child, 1 or 2 rows per snapshot)
            assets S3 keys only (URLs are presigned by API on demand)
            │
            ▼
        Frontend (UI)
        - Poll video status & analysis via API
        - API presigns GET for assets as needed


2) Determinism & invariants (locked)
Region: ap-southeast-2


Bucket: cvitx-uploads-dev-jdfirme


Queues:


Video tasks (inbound to YOLO): cvitx-video-tasks (+ DLQ)


Snapshot tasks (inbound to Main-Model): cvitx-snapshot-tasks (+ DLQ)


Snapshot file: JPEG, exactly 640×640, Content-Type: image/jpeg


Snapshot URI pattern (events carry full s3:// URI):
 s3://cvitx-uploads-dev-jdfirme/demo_user/<wid>/<vid>/snapshots/CTX####_CAM#_{track:06d}_{offsetMs:06d}.jpg


YOLO type taxonomy (8 classes): Car | SUV | Van | LightTruck | Utility | Motorcycle | CarouselBus | E-Jeepney


Status machines:


videos.status: uploaded → queued → processing → done|error


image_analyses.status: queued → processing → done|error


DB uniqueness:


image_analyses.snapshot_s3_key UNIQUE


image_analysis_results UNIQUE(analysis_id, variant)



3) Message contracts (copy-paste exact)
3.1 Inbound to YOLO — PROCESS_VIDEO (primary, full payload)
{
  "event": "PROCESS_VIDEO",
  "video_id": "50651c85-90ab-471b-8707-adc0ef16f91f",
  "workspace_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
  "workspace_code": "CTX1005",
  "camera_code": "CAM1",
  "s3_key_raw": "demo_user/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/50651c85-90ab-471b-8707-adc0ef16f91f/raw/CCTV_Test1.mp4",
  "frame_stride": 3,
  "recordedAt": null
}

Rules: frame_stride ≥ 1; s3_key_raw is a key (bucket is implicit).
(Compat) Minimal inbound — PROCESS_VIDEO_DB (DB-resolve fallback)
{ "event": "PROCESS_VIDEO_DB", "video_id": "UUID", "workspace_id": "UUID" }

YOLO worker will query DB to resolve s3_key_raw, workspace_code, camera_code, frame_stride, recordedAt.

3.2 Outbound from YOLO — SNAPSHOT_READY (to Main-Model)
{
  "event": "SNAPSHOT_READY",
  "video_id": "50651c85-90ab-471b-8707-adc0ef16f91f",
  "workspace_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
  "workspace_code": "CTX1005",
  "camera_code": "CAM1",
  "track_id": 305,
  "snapshot_s3_key": "s3://cvitx-uploads-dev-jdfirme/demo_user/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/50651c85-90ab-471b-8707-adc0ef16f91f/snapshots/CTX1005_CAM1_000305_002272.jpg",
  "recordedAt": null,
  "detectedIn": 2272,
  "detectedAt": null,
  "yolo_type": "SUV"
}

Invariants: snapshot exists; 640×640 JPEG; deterministic name (track_id & offsetMs are 6-digit, zero-padded).

4) Repo layout (minimal, centralized)
/home/ubuntu/cvitx/video_analysis/
├─ README.md                      ← you are here
├─ requirements.txt               ← pinned wheels (reproducible installs)
├─ worker_config.py               ← ONE place for constants (hard-coded by default)
├─ worker_utils/
│  └─ common.py                   ← logging, validators, S3/SQS/DB clients, imaging, key builder, time utils
├─ yolo_worker/
│  ├─ __init__.py
│  ├─ worker.py                   ← PROCESS_VIDEO → snapshots + SNAPSHOT_READY
│  ├─ run_poller.py               ← entrypoint: python -m video_analysis.yolo_worker.run_poller
│  └─ bundle/
│     ├─ yolov8m.pt               ← detector
│     └─ deepsort.engine          ← tracker
├─ main_worker/
│  ├─ __init__.py
│  ├─ worker.py                   ← SNAPSHOT_READY → inference → DB writes
│  ├─ run_poller.py               ← entrypoint: python -m video_analysis.main_worker.run_poller
│  └─ bundle/
│     ├─ main.pt                  ← your single main model OR baseline.pt
│     └─ cmt.pt                   ← optional if comparing Baseline vs CMT
└─ schemas/
   ├─ process_video.json          ← JSON Schema (primary inbound)
   └─ snapshot_ready.json         ← JSON Schema (outbound to Main-Model)


5) Shapes & sizes (so no one gets lost)
YOLO detections (per frame): D × 6 → (x1, y1, x2, y2, conf, cls) float32; D varies [0..N].


Track selection: one best frame per track by priority: area → conf → sharpness → center → earlier.


Snapshot image: 640×640×3 (uint8), JPEG (≈30–200 KB typical), Content-Type: image/jpeg.


Model input: [1, 3, 640, 640] (float32; normalized per backbone).


Color output (FBL array): [{"finish": null|"Matte"|"Metallic", "base": "White", "lightness": "Light"|null, "conf": 0.0–1.0}] (1–3 items).


Results row (per variant): ~2–4 KB JSON (excluding images).



6) ⚠️ AWS SETUP REQUIRED (once per environment)
Create these before starting workers. Keep IAM least-privilege.
S3 Bucket: cvitx-uploads-dev-jdfirme


Folders are logical (keys), no need to create directories.


Lifecycle: (recommended) expire/tier /snapshots/ per retention.


SQS Queues (Standard):


cvitx-video-tasks (+ DLQ) — Inbound to YOLO


cvitx-snapshot-tasks (+ DLQ) — Inbound to Main-Model


Attributes (recommend): VisibilityTimeout=300, MessageRetention=1209600 (14d), ReceiveMessageWaitTimeSeconds=10, DLQ maxReceiveCount=5.


IAM policies (least privilege):


YOLO worker


s3:GetObject on arn:aws:s3:::cvitx-uploads-dev-jdfirme/demo_user/*/*/raw/*


s3:PutObject on arn:aws:s3:::cvitx-uploads-dev-jdfirme/demo_user/*/*/snapshots/*


sqs:ReceiveMessage/DeleteMessage/ChangeMessageVisibility/GetQueueAttributes on cvitx-video-tasks


sqs:SendMessage on cvitx-snapshot-tasks


Main-Model worker


s3:GetObject on /snapshots/*


sqs:ReceiveMessage/DeleteMessage/ChangeMessageVisibility/GetQueueAttributes on cvitx-snapshot-tasks


API


s3:PutObject presign scope to /raw/*


sqs:SendMessage to cvitx-video-tasks



7) Quick smoke tests
7.1 Upload & enqueue (through API, typical)
Presign PUT, upload MP4 to
 demo_user/<wid>/<vid>/raw/CCTV_Test1.mp4


Enqueue via API: it should push PROCESS_VIDEO to cvitx-video-tasks.


7.2 Direct SQS test (bypass API; for ops only)
Send a PROCESS_VIDEO JSON (above) to cvitx-video-tasks.
 YOLO worker should produce snapshots at /snapshots/... and emit SNAPSHOT_READY to cvitx-snapshot-tasks.

8) Troubleshooting (most common → fix)
No snapshots appear in S3


Check YOLO logs for MP4 read errors or empty frames. Verify s3_key_raw matches raw key regex.


Ensure queue visibility: worker heartbeats every 60s; VisibilityTimeout ≥ actual processing time.


SNAPSHOT_READY not arriving


Confirm YOLO has SendMessage to cvitx-snapshot-tasks.


Validate message shape with schemas/snapshot_ready.json (fail fast).


Main-Model writes nothing


Check snapshot exists and is 640×640 JPEG; wrong size/content-type will be rejected.


Ensure DB URL is reachable; look for UNIQUE(analysis_id, variant) collisions (then it’s a duplicate write).


DLQ filling up


Inspect DLQ bodies; typical causes: schema mismatch, missing S3 object, wrong key pattern, or timeouts.


Fix root cause, then re-drive DLQ messages.


UI shows no assets


API must presign GET (workers store keys only, never URLs). Verify presign route and TTL.



9) Running modes (single vs compare)
Single model (default): Main-Model writes one result row per snapshot (you may label it variant='baseline' for consistency).


Baseline vs CMT (optional): Main-Model loads two checkpoints and writes two rows per snapshot: variant in ('baseline','cmt').


Switch by editing a single constant in worker_config.py (kept centralized).

10) Requirements (pinned)
boto3==1.34.*
psycopg2-binary==2.9.*
pydantic==2.9.*
opencv-python-headless==4.10.*
numpy==1.26.*
ultralytics==8.3.*
lapx==0.5.*           # DeepSORT deps
scikit-image==0.24.*


11) Regex quick reference
UUID: [0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}


Raw key (KEY only):
 ^demo_user/[0-9a-f-]{36}/[0-9a-f-]{36}/raw/.+\.mp4$


Snapshot URI (full):
 ^s3://cvitx-uploads-dev-jdfirme/demo_user/[0-9a-f-]{36}/[0-9a-f-]{36}/snapshots/CTX\d{4,}_CAM\d+_\d{6}_\d{6}\.jpg$



12) Why this README is “deterministic & simple”
Single place for knobs: worker_config.py hard-codes all constants by default (no .env needed); both workers import from it.


Minimal files: one shared common.py, one worker.py per worker, two tiny runners, and two JSON Schemas.


Stable contracts: exact message shapes + frozen S3/SQS/DB patterns prevent drift.


Reload-safe: S3 + SQS + DB durability; UI just polls.



Appendix A — Systemd units (copy/paste)
/etc/systemd/system/cvitx-yolo-video.service
[Unit]
Description=CVITX YOLO Video Worker (PROCESS_VIDEO → SNAPSHOT_READY)
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/cvitx
EnvironmentFile=/home/ubuntu/cvitx/api/.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/home/ubuntu/cvitx
ExecStart=/home/ubuntu/cvitx/api/.venv/bin/python -m video_analysis.yolo_worker.run_poller
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target

/etc/systemd/system/cvitx-main-model.service
[Unit]
Description=CVITX Main-Model Worker (SNAPSHOT_READY → inference → DB)
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/cvitx
EnvironmentFile=/home/ubuntu/cvitx/api/.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/home/ubuntu/cvitx
ExecStart=/home/ubuntu/cvitx/api/.venv/bin/python -m video_analysis.main_worker.run_poller
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target

Even though systemd refers to an .env, your code does not depend on it by default. worker_config.py uses hard-coded constants unless you deliberately enable env overrides (a one-line toggle in that file).

Done. This README is your single ramp for onboarding: install, run, contracts, shapes, AWS checklist, failure modes, and exact wiring—all matching the approved setup plan.
