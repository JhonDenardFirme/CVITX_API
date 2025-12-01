import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from app.auth.deps import require_user
from app.config import settings

# ─────────────────────────────────────────────────────────────
# DB & S3 wiring
# ─────────────────────────────────────────────────────────────

DB_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("Missing DATABASE_URL/DB_URL for video analyses")
DB_URL = DB_URL.replace("postgresql+psycopg2", "postgresql")

engine = create_engine(DB_URL, pool_pre_ping=True)

s3 = boto3.client("s3", region_name=settings.aws_region)

# Cap for presigned GET URLs for video analysis assets
ANALYSIS_TTL_DEFAULT = int(os.getenv("VIDEO_ANALYSIS_PRESIGN_TTL", "900"))

router = APIRouter(prefix="/workspaces", tags=["video-analyses"])


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _assert_workspace(conn, workspace_id: str, user_id: str) -> None:
    """
    Ensure the workspace exists and belongs to this user.
    """
    row = conn.execute(
        text(
            """
            SELECT 1
              FROM workspaces
             WHERE id = :wid
               AND owner_user_id = :uid
               AND deleted_at IS NULL
            """
        ),
        {"wid": workspace_id, "uid": user_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found")


def _compute_percent(processed: Optional[int], expected: Optional[int]) -> float:
    p = int(processed or 0)
    e = int(expected or 0)
    if e <= 0:
        return 0.0
    return round((p / e) * 100.0, 2)


def _presign_assets(assets: Dict[str, Any], ttl: int) -> Dict[str, Any]:
    """
    Given an assets JSONB blob from video_detections, attach presigned URLs.
    Expected keys in DB: annotated_image_s3_key, vehicle_image_s3_key, plate_image_s3_key.
    """
    if not assets:
        return {}

    key_annot = assets.get("annotated_image_s3_key")
    key_vehicle = assets.get("vehicle_image_s3_key")
    key_plate = assets.get("plate_image_s3_key")

    def _make_url(key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=ttl,
        )

    out: Dict[str, Any] = {
        "annotated_image_s3_key": key_annot,
        "vehicle_image_s3_key": key_vehicle,
        "plate_image_s3_key": key_plate,
    }
    if key_annot:
        out["annotatedUrl"] = _make_url(key_annot)
    if key_vehicle:
        out["vehicleUrl"] = _make_url(key_vehicle)
    if key_plate:
        out["plateUrl"] = _make_url(key_plate)
    return out


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────

class VideoRowOut(BaseModel):
    id: str
    workspaceId: str
    workspaceCode: Optional[str] = None
    fileName: str
    cameraLabel: Optional[str] = None
    cameraCode: str
    recordedAt: Optional[datetime] = None
    s3KeyRaw: str
    frameStride: int
    status: str
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    errorMsg: Optional[str] = None
    processingStartedAt: Optional[datetime] = None
    processingFinishedAt: Optional[datetime] = None


class VideoRunSummary(BaseModel):
    analysisId: str
    videoId: str
    variant: str
    status: str
    runId: Optional[str] = None
    expectedSnapshots: Optional[int] = None
    processedSnapshots: int
    processedOk: int
    processedErr: int
    runStartedAt: Optional[datetime] = None
    runFinishedAt: Optional[datetime] = None
    lastSnapshotAt: Optional[datetime] = None


class VideoDetailOut(BaseModel):
    video: VideoRowOut
    latestRun: Optional[VideoRunSummary] = None


class VideoRunsListOut(BaseModel):
    workspaceId: str
    videoId: str
    items: List[VideoRunSummary]


class VideoProgressOut(BaseModel):
    workspaceId: str
    videoId: str
    variant: str
    status: str
    runId: Optional[str] = None
    expectedSnapshots: Optional[int] = None
    processedSnapshots: int
    processedOk: int
    processedErr: int
    percent: float


class ColorFBL(BaseModel):
    base: str
    finish: Optional[str] = None
    lightness: Optional[str] = None
    conf: float


class DetectionAssets(BaseModel):
    annotatedImageS3Key: Optional[str] = None
    vehicleImageS3Key: Optional[str] = None
    plateImageS3Key: Optional[str] = None
    annotatedUrl: Optional[str] = None
    vehicleUrl: Optional[str] = None
    plateUrl: Optional[str] = None


class PartEvidence(BaseModel):
    name: str
    conf: float


class DetectionOut(BaseModel):
    id: str
    analysisId: str
    runId: str
    trackId: int
    snapshotS3Key: str
    yoloType: Optional[str] = None
    typeLabel: Optional[str] = None
    typeConf: Optional[float] = None
    makeLabel: Optional[str] = None
    makeConf: Optional[float] = None
    modelLabel: Optional[str] = None
    modelConf: Optional[float] = None
    plateText: Optional[str] = None
    plateConf: Optional[float] = None
    colors: List[ColorFBL] = []
    parts: List[PartEvidence] = []
    latencyMs: Optional[int] = None
    memoryGb: Optional[float] = None
    status: str
    errorMsg: Optional[str] = None
    assets: DetectionAssets


class DetectionListOut(BaseModel):
    workspaceId: str
    videoId: str
    variant: str
    runId: str
    items: List[DetectionOut]


class DetectionUpdateIn(BaseModel):
    typeLabel: Optional[str] = None
    makeLabel: Optional[str] = None
    modelLabel: Optional[str] = None
    plateText: Optional[str] = None
    colors: Optional[List[ColorFBL]] = None


# ─────────────────────────────────────────────────────────────
# GET /workspaces/{workspace_id}/videos/{video_id}
# ─────────────────────────────────────────────────────────────

@router.get("/{workspace_id}/videos/{video_id}", response_model=VideoDetailOut)
def get_video_detail(
    workspace_id: str,
    video_id: str,
    me=Depends(require_user),
):
    """
    Returns the videos row plus the most recent run container (if any).
    """
    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

        v = conn.execute(
            text(
                """
                SELECT id,
                       workspace_id,
                       workspace_code,
                       file_name,
                       camera_label,
                       camera_code,
                       recorded_at,
                       s3_key_raw,
                       frame_stride,
                       status,
                       created_at,
                       updated_at,
                       error_msg,
                       processing_started_at,
                       processing_finished_at
                  FROM videos
                 WHERE id = :vid
                   AND workspace_id = :wid
                """
            ),
            {"vid": video_id, "wid": workspace_id},
        ).mappings().first()

        if not v:
            raise HTTPException(status_code=404, detail="Video not found")

        va = conn.execute(
            text(
                """
                SELECT id,
                       video_id,
                       variant,
                       status,
                       run_id,
                       expected_snapshots,
                       processed_snapshots,
                       processed_ok,
                       processed_err,
                       run_started_at,
                       run_finished_at,
                       last_snapshot_at
                  FROM video_analyses
                 WHERE workspace_id = :wid
                   AND video_id = :vid
                 ORDER BY updated_at DESC
                 LIMIT 1
                """
            ),
            {"wid": workspace_id, "vid": video_id},
        ).mappings().first()

    video = VideoRowOut(
        id=v["id"],
        workspaceId=v["workspace_id"],
        workspaceCode=v.get("workspace_code"),
        fileName=v["file_name"],
        cameraLabel=v.get("camera_label"),
        cameraCode=v["camera_code"],
        recordedAt=v.get("recorded_at"),
        s3KeyRaw=v["s3_key_raw"],
        frameStride=v["frame_stride"],
        status=v["status"],
        createdAt=v.get("created_at"),
        updatedAt=v.get("updated_at"),
        errorMsg=v.get("error_msg"),
        processingStartedAt=v.get("processing_started_at"),
        processingFinishedAt=v.get("processing_finished_at"),
    )

    latest_run: Optional[VideoRunSummary] = None
    if va:
        latest_run = VideoRunSummary(
            analysisId=va["id"],
            videoId=va["video_id"],
            variant=va["variant"],
            status=va["status"],
            runId=str(va["run_id"]) if va["run_id"] else None,
            expectedSnapshots=va["expected_snapshots"],
            processedSnapshots=va["processed_snapshots"] or 0,
            processedOk=va["processed_ok"] or 0,
            processedErr=va["processed_err"] or 0,
            runStartedAt=va.get("run_started_at"),
            runFinishedAt=va.get("run_finished_at"),
            lastSnapshotAt=va.get("last_snapshot_at"),
        )

    return VideoDetailOut(video=video, latestRun=latest_run)


# ─────────────────────────────────────────────────────────────
# GET /workspaces/{workspace_id}/videos/{video_id}/analyses
# ─────────────────────────────────────────────────────────────

@router.get(
    "/{workspace_id}/videos/{video_id}/analyses",
    response_model=VideoRunsListOut,
)
def list_video_runs(
    workspace_id: str,
    video_id: str,
    me=Depends(require_user),
):
    """
    Lists run containers for this video (usually one per variant).
    """
    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))
        rows = conn.execute(
            text(
                """
                SELECT id,
                       video_id,
                       variant,
                       status,
                       run_id,
                       expected_snapshots,
                       processed_snapshots,
                       processed_ok,
                       processed_err,
                       run_started_at,
                       run_finished_at,
                       last_snapshot_at
                  FROM video_analyses
                 WHERE workspace_id = :wid
                   AND video_id = :vid
                 ORDER BY variant, run_started_at NULLS LAST, created_at
                """
            ),
            {"wid": workspace_id, "vid": video_id},
        ).mappings().all()

    items: List[VideoRunSummary] = []
    for r in rows:
        items.append(
            VideoRunSummary(
                analysisId=r["id"],
                videoId=r["video_id"],
                variant=r["variant"],
                status=r["status"],
                runId=str(r["run_id"]) if r["run_id"] else None,
                expectedSnapshots=r["expected_snapshots"],
                processedSnapshots=r["processed_snapshots"] or 0,
                processedOk=r["processed_ok"] or 0,
                processedErr=r["processed_err"] or 0,
                runStartedAt=r.get("run_started_at"),
                runFinishedAt=r.get("run_finished_at"),
                lastSnapshotAt=r.get("last_snapshot_at"),
            )
        )

    return VideoRunsListOut(workspaceId=workspace_id, videoId=video_id, items=items)


# ─────────────────────────────────────────────────────────────
# GET /workspaces/{workspace_id}/videos/{video_id}/progress
# ─────────────────────────────────────────────────────────────

@router.get(
    "/{workspace_id}/videos/{video_id}/progress",
    response_model=VideoProgressOut,
)
def get_video_progress(
    workspace_id: str,
    video_id: str,
    variant: str = Query("cmt"),
    me=Depends(require_user),
):
    """
    Returns run-level counters for the current variant
    (for your progress bar).
    """
    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))
        row = conn.execute(
            text(
                """
                SELECT id,
                       video_id,
                       variant,
                       status,
                       run_id,
                       expected_snapshots,
                       processed_snapshots,
                       processed_ok,
                       processed_err
                  FROM video_analyses
                 WHERE workspace_id = :wid
                   AND video_id     = :vid
                   AND variant      = :variant
                 ORDER BY updated_at DESC,
                          run_started_at DESC NULLS LAST,
                          created_at DESC
                 LIMIT 1
                """
            ),
            {"wid": workspace_id, "vid": video_id, "variant": variant},
        ).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Video analysis run not found")

    percent = _compute_percent(row["processed_snapshots"], row["expected_snapshots"])

    return VideoProgressOut(
        workspaceId=workspace_id,
        videoId=row["video_id"],
        variant=row["variant"],
        status=row["status"],
        runId=str(row["run_id"]) if row["run_id"] else None,
        expectedSnapshots=row["expected_snapshots"],
        processedSnapshots=row["processed_snapshots"] or 0,
        processedOk=row["processed_ok"] or 0,
        processedErr=row["processed_err"] or 0,
        percent=percent,
    )


# ─────────────────────────────────────────────────────────────
# GET /workspaces/{workspace_id}/videos/{video_id}/detections
# ─────────────────────────────────────────────────────────────

@router.get(
    "/{workspace_id}/videos/{video_id}/detections",
    response_model=DetectionListOut,
)
def list_video_detections(
    workspace_id: str,
    video_id: str,
    variant: str = Query("cmt"),
    runId: Optional[str] = Query(None),
    me=Depends(require_user),
):
    """
    Lists per-vehicle/per-snapshot detections for the given video.

    - Default: uses the current run_id from video_analyses for the variant.
    - Returns one row per (analysis_id, run_id, track_id).
    """
    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

        va = conn.execute(
            text(
                """
                SELECT id,
                       video_id,
                       variant,
                       run_id
                  FROM video_analyses
                 WHERE workspace_id = :wid
                   AND video_id     = :vid
                   AND variant      = :variant
                 ORDER BY updated_at DESC,
                          run_started_at DESC NULLS LAST,
                          created_at DESC
                 LIMIT 1
                """
            ),
            {"wid": workspace_id, "vid": video_id, "variant": variant},
        ).mappings().first()

        if not va:
            raise HTTPException(status_code=404, detail="Video analysis run not found")

        analysis_id = va["id"]
        effective_run_id = runId or (str(va["run_id"]) if va["run_id"] else None)

        if not effective_run_id:
            # No run yet → no detections
            return DetectionListOut(
                workspaceId=workspace_id,
                videoId=video_id,
                variant=variant,
                runId="",
                items=[],
            )

        rows = conn.execute(
            text(
                """
                SELECT id,
                       analysis_id,
                       run_id,
                       track_id,
                       snapshot_s3_key,
                       yolo_type,
                       type_label,
                       type_conf,
                       make_label,
                       make_conf,
                       model_label,
                       model_conf,
                       plate_text,
                       plate_conf,
                       colors,
                       parts,
                       assets,
                       latency_ms,
                       memory_usage,
                       status,
                       error_msg
                  FROM video_detections
                 WHERE analysis_id = :aid
                   AND run_id = :run_id
                 ORDER BY track_id, detected_at NULLS LAST, created_at
                """
            ),
            {"aid": analysis_id, "run_id": effective_run_id},
        ).mappings().all()

    items: List[DetectionOut] = []
    for r in rows:
        colors_raw = r.get("colors") or []
        parts_raw = r.get("parts") or []
        assets_raw = r.get("assets") or {}

        items.append(
            DetectionOut(
                id=r["id"],
                analysisId=r["analysis_id"],
                runId=str(r["run_id"]),
                trackId=r["track_id"],
                snapshotS3Key=r["snapshot_s3_key"],
                yoloType=r.get("yolo_type"),
                typeLabel=r.get("type_label"),
                typeConf=r.get("type_conf"),
                makeLabel=r.get("make_label"),
                makeConf=r.get("make_conf"),
                modelLabel=r.get("model_label"),
                modelConf=r.get("model_conf"),
                plateText=r.get("plate_text"),
                plateConf=r.get("plate_conf"),
                colors=[ColorFBL(**c) for c in colors_raw],
                parts=[PartEvidence(**p) for p in parts_raw],
                latencyMs=r.get("latency_ms"),
                memoryGb=r.get("memory_usage"),
                status=r["status"],
                errorMsg=r.get("error_msg"),
                assets=DetectionAssets(
                    annotatedImageS3Key=assets_raw.get("annotated_image_s3_key"),
                    vehicleImageS3Key=assets_raw.get("vehicle_image_s3_key"),
                    plateImageS3Key=assets_raw.get("plate_image_s3_key"),
                ),
            )
        )

    return DetectionListOut(
        workspaceId=workspace_id,
        videoId=video_id,
        variant=variant,
        runId=effective_run_id,
        items=items,
    )


# ─────────────────────────────────────────────────────────────
# GET /workspaces/{workspace_id}/videos/{video_id}/detections/{detection_id}
# ─────────────────────────────────────────────────────────────

@router.get(
    "/{workspace_id}/videos/{video_id}/detections/{detection_id}",
    response_model=DetectionOut,
)
def get_video_detection(
    workspace_id: str,
    video_id: str,
    detection_id: str,
    presign: int = Query(0, ge=0, le=1),
    ttl: int = Query(900, ge=60, le=604800),
    me=Depends(require_user),
):
    """
    Returns a single detection row; optionally presigns asset URLs.
    """
    effective_ttl = min(int(ttl), ANALYSIS_TTL_DEFAULT)

    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))
        r = conn.execute(
            text(
                """
                SELECT d.id,
                       d.analysis_id,
                       d.run_id,
                       d.track_id,
                       d.snapshot_s3_key,
                       d.yolo_type,
                       d.type_label,
                       d.type_conf,
                       d.make_label,
                       d.make_conf,
                       d.model_label,
                       d.model_conf,
                       d.plate_text,
                       d.plate_conf,
                       d.colors,
                       d.parts,
                       d.assets,
                       d.latency_ms,
                       d.memory_usage,
                       d.status,
                       d.error_msg
                  FROM video_detections d
                  JOIN video_analyses a
                    ON d.analysis_id = a.id
                 WHERE d.id = :det_id
                   AND a.workspace_id = :wid
                   AND a.video_id = :vid
                """
            ),
            {"det_id": detection_id, "wid": workspace_id, "vid": video_id},
        ).mappings().first()

    if not r:
        raise HTTPException(status_code=404, detail="Detection not found")

    colors_raw = r.get("colors") or []
    parts_raw = r.get("parts") or []
    assets_raw = r.get("assets") or {}

    if presign:
        assets_raw = _presign_assets(assets_raw, effective_ttl)

    assets = DetectionAssets(
        annotatedImageS3Key=assets_raw.get("annotated_image_s3_key"),
        vehicleImageS3Key=assets_raw.get("vehicle_image_s3_key"),
        plateImageS3Key=assets_raw.get("plate_image_s3_key"),
        annotatedUrl=assets_raw.get("annotatedUrl"),
        vehicleUrl=assets_raw.get("vehicleUrl"),
        plateUrl=assets_raw.get("plateUrl"),
    )

    return DetectionOut(
        id=r["id"],
        analysisId=r["analysis_id"],
        runId=str(r["run_id"]),
        trackId=r["track_id"],
        snapshotS3Key=r["snapshot_s3_key"],
        yoloType=r.get("yolo_type"),
        typeLabel=r.get("type_label"),
        typeConf=r.get("type_conf"),
        makeLabel=r.get("make_label"),
        makeConf=r.get("make_conf"),
        modelLabel=r.get("model_label"),
        modelConf=r.get("model_conf"),
        plateText=r.get("plate_text"),
        plateConf=r.get("plate_conf"),
        colors=[ColorFBL(**c) for c in colors_raw],
        parts=[PartEvidence(**p) for p in parts_raw],
        latencyMs=r.get("latency_ms"),
        memoryGb=r.get("memory_usage"),
        status=r["status"],
        errorMsg=r.get("error_msg"),
        assets=assets,
    )


# ─────────────────────────────────────────────────────────────
# PATCH /workspaces/{workspace_id}/videos/{video_id}/detections/{detection_id}
# ─────────────────────────────────────────────────────────────

@router.patch(
    "/{workspace_id}/videos/{video_id}/detections/{detection_id}",
    response_model=DetectionOut,
)
def update_video_detection(
    workspace_id: str,
    video_id: str,
    detection_id: str,
    body: DetectionUpdateIn,
    me=Depends(require_user),
):
    """
    Partial update endpoint for manual corrections:

    - Editable: typeLabel, makeLabel, modelLabel, plateText, colors.
    - When labels or plate text are edited, their *_conf fields are set to 0.0
      to indicate that the values are no longer pure AI predictions.
    """
    data = body.dict(exclude_unset=True)

    # If nothing to update; just return current snapshot
    if not data:
        return get_video_detection(
            workspace_id=workspace_id,
            video_id=video_id,
            detection_id=detection_id,
            presign=0,
            ttl=ANALYSIS_TTL_DEFAULT,
            me=me,  # type: ignore[arg-type]
        )

    set_clauses: List[str] = []
    params: Dict[str, Any] = {"det_id": detection_id}

    # Manual overrides for labels and plate text: always zero confidence
    if "typeLabel" in data:
        set_clauses.append("type_label = :typeLabel")
        set_clauses.append("type_conf = 0.0")
        params["typeLabel"] = data["typeLabel"]

    if "makeLabel" in data:
        set_clauses.append("make_label = :makeLabel")
        set_clauses.append("make_conf = 0.0")
        params["makeLabel"] = data["makeLabel"]

    if "modelLabel" in data:
        set_clauses.append("model_label = :modelLabel")
        set_clauses.append("model_conf = 0.0")
        params["modelLabel"] = data["modelLabel"]

    if "plateText" in data:
        set_clauses.append("plate_text = :plateText")
        set_clauses.append("plate_conf = 0.0")
        params["plateText"] = data["plateText"]

    # Colors remain directly editable as JSON
    if "colors" in data:
        set_clauses.append("colors = :colors")
        params["colors"] = data["colors"]

    if not set_clauses:
        return get_video_detection(
            workspace_id=workspace_id,
            video_id=video_id,
            detection_id=detection_id,
            presign=0,
            ttl=ANALYSIS_TTL_DEFAULT,
            me=me,  # type: ignore[arg-type]
        )

    set_clauses.append("updated_at = now()")
    set_sql = ", ".join(set_clauses)

    with engine.begin() as conn:
        _assert_workspace(conn, workspace_id, str(me.id))

        # Ensure detection belongs to this workspace+video
        owned = conn.execute(
            text(
                """
                SELECT d.id
                  FROM video_detections d
                  JOIN video_analyses a
                    ON d.analysis_id = a.id
                 WHERE d.id = :det_id
                   AND a.workspace_id = :wid
                   AND a.video_id = :vid
                """
            ),
            {"det_id": detection_id, "wid": workspace_id, "vid": video_id},
        ).fetchone()

        if not owned:
            raise HTTPException(status_code=404, detail="Detection not found")

        conn.execute(
            text(f"UPDATE video_detections SET {set_sql} WHERE id = :det_id"),
            params,
        )

    # Return the updated row (no presign by default)
    return get_video_detection(
        workspace_id=workspace_id,
        video_id=video_id,
        detection_id=detection_id,
        presign=0,
        ttl=ANALYSIS_TTL_DEFAULT,
        me=me,  # type: ignore[arg-type]
    )

