from uuid import UUID
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import text
import io, csv

from app.db import engine
from app.security import require_api_key

router = APIRouter(
    prefix="/api/exports",
    tags=["exports"],
    dependencies=[Depends(require_api_key)],
)

HEADERS = [
    "display_id","workspace_code","camera_code","track_id","plate_text",
    "type","type_conf","make","make_conf","model","model_conf","colors",
    "recordedAt","detectedIn_ms","detectedAt","snapshot_s3_key","plate_image_s3_key"
]

SELECT_FOR_EXPORT = """
SELECT
  d.display_id,
  split_part(d.display_id, '-', 1) AS workspace_code,
  split_part(d.display_id, '-', 2) AS camera_code,
  d.track_id,
  d.plate_text,
  d.type, d.type_conf, d.make, d.make_conf, d.model, d.model_conf,
  d.colors,
  d.recorded_at, d.detected_in_ms, d.detected_at,
  d.snapshot_s3_key, d.plate_image_s3_key
FROM detections d
WHERE d.workspace_id = :workspace_id
  {video_clause}
  {min_clause}
  {max_clause}
ORDER BY d.detected_at DESC, d.id
"""

@router.get("/detections.csv")
def export_detections_csv(
    workspace_id: UUID = Query(...),
    video_id: UUID | None = Query(None),
    min_detected_at: str | None = Query(None),
    max_detected_at: str | None = Query(None),
):
    params = {"workspace_id": str(workspace_id)}
    video_clause = "AND d.video_id = :video_id" if video_id else ""
    if video_id: params["video_id"] = str(video_id)
    min_clause = "AND d.detected_at >= :min_dt" if min_detected_at else ""
    if min_detected_at: params["min_dt"] = min_detected_at
    max_clause = "AND d.detected_at <= :max_dt" if max_detected_at else ""
    if max_detected_at: params["max_dt"] = max_detected_at

    sql = SELECT_FOR_EXPORT.format(
        video_clause=video_clause, min_clause=min_clause, max_clause=max_clause
    )

    def stream():
        with engine.begin() as conn:
            res = conn.execute(text(sql), params).mappings()
            buf = io.StringIO()
            w = csv.writer(buf)
            # header
            w.writerow(HEADERS); buf.seek(0)
            yield buf.read(); buf.seek(0); buf.truncate(0)
            # rows
            for row in res:
                colors = "|".join(row["colors"] or []) if row.get("colors") else ""
                w.writerow([
                    row["display_id"], row["workspace_code"], row["camera_code"], row["track_id"],
                    row.get("plate_text") or "",
                    row.get("type") or "", row.get("type_conf") or "",
                    row.get("make") or "", row.get("make_conf") or "",
                    row.get("model") or "", row.get("model_conf") or "",
                    colors,
                    (row["recorded_at"].isoformat() if row.get("recorded_at") else ""),
                    row.get("detected_in_ms") or "",
                    (row["detected_at"].isoformat() if row.get("detected_at") else ""),
                    row.get("snapshot_s3_key") or "",
                    row.get("plate_image_s3_key") or "",
                ])
                buf.seek(0)
                chunk = buf.read()
                buf.seek(0); buf.truncate(0)
                yield chunk

    return StreamingResponse(
        stream(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="detections.csv"'}
    )
