from uuid import UUID
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db import engine
from app.security import require_api_key

router = APIRouter(
    prefix="/api/videos",
    tags=["videos"],
    dependencies=[Depends(require_api_key)],
)

@router.get("/{video_id}/status")
def video_status(video_id: UUID):
    # Use updated_at (your DB doesn't have created_at). Alias it as created_at for the API shape.
    sql = text("""
        SELECT
            id::text                AS id,
            status::text            AS status,
            error_msg,
            recorded_at,
            updated_at              AS created_at
        FROM videos
        WHERE id = :vid
        LIMIT 1
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {"vid": str(video_id)}).mappings().first()
        if not row:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"ok": False, "error": {"code": "NOT_FOUND", "message": "Video not found"}}
            )
        return {
            "ok": True,
            "video_id": row["id"],
            "status": row["status"],
            "error_msg": row["error_msg"],
            "recorded_at": row["recorded_at"].isoformat() if row["recorded_at"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
