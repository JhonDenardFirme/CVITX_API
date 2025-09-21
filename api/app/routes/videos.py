from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..deps import get_db, require_api_key
from ..models import Video, VideoStatus

router = APIRouter()

class RegisterVideoRequest(BaseModel):
    video_id: UUID
    workspace_id: UUID
    file_name: str
    camera_label: str
    camera_code: str
    s3_key_raw: str
    recordedAt: datetime

@router.post("/videos/register", dependencies=[Depends(require_api_key)])
def register_video(body: RegisterVideoRequest, db: Session = Depends(get_db)):
    # verify workspace exists
    exists = db.execute(
        text("select 1 from workspaces where id = :wid limit 1"),
        {"wid": str(body.workspace_id)},
    ).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Workspace not found")

    v = Video(
        id=body.video_id,
        workspace_id=body.workspace_id,
        file_name=body.file_name,
        camera_label=body.camera_label,
        camera_code=body.camera_code,
        recorded_at=body.recordedAt,
        s3_key_raw=body.s3_key_raw,
        status=VideoStatus.uploaded,
    )
    db.add(v)
    db.commit()
    return {"ok": True, "video_id": str(v.id), "camera_code": v.camera_code, "status": v.status.value}
