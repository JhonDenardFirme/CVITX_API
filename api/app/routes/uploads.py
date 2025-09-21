from fastapi import APIRouter, Depends
from pydantic import BaseModel
from uuid import uuid4
from ..deps import require_api_key
from ..services.s3 import make_raw_key, presign_put

router = APIRouter()

class PresignReq(BaseModel):
    workspace_id: str
    file_name: str

@router.post("/uploads/presign", dependencies=[Depends(require_api_key)])
def presign_upload(req: PresignReq):
    video_id = str(uuid4())
    camera_code = "CAM1"  # Day-1 stub; refine later on /videos/register
    s3_key_raw = make_raw_key("demo_user", req.workspace_id, video_id, req.file_name)
    url = presign_put(s3_key_raw, expires=900, content_type="video/mp4")
    return {
        "video_id": video_id,
        "camera_code": camera_code,
        "s3_key_raw": s3_key_raw,
        "presigned_url": url,
        "expires_in": 900
    }
