from app.utils.version import git_sha_short
from fastapi import APIRouter
router = APIRouter()

@router.get("/healthz")
def healthz():
    return {"ok": True, "service": "cvitx-api", "version": git_sha_short() }
