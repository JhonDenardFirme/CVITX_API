from fastapi import APIRouter
router = APIRouter()

@router.get("/healthz")
def healthz():
    return {"ok": True, "service": "cvitx-api"}
