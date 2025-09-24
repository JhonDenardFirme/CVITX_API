from app.routes import detections as detections_routes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routes import health, uploads, videos

from .routes import queue as queue_routes
from .routes import snapshots as snapshot_routes
from .routes import jobs as jobs_routes


app = FastAPI(title="CVITX API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(uploads.router)
app.include_router(videos.router)
app.include_router(queue_routes.router)
app.include_router(snapshot_routes.router)
app.include_router(jobs_routes.router)
app.include_router(detections_routes.router)



@app.get("/healthz")
def healthz():
    return {"ok": True}
from app.routes.videos_status import router as videos_status_router
from app.routes.exports import router as exports_router
app.include_router(videos_status_router)
app.include_router(exports_router)
