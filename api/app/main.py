from fastapi.middleware.cors import CORSMiddleware
from app.routes import health
from app.routes.workspaces import router as workspaces_router
from dotenv import load_dotenv
from app.auth.router import router as auth_router
from app.routes.workspace_files import router as workspace_files_router
from app.routes.users_profile import router as users_profile_router
from app.routes.videos_status import router as videos_status_router
from app.routes.exports import router as exports_router
from app.utils.version import git_sha_short
from app.routes import detections as detections_routes
from app.routes import analysis_images
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routes import health, uploads, videos

from .routes import queue as queue_routes
from .routes import snapshots as snapshot_routes
from .routes import jobs as jobs_routes


load_dotenv()
app = FastAPI(title="CVITX API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analysis_images.router)
app.include_router(analysis_images.alias)
app.include_router(uploads.router)
app.include_router(videos.router)
app.include_router(queue_routes.router)
app.include_router(snapshot_routes.router)
app.include_router(jobs_routes.router)
app.include_router(detections_routes.router)



app.include_router(exports_router)
from app.middleware.error_envelope import enable_error_envelope
enable_error_envelope(app)

app.include_router(videos_status_router)

app.include_router(users_profile_router)

app.include_router(workspace_files_router)

app.include_router(auth_router)

app.include_router(workspaces_router)

# --- BEGIN add: workspace videos router ---
from app.routes import workspace_videos
app.include_router(workspace_videos.router)
# --- END add: workspace videos router ---

from app.routes import analyses
app.include_router(analyses.router)

# --- Basic rate limiting (SlowAPI) ---
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def ratelimit_handler(request, exc):
    return PlainTextResponse("Too Many Requests", status_code=429)
# -------------------------------------

# --- CORS hardening ---
import os as _os
_ORIGINS = [o.strip() for o in _os.environ.get("FRONTEND_ORIGINS", _os.environ.get("FRONTEND_ORIGIN", "")).split(",") if o.strip()]
if _ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
# ----------------------
