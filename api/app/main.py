from app.routes.workspaces import router as workspaces_router
from dotenv import load_dotenv
from app.auth.router import router as auth_router
from app.routes.workspace_files import router as workspace_files_router
from app.routes.users_profile import router as users_profile_router
from app.routes.videos_status import router as videos_status_router
from app.routes.exports import router as exports_router
from app.utils.version import git_sha_short
from app.routes import detections as detections_routes
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
