import os
from fastapi.middleware.cors import CORSMiddleware

def enable_cors(app):
    origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if not origins:
        return
    allow_origins = [o.strip() for o in origins.split(",") if o.strip()]
    allow_headers = os.getenv("CORS_ALLOW_HEADERS", "*,X-API-Key").split(",")
    allow_methods = os.getenv("CORS_ALLOW_METHODS", "GET,POST,OPTIONS").split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=[m.strip() for m in allow_methods],
        allow_headers=[h.strip() for h in allow_headers],
        expose_headers=["Content-Disposition"],
        max_age=int(os.getenv("CORS_MAX_AGE", "600")),
    )
