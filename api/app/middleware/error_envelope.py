import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarHTTPException

def enable_error_envelope(app: FastAPI):
    if os.getenv("ERROR_ENVELOPE", "0") not in ("1", "true", "TRUE"):
        return

    @app.exception_handler(StarHTTPException)
    async def http_exc_handler(request: Request, exc: StarHTTPException):
        return JSONResponse(status_code=exc.status_code, content={
            "ok": False, "error": {"code": str(exc.status_code), "message": exc.detail if exc.detail else "HTTP error"}
        })

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "ok": False, "error": {"code": "VALIDATION_ERROR", "message": "Invalid request", "details": exc.errors()}
        })
