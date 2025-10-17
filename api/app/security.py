import os
from fastapi import Header, HTTPException, status

API_KEY = os.environ.get("API_KEY", "")

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
