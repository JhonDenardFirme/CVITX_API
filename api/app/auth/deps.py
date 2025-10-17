import os
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import create_engine, text
from jose import JWTError
from .security import decode_token

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL not set")
SQLA_URL = DB_URL.replace("postgresql+psycopg2","postgresql")
engine = create_engine(SQLA_URL, pool_pre_ping=True)

auth_scheme = HTTPBearer(auto_error=False)

def require_user(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(401, "Missing token")
    try:
        payload = decode_token(creds.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid/expired token")

    with engine.begin() as conn:
        row = conn.execute(text("SELECT id, email FROM users WHERE id=:id"), {"id": user_id}).fetchone()
    if not row:
        raise HTTPException(401, "User not found")

    class U: ...
    u = U(); u.id = row[0]; u.email = row[1]
    return u
