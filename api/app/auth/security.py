import os, datetime as dt
from typing import Any, Dict
from jose import jwt

JWT_SECRET = os.getenv("JWT_SECRET") or "dev-secret-change-me"
JWT_ALG    = "HS256"
JWT_ISS    = os.getenv("JWT_ISSUER", "cvitx-api")
ACC_MIN    = int(os.getenv("JWT_ACCESS_EXPIRE_MIN", "60"))

def _json_safe(v: Any) -> Any:
    # Ensure payload is JSON serializable (UUID, datetime, Decimal -> str)
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)

def create_access_token(sub: Any, extra: Dict[str, Any] | None = None) -> str:
    now = dt.datetime.utcnow()
    payload: Dict[str, Any] = {
        "sub": _json_safe(sub),
        "iss": JWT_ISS,
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(minutes=ACC_MIN)).timestamp()),
    }
    if extra:
        for k, v in extra.items():
            payload[k] = _json_safe(v)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options={"verify_aud": False})
