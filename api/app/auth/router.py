import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, text
import bcrypt
from .security import create_access_token

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL not set")
SQLA_URL = DB_URL.replace("postgresql+psycopg2","postgresql")
engine = create_engine(SQLA_URL, pool_pre_ping=True)

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class LoginOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

@router.post("/login", response_model=LoginOut)
def login(body: LoginIn):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, email, password_hash FROM users WHERE email=:e"),
            {"e": body.email}
        ).fetchone()
    if not row or not row[2]:
        raise HTTPException(401, "Invalid credentials")

    if not bcrypt.checkpw(body.password.encode(), row[2].encode()):
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token(sub=str(row[0]), extra={"email": row[1]})
    return {"access_token": token, "user": {"id": str(row[0]), "email": row[1]}}

# --- BEGIN: add GET /auth/me alias ---
from fastapi import Depends
from app.auth.deps import require_user

@router.get("/me", tags=["auth"])
def auth_me(me=Depends(require_user)):
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT id, email, first_name, last_name, affiliation_name, role, is_active,
                   force_password_reset, created_at, updated_at, avatar_s3_key
            FROM users
            WHERE id = :id
        """), {"id": me.id}).fetchone()
    if not row:
        raise HTTPException(404, "User not found")
    return {
        "id": str(row[0]),
        "email": row[1],
        "first_name": row[2],
        "last_name": row[3],
        "affiliation_name": row[4],
        "role": row[5],
        "is_active": row[6],
        "force_password_reset": row[7],
        "created_at": row[8],
        "updated_at": row[9],
        "avatar_s3_key": row[10],
    }
# --- END: add GET /auth/me alias ---
