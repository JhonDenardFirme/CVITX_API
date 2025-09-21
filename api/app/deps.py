from fastapi import Header, HTTPException, status
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import settings

# Note: create_engine doesn't connect yet; it builds a pool.
engine = create_engine(settings.db_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_api_key(x_api_key: str | None = Header(None)):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
