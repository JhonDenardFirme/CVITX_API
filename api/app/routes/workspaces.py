import os, uuid, re
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy import exc as sa_exc
from app.auth.deps import require_user

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL not set")
SQLA_URL = DB_URL.replace("postgresql+psycopg2","postgresql")
engine = create_engine(SQLA_URL, pool_pre_ping=True)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

class WsOut(BaseModel):
    id: str
    code: str | None = None
    title: str | None = None
    description: str | None = None
    created_at: str | None = None

class WsCreate(BaseModel):
    # code is ignored (server-generated)
    code: str | None = None
    title: str | None = None
    description: str | None = None

class WsPatch(BaseModel):
    title: str | None = None
    description: str | None = None
    code: str | None = None  # immutable, reject if provided

def _ensure_counter_table(conn):
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS workspace_counters (
            owner_user_id uuid PRIMARY KEY,
            last_no integer NOT NULL
        )
    """))

def _next_ctx_code(conn, owner_id: uuid.UUID) -> str:
    # Upsert-style: bump existing counter; if missing, seed with 1001
    r = conn.execute(text("""
        WITH up AS (
            UPDATE workspace_counters
               SET last_no = last_no + 1
             WHERE owner_user_id = :uid
         RETURNING last_no
        ), ins AS (
            INSERT INTO workspace_counters(owner_user_id, last_no)
            SELECT :uid, 1001
            WHERE NOT EXISTS (SELECT 1 FROM up)
         RETURNING last_no
        )
        SELECT last_no FROM up
        UNION ALL
        SELECT last_no FROM ins
    """), {"uid": str(owner_id)}).fetchone()
    return f"CTX{int(r[0])}"

@router.get("", response_model=list[WsOut])
def list_workspaces(me=Depends(require_user)):
    uid = uuid.UUID(str(me.id))
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, code, title, description, created_at
              FROM workspaces
             WHERE owner_user_id = :uid
               AND deleted_at IS NULL
          ORDER BY created_at DESC NULLS LAST
        """), {"uid": str(uid)}).fetchall()
    return [{"id": str(r[0]), "code": r[1], "title": r[2], "description": r[3],
             "created_at": (r[4].isoformat() if r[4] else None)} for r in rows]

@router.get("/{workspace_id}", response_model=WsOut)
def get_workspace(workspace_id: str, me=Depends(require_user)):
    uid = uuid.UUID(str(me.id))
    with engine.begin() as conn:
        r = conn.execute(text("""
            SELECT id, code, title, description, created_at
              FROM workspaces
             WHERE id = :id
               AND owner_user_id = :uid
               AND deleted_at IS NULL
        """), {"id": str(workspace_id), "uid": str(uid)}).fetchone()
    if not r:
        raise HTTPException(404, "Not found")
    return {"id": str(r[0]), "code": r[1], "title": r[2], "description": r[3],
            "created_at": (r[4].isoformat() if r[4] else None)}

@router.post("", response_model=WsOut)
def create_workspace(body: WsCreate, me=Depends(require_user)):
    uid = uuid.UUID(str(me.id))
    with engine.begin() as conn:
        _ensure_counter_table(conn)
        code = _next_ctx_code(conn, uid)  # ignore client code
        try:
            row = conn.execute(text("""
                INSERT INTO workspaces (id, owner_user_id, code, title, description, created_at)
                VALUES (:id, :uid, :code, :title, :desc, now())
             RETURNING id, code, title, description, created_at
            """), {"id": str(uuid.uuid4()), "uid": str(uid), "code": code,
                   "title": body.title, "desc": body.description}).fetchone()
        except sa_exc.IntegrityError:
            # could be unique(owner,code) or your "max 3 active" check
            raise HTTPException(409, "Workspace already exists or violates a constraint")
    return {"id": str(row[0]), "code": row[1], "title": row[2], "description": row[3],
            "created_at": (row[4].isoformat() if row[4] else None)}

@router.patch("/{workspace_id}", response_model=WsOut)
def update_workspace(workspace_id: str, body: WsPatch, me=Depends(require_user)):
    if body.code is not None:
        raise HTTPException(400, "Code is system-managed and cannot be edited")
    uid = uuid.UUID(str(me.id))
    sets, params = [], {"id": str(workspace_id), "uid": str(uid)}
    if body.title is not None:
        sets.append("title = :title"); params["title"] = body.title
    if body.description is not None:
        sets.append("description = :desc"); params["desc"] = body.description
    if not sets:
        return get_workspace(workspace_id, me)
    with engine.begin() as conn:
        r = conn.execute(text(f"""
            UPDATE workspaces
               SET {", ".join(sets)}
             WHERE id = :id
               AND owner_user_id = :uid
               AND deleted_at IS NULL
         RETURNING id, code, title, description, created_at
        """), params).fetchone()
    if not r:
        raise HTTPException(404, "Not found")
    return {"id": str(r[0]), "code": r[1], "title": r[2], "description": r[3],
            "created_at": (r[4].isoformat() if r[4] else None)}

@router.delete("/{workspace_id}", status_code=204)
def delete_workspace(workspace_id: str, me=Depends(require_user)):
    uid = uuid.UUID(str(me.id))
    with engine.begin() as conn:
        res = conn.execute(text("""
            UPDATE workspaces
               SET deleted_at = now()
             WHERE id = :id
               AND owner_user_id = :uid
               AND deleted_at IS NULL
        """), {"id": str(workspace_id), "uid": str(uid)})
    if res.rowcount == 0:
        raise HTTPException(404, "Not found")
    return Response(status_code=204)
