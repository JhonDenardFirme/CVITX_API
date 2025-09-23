import psycopg2
from ..config import settings

def get_workspace_code_by_id(workspace_id: str) -> str | None:
    conn = psycopg2.connect(settings.db_url.replace('postgresql+psycopg2://','postgresql://'))
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT workspace_code FROM workspaces WHERE id = %s", (workspace_id,))
            row = cur.fetchone()
            return row[0] if row and row[0] else None
    finally:
        conn.close()
