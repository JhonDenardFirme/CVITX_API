from sqlalchemy import text
from app.deps import engine
from app.models import Base

# create tables if they don't exist
with engine.begin() as conn:
    Base.metadata.create_all(conn)

    # seed demo workspace (id/code locked for Day-1)
    conn.execute(text("""
        INSERT INTO workspaces (id, workspace_code, title, description)
        VALUES ('11111111-1111-1111-1111-111111111111','CVX1027','Demo Case','')
        ON CONFLICT (id) DO NOTHING
    """))

print("DB initialized / tables ensured / workspace seeded.")
