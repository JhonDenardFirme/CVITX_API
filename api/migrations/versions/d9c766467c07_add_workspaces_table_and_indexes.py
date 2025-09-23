"""add workspaces table and indexes

Revision ID: d9c766467c07
Revises: fa13f3c0e25a
Create Date: 2025-09-22 14:43:43
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "d9c766467c07"
down_revision: Union[str, Sequence[str], None] = "fa13f3c0e25a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1) Create workspaces
    op.create_table(
        "workspaces",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("workspace_code", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_workspaces_workspace_code"), "workspaces", ["workspace_code"], unique=True)

    # 2) Seed any missing workspaces from existing videos BEFORE we add FK
    op.execute(
        """
        INSERT INTO workspaces (id, workspace_code)
        SELECT DISTINCT v.workspace_id, 'WS_' || substr(v.workspace_id::text, 1, 8)
        FROM videos v
        LEFT JOIN workspaces w ON w.id = v.workspace_id
        WHERE v.workspace_id IS NOT NULL AND w.id IS NULL;
        """
    )

    # 3) Add a NAMED FK so downgrade can drop it cleanly
    op.create_foreign_key(
        "fk_videos_workspace_id",
        "videos", "workspaces",
        ["workspace_id"], ["id"]
    )

def downgrade() -> None:
    op.drop_constraint("fk_videos_workspace_id", "videos", type_="foreignkey")
    op.drop_index(op.f("ix_workspaces_workspace_code"), table_name="workspaces")
    op.drop_table("workspaces")
