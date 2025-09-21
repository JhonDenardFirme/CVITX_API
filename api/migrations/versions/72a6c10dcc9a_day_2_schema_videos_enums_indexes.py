"""day-2 schema (videos + enums + indexes)

Revision ID: 72a6c10dcc9a
Revises:
Create Date: 2025-09-21 15:39:37.910063
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '72a6c10dcc9a'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema (non-destructive; keeps existing tables/FKs)."""

    # --- Normalize enum type name & default for videos.status ---
    op.execute("""
    DO $$
    BEGIN
        -- If the old type exists, rename to the canonical name.
        IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'videostatus') THEN
            ALTER TYPE videostatus RENAME TO video_status_enum;
        ELSIF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'video_status_enum') THEN
            CREATE TYPE video_status_enum AS ENUM ('uploaded','queued','processing','done','error');
        END IF;
    END$$;
    """)

    # Add new nullable columns (safe)
    op.add_column('videos', sa.Column('workspace_code', sa.Text(), nullable=True))
    op.add_column('videos', sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('videos', sa.Column('processing_finished_at', sa.DateTime(timezone=True), nullable=True))

    # Widen types & allow NULLs where needed (matches models)
    op.alter_column('videos', 'file_name',
        existing_type=sa.VARCHAR(),
        type_=sa.Text(),
        nullable=True)
    op.alter_column('videos', 'camera_label',
        existing_type=sa.VARCHAR(),
        type_=sa.Text(),
        nullable=True)
    op.alter_column('videos', 'camera_code',
        existing_type=sa.VARCHAR(),
        type_=sa.Text(),
        nullable=True)
    op.alter_column('videos', 's3_key_raw',
        existing_type=sa.TEXT(),
        nullable=True)
    op.alter_column('videos', 'recorded_at',
        existing_type=postgresql.TIMESTAMP(timezone=True),
        nullable=True)
    op.alter_column('videos', 'frame_stride',
        existing_type=sa.INTEGER(),
        nullable=True)

    # Align videos.status to canonical enum + stable default
    op.alter_column('videos', 'status',
        existing_type=postgresql.ENUM('uploaded', 'queued', 'processing', 'done', 'error', name='videostatus'),
        type_=sa.Enum('uploaded', 'queued', 'processing', 'done', 'error', name='video_status_enum'),
        server_default=sa.text("'uploaded'::video_status_enum"),
        existing_nullable=False,
        postgresql_using="status::text::video_status_enum")

    # Ensure updated_at is NOT NULL (keep server default now())
    op.alter_column('videos', 'updated_at',
        existing_type=postgresql.TIMESTAMP(timezone=True),
        nullable=False,
        existing_server_default=sa.text('now()'))

    # Helpful indexes
    op.create_index('ix_videos_status', 'videos', ['status'], unique=False)
    op.create_index('ix_videos_workspace_status', 'videos', ['workspace_id', 'status'], unique=False)

    # NOTE: We intentionally do NOT drop the workspaces table or the videos FK.
    # If a FK exists in DB but not in models, we keep it (non-destructive).


def downgrade() -> None:
    """Best-effort downgrade (non-destructive to other tables)."""

    # Drop the helper indexes
    op.drop_index('ix_videos_workspace_status', table_name='videos')
    op.drop_index('ix_videos_status', table_name='videos')

    # Revert videos.status default to plain text default (if needed) and keep type
    op.alter_column('videos', 'status',
        existing_type=sa.Enum('uploaded', 'queued', 'processing', 'done', 'error', name='video_status_enum'),
        server_default=None,
        existing_nullable=False)

    # Re-tighten nullable columns back to prior state as best as possible
    op.alter_column('videos', 'frame_stride',
        existing_type=sa.INTEGER(),
        nullable=False)
    op.alter_column('videos', 'recorded_at',
        existing_type=postgresql.TIMESTAMP(timezone=True),
        nullable=False)
    op.alter_column('videos', 's3_key_raw',
        existing_type=sa.TEXT(),
        nullable=False)
    op.alter_column('videos', 'camera_code',
        existing_type=sa.Text(),
        type_=sa.VARCHAR(),
        nullable=False)
    op.alter_column('videos', 'camera_label',
        existing_type=sa.Text(),
        type_=sa.VARCHAR(),
        nullable=False)
    op.alter_column('videos', 'file_name',
        existing_type=sa.Text(),
        type_=sa.VARCHAR(),
        nullable=False)

    # Drop the added columns
    op.drop_column('videos', 'processing_finished_at')
    op.drop_column('videos', 'processing_started_at')
    op.drop_column('videos', 'workspace_code')

    # We do NOT recreate/drop the 'workspaces' table or videos FK here.
