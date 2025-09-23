import uuid
from sqlalchemy import (
    Column, Text, DateTime, Integer, Enum, Index, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

# --- Enums ---
VideoStatusEnum = Enum(
    "uploaded", "queued", "processing", "done", "error",
    name="video_status_enum",
)

# --- Tables ---
class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    workspace_code = Column(Text, nullable=False, unique=True, index=True)
    title = Column(Text, nullable=True)
    description = Column(Text, nullable=True)


class Video(Base):
    __tablename__ = "videos"

    # IDs
    id = Column(PG_UUID(as_uuid=False), primary_key=True)  # app provides UUID
    workspace_id = Column(PG_UUID(as_uuid=False), ForeignKey("workspaces.id"), nullable=False)

    # Optional persisted code; still sent in messages
    workspace_code = Column(Text, nullable=True)

    # File & camera
    file_name = Column(Text, nullable=True)
    camera_label = Column(Text, nullable=True)
    camera_code = Column(Text, nullable=True)

    # Storage
    s3_key_raw = Column(Text, nullable=True)

    # Timing
    recorded_at = Column(DateTime(timezone=True), nullable=True)
    frame_stride = Column(Integer, nullable=True)  # default handled in code

    # Processing status
    status = Column(VideoStatusEnum, nullable=False, server_default="uploaded")
    error_msg = Column(Text, nullable=True)

    # Timestamps
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_finished_at = Column(DateTime(timezone=True), nullable=True)


# Indexes (deterministic names)
Index("ix_videos_workspace_status", Video.workspace_id, Video.status)
Index("ix_videos_status", Video.status)
