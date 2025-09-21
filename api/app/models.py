from sqlalchemy import (
    Column, Text, DateTime, Integer, Enum, Index, func
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()

VideoStatusEnum = Enum(
    "uploaded", "queued", "processing", "done", "error",
    name="video_status_enum",
)

class Video(Base):
    __tablename__ = "videos"

    id = Column(PG_UUID(as_uuid=False), primary_key=True)   # UUID stored as text
    workspace_id = Column(PG_UUID(as_uuid=False), nullable=False)

    # optional persisted code; we still send it in messages
    workspace_code = Column(Text, nullable=True)

    file_name = Column(Text, nullable=True)
    camera_label = Column(Text, nullable=True)
    camera_code = Column(Text, nullable=True)

    s3_key_raw = Column(Text, nullable=True)

    recorded_at = Column(DateTime(timezone=True), nullable=True)

    frame_stride = Column(Integer, nullable=True)  # default handled in code (3)

    status = Column(VideoStatusEnum, nullable=False, server_default="uploaded")
    error_msg = Column(Text, nullable=True)

    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_finished_at = Column(DateTime(timezone=True), nullable=True)

Index("ix_videos_workspace_status", Video.workspace_id, Video.status)
Index("ix_videos_status", Video.status)
