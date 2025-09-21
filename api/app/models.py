import enum, uuid
from datetime import datetime
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, TIMESTAMP, Enum, Integer, Text, ForeignKey

Base = declarative_base()

class VideoStatus(str, enum.Enum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"

class Workspace(Base):
    __tablename__ = "workspaces"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    workspace_code: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

class Video(Base):
    __tablename__ = "videos"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id"), nullable=False)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    camera_label: Mapped[str] = mapped_column(String, nullable=False)
    camera_code: Mapped[str] = mapped_column(String, nullable=False)  # CAM1...
    recorded_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    s3_key_raw: Mapped[str] = mapped_column(Text, nullable=False)
    frame_stride: Mapped[int] = mapped_column(Integer, default=3)
    status: Mapped[VideoStatus] = mapped_column(Enum(VideoStatus), default=VideoStatus.uploaded)
    error_msg: Mapped[str | None] = mapped_column(Text)
