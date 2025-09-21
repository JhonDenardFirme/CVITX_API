from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API
    api_key: str = "supersecret"

    # AWS / S3
    aws_region: str = "ap-southeast-2"
    s3_bucket: str
    s3_presign_expire: int = 900

    # DB
    db_url: str

    # SQS (Day-2)
    sqs_video_queue_url: Optional[str] = None
    sqs_snapshot_queue_url: Optional[str] = None

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "https://cvitx.vercel.app"]

    # Settings loader
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",  # future-proof: ignore unexpected .env keys instead of crashing
    )

settings = Settings()

