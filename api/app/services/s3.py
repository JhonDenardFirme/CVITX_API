import boto3
from ..config import settings

# Locally: uses your AWS creds/profile. On EC2: uses the instance role.
s3 = boto3.client("s3", region_name=settings.aws_region)

def make_raw_key(user_id: str, workspace_id: str, video_id: str, file_name: str) -> str:
    return f"{user_id}/{workspace_id}/{video_id}/raw/{file_name}"

def presign_put(key: str, expires: int, content_type: str = "video/mp4") -> str:
    # Important: include ContentType here so the browser can PUT with the same header
    return s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": settings.s3_bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=expires,
        HttpMethod="PUT",
    )
