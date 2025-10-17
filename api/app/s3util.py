import os, re, io, boto3
from typing import Tuple

_s3 = boto3.client("s3", region_name=os.environ["AWS_REGION"])
_BUCKET = os.environ["S3_BUCKET"]
_EXPIRES = int(os.environ.get("S3_PRESIGN_EXPIRE", "900"))

_S3_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")

def split_s3(url_or_key: str) -> Tuple[str, str]:
    """Accept 's3://bucket/key' or 'key' and return (bucket, key)."""
    m = _S3_RE.match(url_or_key or "")
    if m:
        return m.group("bucket"), m.group("key")
    return _BUCKET, (url_or_key or "").lstrip("/")

def presign_get(url_or_key: str) -> str:
    bucket, key = split_s3(url_or_key)
    return _s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=_EXPIRES,
    )

def get_bytes(url_or_key: str) -> bytes:
    bucket, key = split_s3(url_or_key)
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def put_png_bytes(key: str, data: bytes) -> str:
    _s3.put_object(Bucket=_BUCKET, Key=key, Body=data, ContentType="image/png")
    return f"s3://{_BUCKET}/{key}"
