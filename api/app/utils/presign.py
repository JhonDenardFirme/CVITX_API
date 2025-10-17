import os, boto3, botocore
from urllib.parse import urlparse

AWS_REGION    = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
DEFAULT_BUCKET= os.getenv("S3_BUCKET")  or os.getenv("BUCKET")
EXPIRE        = int(os.getenv("S3_PRESIGN_EXPIRE", "900"))

_s3 = boto3.client("s3", region_name=AWS_REGION)

def _split_bucket_key_any(value: str):
    if not value:
        return None, None
    # s3://bucket/key
    if value.startswith("s3://"):
        u = urlparse(value)
        return u.netloc, u.path.lstrip("/")
    # https://bucket.s3.amazonaws.com/key  OR  https://s3.amazonaws.com/bucket/key
    if value.startswith("http://") or value.startswith("https://"):
        u = urlparse(value)
        host = u.netloc
        path = u.path.lstrip("/")
        bucket = None
        if ".s3" in host:                      # virtual-hosted-style
            bucket = host.split(".s3", 1)[0]
        elif host == "s3.amazonaws.com":       # path-style
            if "/" in path:
                bucket, path = path.split("/", 1)
        return bucket, path
    # relative key → use DEFAULT_BUCKET
    return DEFAULT_BUCKET, value.lstrip("/")

def _looks_signed(u: str) -> bool:
    # S3 SigV4 or CloudFront styles
    markers = ("X-Amz-Signature=", "X-Amz-Credential=", "X-Amz-Expires=", "Signature=", "Key-Pair-Id=")
    return any(m in u for m in markers)

def _presign(bucket: str|None, key: str|None) -> str|None:
    if not bucket or not key:
        return None
    try:
        _s3.head_object(Bucket=bucket, Key=key)  # best-effort existence check
    except botocore.exceptions.ClientError:
        pass
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=EXPIRE,
        HttpMethod="GET",
    )

def _ensure_presigned(url_or_key: str|None) -> str|None:
    if not url_or_key:
        return None
    s = str(url_or_key)
    if s.startswith("http"):
        if _looks_signed(s):
            return s
        # unsigned http(s) URL → parse and presign
        b, k = _split_bucket_key_any(s)
        return _presign(b, k)
    if s.startswith("s3://"):
        b, k = _split_bucket_key_any(s)
        return _presign(b, k)
    # relative key
    return _presign(DEFAULT_BUCKET, s.lstrip("/"))

def presign_get(key: str) -> str|None:
    # Prefer legacy helpers, but validate output; fall back to local presign.
    try:
        from app import s3util as _s3util
        if hasattr(_s3util, "presign_get"):
            u = _s3util.presign_get(key)
            u2 = _ensure_presigned(u)
            if u2:
                return u2
    except Exception:
        pass
    try:
        from app.services import s3 as _svc_s3
        if hasattr(_svc_s3, "presign_get"):
            u = _svc_s3.presign_get(key)
            u2 = _ensure_presigned(u)
            if u2:
                return u2
    except Exception:
        pass
    return _ensure_presigned(key)
