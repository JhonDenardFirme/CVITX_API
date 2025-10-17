import os, time, base64, re
from typing import Tuple
import boto3

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_IMAGE_ANALYSIS_PREFIX","imageanalysis")

s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION","ap-southeast-2"))

def parse_s3_uri(uri: str) -> Tuple[str,str]:
    # s3://bucket/key...
    m = re.match(r'^s3://([^/]+)/(.+)$', uri)
    if not m:
        raise ValueError(f"Bad S3 URI: {uri}")
    return m.group(1), m.group(2)

def s3_get_bytes(uri: str) -> bytes:
    bkt, key = parse_s3_uri(uri)
    r = s3.get_object(Bucket=bkt, Key=key)
    return r["Body"].read()

def s3_put_bytes(bucket: str, key: str, data: bytes, content_type: str="image/jpeg") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type, ACL="private")
