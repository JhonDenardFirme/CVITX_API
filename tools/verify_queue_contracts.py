#!/usr/bin/env python3
import argparse, json, sys, re
from datetime import datetime

def _is_uuid(s):
    return bool(re.fullmatch(r"[0-9a-fA-F-]{36}", str(s or "")))

def _is_iso_or_null(v):
    if v is None: return True
    s=str(v)
    try:
        # tolerate 'Z'
        if s.endswith('Z'): s=s[:-1] + "+00:00"
        datetime.fromisoformat(s)
        return True
    except Exception:
        return False

def _err(fmt, *a):
    return fmt.format(*a)

def _check_s3_key_raw(key, wid, vid):
    if not isinstance(key, str): return False
    return key.startswith(f"demo_user/{wid}/{vid}/raw/") and key.lower().endswith(".mp4")

def _check_snapshot_s3_url(u, wid, vid):
    # s3://bucket/demo_user/<wid>/<vid>/snapshots/....jpg
    if not isinstance(u, str): return False
    if not u.startswith("s3://"): return False
    parts = u[5:].split("/", 1)
    if len(parts) != 2: return False
    key = parts[1]
    return key.startswith(f"demo_user/{wid}/{vid}/snapshots/") and key.lower().endswith(".jpg")

def validate_process_video(obj):
    errs=[]
    need = ["event","video_id","workspace_id","workspace_code","camera_code","s3_key_raw","frame_stride","recordedAt"]
    for k in need:
        if k not in obj: errs.append(_err("missing field: {}", k))
    if obj.get("event") != "PROCESS_VIDEO":
        errs.append("event must be 'PROCESS_VIDEO'")
    wid, vid = obj.get("workspace_id"), obj.get("video_id")
    if not _is_uuid(wid): errs.append("workspace_id must be UUID")
    if not _is_uuid(vid): errs.append("video_id must be UUID")
    if not isinstance(obj.get("workspace_code"), str): errs.append("workspace_code must be string")
    if not isinstance(obj.get("camera_code"), str): errs.append("camera_code must be string")
    if not _check_s3_key_raw(obj.get("s3_key_raw"), wid, vid):
        errs.append("s3_key_raw must be 'demo_user/<wid>/<vid>/raw/<file>.mp4'")
    fs = obj.get("frame_stride")
    if not (isinstance(fs,int) and fs>=1): errs.append("frame_stride must be int >=1")
    if not _is_iso_or_null(obj.get("recordedAt")): errs.append("recordedAt must be ISO-8601 or null")
    return errs

def validate_snapshot_ready(obj):
    errs=[]
    need = ["event","video_id","workspace_id","workspace_code","camera_code",
            "track_id","snapshot_s3_key","detectedIn","detectedAt","yolo_type"]
    for k in need:
        if k not in obj: errs.append(_err("missing field: {}", k))
    if obj.get("event") != "SNAPSHOT_READY":
        errs.append("event must be 'SNAPSHOT_READY'")
    wid, vid = obj.get("workspace_id"), obj.get("video_id")
    if not _is_uuid(wid): errs.append("workspace_id must be UUID")
    if not _is_uuid(vid): errs.append("video_id must be UUID")
    if not isinstance(obj.get("workspace_code"), str): errs.append("workspace_code must be string")
    if not isinstance(obj.get("camera_code"), str): errs.append("camera_code must be string")
    if not isinstance(obj.get("track_id"), int): errs.append("track_id must be int")
    if not _check_snapshot_s3_url(obj.get("snapshot_s3_key"), wid, vid):
        errs.append("snapshot_s3_key must be s3://<bucket>/demo_user/<wid>/<vid>/snapshots/<file>.jpg")
    di = obj.get("detectedIn")
    if not (isinstance(di,int) and di>=0): errs.append("detectedIn must be int ms >=0")
    if not _is_iso_or_null(obj.get("detectedAt")): errs.append("detectedAt must be ISO-8601 or null")
    if not isinstance(obj.get("yolo_type"), str): errs.append("yolo_type must be string")
    return errs

def main():
    ap = argparse.ArgumentParser(description="Validate CVITX SQS message contracts")
    ap.add_argument("--type", required=True, choices=["PROCESS_VIDEO","SNAPSHOT_READY"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="Inline JSON string")
    src.add_argument("--file", help="Path to JSON file")
    args = ap.parse_args()

    try:
        if args.json:
            obj = json.loads(args.json)
        else:
            with open(args.file, "r", encoding="utf-8") as f:
                obj = json.load(f)
    except Exception as e:
        print("ERR: failed to parse JSON:", e, file=sys.stderr)
        sys.exit(2)

    if args.type == "PROCESS_VIDEO":
        errs = validate_process_video(obj)
    else:
        errs = validate_snapshot_ready(obj)

    if errs:
        print("INVALID:", *errs, sep="\n - ")
        sys.exit(1)
    else:
        print("OK:", args.type, "message is valid")
        sys.exit(0)

if __name__ == "__main__":
    main()
