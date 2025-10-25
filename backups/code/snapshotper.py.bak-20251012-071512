import os, io
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import cv2
from PIL import Image

SNAPSHOT_SIZE = int(os.getenv("SNAPSHOT_SIZE", "640"))
JPG_QUALITY   = int(os.getenv("JPG_QUALITY", "95"))
MIN_TRACK_AGE = int(os.getenv("MIN_TRACK_AGE", "3"))

@dataclass
class BestRec:
    frame_idx: int
    bbox: Tuple[int,int,int,int]
    conf: float
    cls_id: int
    area: float
    center_dist: float
    age: int

class BestFrameKeeper:
    """Prefer larger area → higher conf → closer to center → earlier frame; emit only if age ≥ MIN_TRACK_AGE."""
    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self._best: Dict[int, BestRec] = {}

    def _score(self, bbox, conf):
        x1,y1,x2,y2 = bbox
        area = max(0,x2-x1)*max(0,y2-y1)
        cx,cy=(x1+x2)/2.0,(y1+y2)/2.0
        dx,dy=abs(cx-self.W/2),abs(cy-self.H/2)
        return area, (dx*dx+dy*dy)**0.5

    def consider(self, tid: int, frame_idx: int, bbox, conf: float, cls_id: int):
        area, cd = self._score(bbox, conf)
        cur = self._best.get(tid)
        if cur is None:
            self._best[tid] = BestRec(frame_idx, tuple(map(int,bbox)), float(conf), int(cls_id), area, cd, age=1); return
        cur.age += 1
        cand = BestRec(frame_idx, tuple(map(int,bbox)), float(conf), int(cls_id), area, cd, age=cur.age)
        better = (
            (cand.area > cur.area) or
            (cand.area == cur.area and cand.conf > cur.conf) or
            (cand.area == cur.area and cand.conf == cur.conf and cand.center_dist < cur.center_dist) or
            (cand.area == cur.area and cand.conf == cur.conf and cand.center_dist == cur.center_dist and cand.frame_idx < cur.frame_idx)
        )
        if better: self._best[tid] = cand

    def items_ready(self):
        return [(tid, rec) for tid, rec in self._best.items() if rec.age >= MIN_TRACK_AGE]

def _expand_square(bbox, W, H, margin=0.50):
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1
    side = max(w,h) * (1 + margin*2)
    cx,cy=(x1+x2)/2.0,(y1+y2)/2.0
    x1=int(max(0, cx-side/2)); y1=int(max(0, cy-side/2))
    x2=int(min(W, cx+side/2));  y2=int(min(H, cy+side/2))
    return x1,y1,x2,y2

def _letterbox_640(img_bgr):
    h,w = img_bgr.shape[:2]
    if h==0 or w==0: return img_bgr
    scale = min(SNAPSHOT_SIZE/w, SNAPSHOT_SIZE/h)
    nw,nh=int(w*scale),int(h*scale)
    resized = cv2.resize(img_bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((SNAPSHOT_SIZE,SNAPSHOT_SIZE,3),dtype=np.uint8)
    sx,sy=(SNAPSHOT_SIZE-nw)//2,(SNAPSHOT_SIZE-nh)//2
    canvas[sy:sy+nh, sx:sx+nw] = resized
    return canvas

def save_and_upload_snapshot(frame_bgr, bbox, s3, bucket, key):
    H,W = frame_bgr.shape[:2]
    x1,y1,x2,y2 = _expand_square(bbox, W, H)
    crop = frame_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0: crop = frame_bgr
    boxed = _letterbox_640(crop)
    # JPEG encode and upload
    import io
    img = Image.fromarray(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPG_QUALITY, optimize=True)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="image/jpeg")

def build_snapshot_key(user_id, workspace_id, video_id, workspace_code, camera_code, track_id, offset_ms):
    tid = f"{int(track_id):06d}"
    off = f"{int(offset_ms):06d}"
    fname = f"{workspace_code}_{camera_code}_{tid}_{off}.jpg"
    return f"{user_id}/{workspace_id}/{video_id}/snapshots/{fname}"
