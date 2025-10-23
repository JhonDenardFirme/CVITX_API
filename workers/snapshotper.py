import os, io, math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
from PIL import Image

# ======= Tunables (env-driven) =======
SNAPSHOT_SIZE                = int(os.getenv("SNAPSHOT_SIZE", "640"))
JPG_QUALITY                  = int(os.getenv("JPG_QUALITY", "95"))
MIN_TRACK_AGE                = int(os.getenv("MIN_TRACK_AGE", "3"))

# Crop margins / quality guards
SNAPSHOT_MARGIN              = float(os.getenv("SNAPSHOT_MARGIN", "0.15"))
SNAPSHOT_MARGIN_MIN          = float(os.getenv("SNAPSHOT_MARGIN_MIN", "0.08"))
SNAPSHOT_MARGIN_MAX          = float(os.getenv("SNAPSHOT_MARGIN_MAX", "0.20"))
SNAPSHOT_MIN_SIDE            = int(os.getenv("SNAPSHOT_MIN_SIDE", "32"))

# Neighbor suppression
SNAPSHOT_NEIGHBOR_IOU        = float(os.getenv("SNAPSHOT_NEIGHBOR_IOU", "0.03"))
SNAPSHOT_MARGIN_SHRINK_STEP  = float(os.getenv("SNAPSHOT_MARGIN_SHRINK_STEP", "0.8"))
SNAPSHOT_MARGIN_MAX_STEPS    = int(os.getenv("SNAPSHOT_MARGIN_MAX_STEPS", "5"))

# Light unsharp mask when upscaling
SNAPSHOT_SHARPEN             = int(os.getenv("SNAPSHOT_SHARPEN", "1"))
SNAPSHOT_SHARPEN_AMOUNT      = float(os.getenv("SNAPSHOT_SHARPEN_AMOUNT", "1.2"))
SNAPSHOT_SHARPEN_RADIUS      = float(os.getenv("SNAPSHOT_SHARPEN_RADIUS", "1.0"))
SNAPSHOT_SHARPEN_THRESHOLD   = int(os.getenv("SNAPSHOT_SHARPEN_THRESHOLD", "3"))

# Optional super-light SR (guarded; best effort)
SNAPSHOT_SR                  = int(os.getenv("SNAPSHOT_SR", "0"))
SNAPSHOT_SR_MODEL            = os.getenv("SNAPSHOT_SR_MODEL", "")
SNAPSHOT_SR_UPSCALE          = int(os.getenv("SNAPSHOT_SR_UPSCALE", "2"))

# ======= Data structures =======
@dataclass
class BestRec:
    frame_idx: int
    bbox: Tuple[int,int,int,int]  # tlbr
    conf: float
    cls_id: int
    area: float
    center_dist: float
    sharp: float                   # Laplacian variance (estimated in pass-1)
    age: int

class BestFrameKeeper:
    """
    Prefer larger area → higher conf → higher sharpness → closer to center → earlier frame;
    Emit only if age ≥ MIN_TRACK_AGE and min side ≥ SNAPSHOT_MIN_SIDE.
    """
    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self._best: Dict[int, BestRec] = {}

    def _score(self, bbox):
        x1,y1,x2,y2 = bbox
        area = max(0,x2-x1)*max(0,y2-y1)
        cx,cy=(x1+x2)/2.0,(y1+y2)/2.0
        dx,dy=abs(cx-self.W/2),abs(cy-self.H/2)
        return area, (dx*dx+dy*dy)**0.5

    def consider(self, tid: int, frame_idx: int, bbox, conf: float, cls_id: int, sharp: float):
        area, cd = self._score(bbox)
        cur = self._best.get(tid)
        if cur is None:
            self._best[tid] = BestRec(frame_idx, tuple(map(int,bbox)), float(conf), int(cls_id), area, cd, float(sharp), age=1); return
        cur.age += 1
        cand = BestRec(frame_idx, tuple(map(int,bbox)), float(conf), int(cls_id), area, cd, float(sharp), age=cur.age)
        better = (
            (cand.area > cur.area) or
            (cand.area == cur.area and cand.conf > cur.conf) or
            (cand.area == cur.area and cand.conf == cur.conf and cand.sharp > cur.sharp) or
            (cand.area == cur.area and cand.conf == cur.conf and cand.sharp == cur.sharp and cand.center_dist < cur.center_dist) or
            (cand.area == cur.area and cand.conf == cur.conf and cand.sharp == cur.sharp and cand.center_dist == cur.center_dist and cand.frame_idx < cur.frame_idx)
        )
        if better: self._best[tid] = cand

    def items_ready(self) -> List[Tuple[int, "BestRec"]]:
        out = []
        for tid, rec in self._best.items():
            if rec.age < MIN_TRACK_AGE:
                continue
            x1,y1,x2,y2 = rec.bbox
            w, h = max(0, x2-x1), max(0, y2-y1)
            if min(w, h) < SNAPSHOT_MIN_SIDE:
                continue
            out.append((tid, rec))
        return out

# ======= Geometry & quality helpers =======
def _expand_square(bbox, W, H, margin: float):
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1
    side = max(w,h) * (1 + margin*2)
    cx,cy=(x1+x2)/2.0,(y1+y2)/2.0
    x1=int(max(0, cx-side/2)); y1=int(max(0, cy-side/2))
    x2=int(min(W, cx+side/2));  y2=int(min(H, cy+side/2))
    return x1,y1,x2,y2

def _rect_iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    x1,y1 = max(ax1,bx1), max(ay1,by1)
    x2,y2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,x2-x1), max(0,y2-y1)
    inter = iw*ih
    if inter <= 0: return 0.0
    areaA = max(1, (ax2-ax1)) * max(1, (ay2-ay1))
    areaB = max(1, (bx2-bx1)) * max(1, (by2-by1))
    return inter / float(areaA + areaB - inter + 1e-9)

def choose_margin_for_neighbors(
    bbox: Tuple[int,int,int,int],
    neighbors: List[Tuple[int,int,int,int]],
    W: int, H: int,
    base_margin: Optional[float] = None,
    min_margin: Optional[float] = None,
    max_margin: Optional[float] = None,
    neighbor_iou: Optional[float] = None,
    shrink_step: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> float:
    m  = base_margin  if base_margin  is not None else SNAPSHOT_MARGIN
    mN = min_margin   if min_margin   is not None else SNAPSHOT_MARGIN_MIN
    mX = max_margin   if max_margin   is not None else SNAPSHOT_MARGIN_MAX
    thr= neighbor_iou if neighbor_iou is not None else SNAPSHOT_NEIGHBOR_IOU
    step = shrink_step if shrink_step is not None else SNAPSHOT_MARGIN_SHRINK_STEP
    steps = max_steps if max_steps is not None else SNAPSHOT_MARGIN_MAX_STEPS
    m = max(mN, min(mX, m))
    for _ in range(max(1, steps)):
        sx1,sy1,sx2,sy2 = _expand_square(bbox, W, H, margin=m)
        if all(_rect_iou((sx1,sy1,sx2,sy2), nb) <= thr for nb in (neighbors or [])):
            return m
        m = max(mN, m * step)
        if m <= mN: return m
    return m

def _unsharp(img_bgr: np.ndarray, amount=SNAPSHOT_SHARPEN_AMOUNT, radius=SNAPSHOT_SHARPEN_RADIUS, thr=SNAPSHOT_SHARPEN_THRESHOLD):
    if amount <= 0:
        return img_bgr
    blurred = cv2.GaussianBlur(img_bgr, (0,0), radius)
    diff = cv2.absdiff(img_bgr, blurred)
    if thr > 0:
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > thr
        mask = np.repeat(mask[:, :, None], 3, axis=2)
    else:
        mask = np.ones_like(img_bgr, dtype=bool)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)
    out = img_bgr.copy()
    out[mask] = sharp[mask]
    return np.clip(out, 0, 255).astype(np.uint8)

# Optional SR (best effort)
_sr_ready = False
_sr_engine = None
def _sr_try(img_bgr: np.ndarray) -> np.ndarray:
    global _sr_ready, _sr_engine
    if not SNAPSHOT_SR:
        return img_bgr
    try:
        if not _sr_ready:
            from cv2 import dnn_superres
            _sr_engine = dnn_superres.DnnSuperResImpl_create()
            if SNAPSHOT_SR_MODEL and os.path.isfile(SNAPSHOT_SR_MODEL):
                # Guess model type from filename; fallback to EDSR
                model = "edsr"
                if "espcn" in SNAPSHOT_SR_MODEL.lower(): model="espcn"
                elif "fsrcnn" in SNAPSHOT_SR_MODEL.lower(): model="fsrcnn"
                elif "lapsrn" in SNAPSHOT_SR_MODEL.lower(): model="lapsrn"
                _sr_engine.readModel(SNAPSHOT_SR_MODEL)
                _sr_engine.setModel(model, max(2, min(4, SNAPSHOT_SR_UPSCALE)))
                _sr_ready = True
            else:
                _sr_ready = False
        if _sr_ready and _sr_engine is not None:
            return _sr_engine.upsample(img_bgr)
        # Fallback: simple cubic upscale
        f = max(2, min(4, SNAPSHOT_SR_UPSCALE))
        h,w = img_bgr.shape[:2]
        return cv2.resize(img_bgr, (w*f, h*f), interpolation=cv2.INTER_CUBIC)
    except Exception:
        return img_bgr

def _letterbox_640_qa(img_bgr: np.ndarray) -> np.ndarray:
    """
    Quality-aware letterbox to 640x640:
      - If scaling DOWN -> INTER_AREA
      - If scaling UP   -> INTER_CUBIC (+ optional SR, then unsharp)
    """
    h,w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((SNAPSHOT_SIZE,SNAPSHOT_SIZE,3),dtype=np.uint8)

    # If very small, optionally pre-upscale a bit
    if SNAPSHOT_SR and min(h,w) < 96:
        img_bgr = _sr_try(img_bgr)

    scale = min(SNAPSHOT_SIZE / float(w), SNAPSHOT_SIZE / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
    if scale > 1.0 and SNAPSHOT_SHARPEN:
        resized = _unsharp(resized)
    canvas = np.zeros((SNAPSHOT_SIZE, SNAPSHOT_SIZE, 3), dtype=np.uint8)
    sx, sy = (SNAPSHOT_SIZE - nw)//2, (SNAPSHOT_SIZE - nh)//2
    canvas[sy:sy+nh, sx:sx+nw] = resized
    return canvas

def save_and_upload_snapshot(frame_bgr, bbox, s3, bucket, key, margin: float | None = None):
    H,W = frame_bgr.shape[:2]
    m = margin if margin is not None else SNAPSHOT_MARGIN
    x1,y1,x2,y2 = _expand_square(bbox, W, H, margin=m)
    crop = frame_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = frame_bgr
    boxed = _letterbox_640_qa(crop)

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
