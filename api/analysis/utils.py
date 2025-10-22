import os, sys, json, time, re, base64
from io import BytesIO
from typing import Dict, Tuple, List, Optional
from PIL import Image, ImageDraw

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"

class JsonLogger:
    def __init__(self, name: str = "cvitx", json_logs: Optional[bool] = None, level: str = None):
        self.name = name
        self.json_logs = True if json_logs is None else bool(int(json_logs)) if isinstance(json_logs, (int,str)) else bool(json_logs)
        self.level = (level or os.getenv("LOG_LEVEL","INFO")).upper()

    def _emit(self, level: str, event: str, **fields):
        payload = {
            "ts": _now_iso(),
            "level": level,
            "svc": self.name,
            "event": event,
            **fields
        }
        # Always print JSON; keep it simple/robust
        print(json.dumps(payload), flush=True)

    def info(self, event: str, **fields):  self._emit("INFO",  event, **fields)
    def warn(self, event: str, **fields):  self._emit("WARN",  event, **fields)
    def error(self, event: str, **fields): self._emit("ERROR", event, **fields)

# ---------- ENV GETTERS (support your current BUCKET var as fallback) ----------
def getenv_region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-2"

def getenv_bucket() -> Optional[str]:
    return os.getenv("S3_BUCKET") or os.getenv("BUCKET")

def getenv_prefix() -> str:
    return os.getenv("S3_IMAGE_ANALYSIS_PREFIX", "imageanalysis")

def getenv_bool(key: str, default: bool=False) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","on")

# ---------- S3 HELPERS ----------
_S3_CLIENT = None
def _s3():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        _S3_CLIENT = boto3.client("s3", region_name=getenv_region())
    return _S3_CLIENT

_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")
def parse_s3_uri(uri: str) -> Tuple[str,str]:
    m = _S3_URI_RE.match(uri or "")
    if not m:
        raise ValueError(f"Bad S3 URI: {uri}")
    return m.group(1), m.group(2)

def s3_get_bytes(uri: str) -> bytes:
    bkt, key = parse_s3_uri(uri)
    r = _s3().get_object(Bucket=bkt, Key=key)
    return r["Body"].read()

def s3_put_bytes(bucket: str, key: str, data: bytes, content_type: str="image/jpeg") -> None:
    _s3().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type, ACL="private")

def s3_put_image_jpg(bucket: str, key: str, image_bytes: bytes) -> None:
    s3_put_bytes(bucket, key, image_bytes, "image/jpeg")

# ---------- IMAGE HELPERS ----------
def annotate(image_bytes: bytes, dets: Dict) -> bytes:
    im = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(im)
    boxes: List[Tuple[int,int,int,int]] = dets.get("boxes") or []
    label = f"{dets.get('make','?')} {dets.get('model','?')} ({dets.get('type','?')})"
    for (x1,y1,x2,y2) in boxes:
        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
        draw.text((x1+4, max(0,y1-14)), label, fill="red")
    out = BytesIO(); im.save(out, format="JPEG", quality=90)
    return out.getvalue()

def crop_by_xyxy(image_bytes: bytes, box: Tuple[int,int,int,int]) -> bytes:
    im = Image.open(BytesIO(image_bytes)).convert("RGB")
    x1,y1,x2,y2 = box
    x1,y1,x2,y2 = map(int, (max(0,x1), max(0,y1), min(im.width, x2), min(im.height, y2)))
    crop = im.crop((x1,y1,x2,y2))
    out = BytesIO(); crop.save(out, format="JPEG", quality=90)
    return out.getvalue()
# ===================== Adapters: Color & Plate (env-gated, fault-tolerant) =====================
import os, io, json, base64, time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

__COLOR_WARNED = False
__PLATE_WARNED = False

def _log_json(level: str, event: str, **kw):
    try:
        from datetime import datetime, timezone
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z"),
               "level": level, "svc":"cvitx-engine", "event": event}
        rec.update(kw); print(json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass

def _b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode("ascii")

def detect_vehicle_color(image_pil: Image.Image, veh_box: Optional[Tuple[float,float,float,float]] = None) -> List[Dict[str, Any]]:
    global __COLOR_WARNED
    api_key = os.getenv("COLOR_UTILS_API_KEY") or os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("COLOR_UTILS_ENDPOINT")
    if not api_key:
        if not __COLOR_WARNED:
            __COLOR_WARNED = True
            _log_json("WARN", "color_missing_key", note="ENABLE_COLOR=1 but no COLOR_UTILS_API_KEY/OPENAI_API_KEY provided")
        return []
    if not endpoint:
        if not __COLOR_WARNED:
            __COLOR_WARNED = True
            _log_json("WARN", "color_endpoint_unset", note="No COLOR_UTILS_ENDPOINT set; skipping color stage")
        return []
    try:
        import requests
        payload = {
            "image_b64": _b64_jpeg(image_pil if veh_box is None else image_pil.crop(tuple(map(int, veh_box)))),
            "veh_box": [float(v) for v in veh_box] if veh_box else None,
            "max_colors": 3,
        }
        headers = {"Authorization": "Bearer ***", "X-API-Key": api_key}
        for _ in range(2):
            try:
                r = requests.post(endpoint, json=payload, headers=headers, timeout=6)
                if r.status_code == 200:
                    js = r.json()
                    out = []
                    for i in js.get("colors", [])[:3]:
                        out.append({
                            "base": i.get("base") or i.get("label"),
                            "finish": i.get("finish"),
                            "lightness": i.get("lightness"),
                            "conf": i.get("conf"),
                            "fraction": i.get("fraction")
                        })
                    return out
                time.sleep(0.3)
            except Exception:
                time.sleep(0.2)
        _log_json("WARN","color_timeout_or_error", note=f"http error or timeout at {endpoint}")
        return []
    except Exception:
        _log_json("WARN","color_adapter_error")
        return []

def read_plate_text(image_pil: Image.Image, plate_box: Optional[Tuple[float,float,float,float]] = None) -> Dict[str, Any]:
    global __PLATE_WARNED
    api_key = os.getenv("PLATE_UTILS_API_KEY") or os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("PLATE_UTILS_ENDPOINT")
    if not api_key:
        if not __PLATE_WARNED:
            __PLATE_WARNED = True
            _log_json("WARN", "plate_missing_key", note="ENABLE_PLATE=1 but no PLATE_UTILS_API_KEY/OPENAI_API_KEY provided")
        return {}
    if not endpoint:
        if not __PLATE_WARNED:
            __PLATE_WARNED = True
            _log_json("WARN", "plate_endpoint_unset", note="No PLATE_UTILS_ENDPOINT set; skipping plate stage")
        return {}
    try:
        import requests
        crop = image_pil if plate_box is None else image_pil.crop(tuple(map(int, plate_box)))
        payload = {"image_b64": _b64_jpeg(crop)}
        headers = {"Authorization": "Bearer ***", "X-API-Key": api_key}
        for _ in range(2):
            try:
                r = requests.post(endpoint, json=payload, headers=headers, timeout=6)
                if r.status_code == 200:
                    js = r.json()
                    text = (js.get("text") or "").strip()
                    conf = js.get("conf")
                    return {"text": text, "conf": conf}
                time.sleep(0.3)
            except Exception:
                time.sleep(0.2)
        _log_json("WARN","plate_timeout_or_error", note=f"http error or timeout at {endpoint}")
        return {}
    except Exception:
        _log_json("WARN","plate_adapter_error")
        return {}
