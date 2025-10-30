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
    if not api_key or not endpoint:
        if not __COLOR_WARNED:
            __COLOR_WARNED = True
            _log_json("WARN", "color_adapter_skipped", note="missing COLOR_UTILS_API_KEY or COLOR_UTILS_ENDPOINT")
        return []
    try:
        import requests
        crop = image_pil if veh_box is None else image_pil.crop(tuple(map(int, veh_box)))
        payload = {"image_b64": _b64_jpeg(crop), "max_colors": 3}
        headers = {"X-API-Key": api_key, "Authorization": "Bearer ***"}
        r = requests.post(endpoint, json=payload, headers=headers, timeout=6)
        if r.status_code != 200:
            _log_json("WARN","color_http_error", code=r.status_code); return []
        js = r.json() or {}
        out = []
        for i in (js.get("colors") or [])[:3]:
            name = i.get("name") or i.get("base") or i.get("label")
            if not name: continue
            out.append({
                "name": str(name),
                "fraction": float(i.get("fraction", 0.0)),
                "conf": float(i.get("conf", i.get("confidence", 0.0)))
            })
        return out
    except Exception:
        _log_json("WARN","color_adapter_error")
        return []
def read_plate_text(image_pil: Image.Image, plate_box: Optional[Tuple[float,float,float,float]] = None) -> Dict[str, Any]:
    """
    Tries custom microservice first (PLATE_UTILS_ENDPOINT + X-API-Key), then falls back to Plate Recognizer
    if PLATE_RECOGNITION_API_KEY / PLATE_API_KEY / PLATE_RECOGNIZER_TOKEN is set.
    Returns {"text": str, "conf": float} or {} on failure.
    """
    crop = image_pil if plate_box is None else image_pil.crop(tuple(map(int, plate_box)))

    # (A) Custom microservice
    ep  = os.getenv("PLATE_UTILS_ENDPOINT")
    key = os.getenv("PLATE_UTILS_API_KEY") or os.getenv("OPENAI_API_KEY")
    if ep and key:
        try:
            import requests
            payload = {"image_b64": _b64_jpeg(crop)}
            headers = {"X-API-Key": key, "Authorization": "Bearer ***"}
            r = requests.post(ep, json=payload, headers=headers, timeout=8)
            if r.ok:
                js = r.json() or {}
                text = (js.get("text") or js.get("plate") or "").strip().upper()
                conf = js.get("conf", js.get("confidence"))
                return {"text": text, "conf": float(conf) if conf is not None else 0.0}
        except Exception:
            pass  # fall through to (B)

    # (B) Plate Recognizer direct REST
    token = (os.getenv("PLATE_RECOGNITION_API_KEY")
             or os.getenv("PLATE_API_KEY")
             or os.getenv("PLATE_RECOGNIZER_TOKEN"))
    if token:
        try:
            import requests, io
            buf = io.BytesIO(); crop.convert("RGB").save(buf, "JPEG", quality=95)
            files = {"upload": ("plate.jpg", buf.getvalue())}
            base = os.getenv("PLATE_API_BASE", "https://api.platerecognizer.com/v1")
            hdrs = {"Authorization": f"Token {token}"}
            data = {}
            regions = [r.strip() for r in (os.getenv("PLATE_REGIONS", "ph").split(",")) if r.strip()]
            if regions: data["regions"] = ",".join(regions)
            rr = requests.post(f"{base}/plate-reader/", headers=hdrs, data=data, files=files, timeout=12)
            if rr.ok:
                best = None
                for res in (rr.json().get("results") or []):
                    p = (res.get("plate") or "").upper()
                    s = float(res.get("score") or 0.0)
                    if p and (best is None or s > best[1]): best = (p, s)
                if best: return {"text": best[0], "conf": best[1]}
        except Exception:
            pass

    return {}
