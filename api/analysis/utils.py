import os, sys, json, time, re, base64, io
from io import BytesIO
from typing import Dict, Tuple, List, Optional, Any
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
        print(json.dumps(payload), flush=True)

    def info(self, event: str, **fields):  self._emit("INFO",  event, **fields)
    def warn(self, event: str, **fields):  self._emit("WARN",  event, **fields)
    def error(self, event: str, **fields): self._emit("ERROR", event, **fields)

# ---------- ENV HELPERS ----------
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

def _http_timeout() -> float:
    """Read timeout from COLOR_TIMEOUT env (default 30s). Safe for use anywhere."""
    try:
        return float(os.getenv("COLOR_TIMEOUT", "30"))
    except Exception:
        return 30.0

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
__COLOR_WARNED = False
__PLATE_WARNED = False

def _log_json(level: str, event: str, **kw):
    try:
        from datetime import datetime, timezone
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z"),
            "level": level, "svc":"cvitx-engine", "event": event
        }
        rec.update(kw); print(json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass

def _b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def detect_vehicle_color(image_pil: Image.Image, veh_box: Optional[Tuple[float,float,float,float]] = None) -> List[Dict[str, Any]]:
    """
    Tolerant adapter:
      A) Microservice: COLOR_UTILS_ENDPOINT (+ optional COLOR_UTILS_API_KEY)
      B) Direct OpenAI (Colab-style): COLOR_PROVIDER + COLOR_MODEL_ID + OPENAI_API_KEY
    Returns ≤3 colors normalized to: {"finish","base","lightness","conf"}.
    """
    # local helpers
    def _b64(img: Image.Image) -> str:
        buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    _FIN  = {"metallic","matte","glossy"}
    _LUX  = {"light","dark"}
    _BASE = {"red","orange","yellow","green","blue","purple","pink","white","gray","black","silver","gold","brown","beige","maroon","cyan"}

    def _parse_label(lbl: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        toks = [t.strip() for t in (lbl or "").replace(","," ").split() if t.strip()]
        fin = base = lux = None
        for t in toks:
            tl = t.lower()
            if tl in _FIN and fin is None: fin = t.title()
            elif tl in _BASE and base is None: base = t.title()
            elif tl in _LUX and lux is None: lux = t.title()
        if base is None:
            for t in toks:
                tl = t.lower()
                if tl not in _FIN and tl not in _LUX:
                    base = t.title(); break
        return fin, base, lux

    def _normalize(js: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Tuple[Optional[str],Optional[str],Optional[str],float,float]] = []
        def _push(fin, base, lux, conf, frac):
            try: c = float(conf) if conf is not None else 0.0
            except Exception: c = 0.0
            try: f = float(frac) if frac is not None else 0.0
            except Exception: f = 0.0
            items.append((fin, base, lux, c, f))

        cols = js.get("colors")
        if isinstance(cols, list) and cols:
            for i in cols:
                if isinstance(i, dict):
                    fin = i.get("finish")
                    base = i.get("base") or i.get("name") or i.get("label")
                    lux  = i.get("lightness")
                    if not fin or not base or not lux:
                        fin2, base2, lux2 = _parse_label(str(base or ""))
                        fin = fin or fin2; base = base2 or base; lux = lux or lux2
                    conf = i.get("conf", i.get("confidence"))
                    frac = i.get("fraction", i.get("coverage"))
                    _push(fin, base.title() if base else None, lux, conf, frac)

        elif isinstance(js.get("labels"), list):
            labels = js.get("labels") or []
            fracs  = js.get("fractions") or [1.0] * len(labels)
            confs  = js.get("color_confidences") or [js.get("confidence", 0.0)] * len(labels)
            for idx, lbl in enumerate(labels[:3]):
                fin, base, lux = _parse_label(str(lbl))
                conf = confs[idx] if idx < len(confs) else js.get("confidence")
                frac = fracs[idx] if idx < len(fracs) else 0.0
                _push(fin, base, lux, conf, frac)

        items.sort(key=lambda t: (t[4], t[3]), reverse=True)
        out: List[Dict[str,Any]] = []
        for fin, base, lux, c, _f in items[:3]:
            out.append({"finish": fin or None, "base": base or None, "lightness": lux or None, "conf": float(c)})
        return out

    crop = image_pil if veh_box is None else image_pil.crop(tuple(map(int, veh_box)))

    endpoint   = os.getenv("COLOR_UTILS_ENDPOINT")
    x_api_key  = os.getenv("COLOR_UTILS_API_KEY")
    provider   = os.getenv("COLOR_PROVIDER") or ""
    model_id   = os.getenv("COLOR_MODEL_ID") or ""
    openai_key = os.getenv("OPENAI_API_KEY")

    # (A) microservice path if endpoint present
    if endpoint:
        headers = {"Content-Type":"application/json"}
        if x_api_key: headers["X-API-Key"] = x_api_key
        if provider and model_id and openai_key:
            payload = {
                "mode": "dominant_colors",
                "provider": provider,
                "model": model_id,
                "prompt": (
                    "You are 'Color Detector AI' for VEHICLE BODY PAINT.\n\n"
                    "SCOPE (IMPORTANT): analyze BODY PAINT ONLY; exclude glass/lights/plates/chrome/wheels/tires/sky/reflections.\n"
                    "TASK: choose N∈{1,2,3}; labels=[Finish][Base][Lightness]; fractions sum to 1.0; sort by coverage.\n"
                    "Return JSON: {\"labels\":[],\"fractions\":[],\"color_confidences\":[],\"pixels\":0,\"confidence\":0.0,\"stainless\":false,\"body_paint_only\":true}\n"
                ),
                "image_b64": _b64(crop),
                "max_colors": 3,
                "response_shape": "colors_v1"
            }
            headers["Authorization"] = f"Bearer {openai_key}"
        else:
            payload = {"image_b64": _b64(crop), "max_colors": 3}

        try:
            import requests
            r = requests.post(endpoint, json=payload, headers=headers, timeout=_http_timeout())
            if not r.ok:
                _log_json("WARN","color_http_error", code=r.status_code)
                return []
            return _normalize(r.json() or {})
        except Exception as e:
            _log_json("WARN","color_adapter_error", err=str(e))
            return []

    # (B) direct OpenAI (no endpoint) if model & key exist
    if provider and model_id and openai_key:
        data_url = "data:image/jpeg;base64," + _b64(crop)
        COLOR_SCHEMA = {
            "name": "VehicleBodyColorStructuredV2",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "labels":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"string"}},
                    "fractions":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"number","minimum":0.0,"maximum":1.0}},
                    "color_confidences":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"number","minimum":0.0,"maximum":1.0}},
                    "pixels":{"type":"integer","minimum":1},
                    "confidence":{"type":"number","minimum":0.0,"maximum":1.0},
                    "stainless":{"type":"boolean"},
                    "body_paint_only":{"type":"boolean"}
                },
                "required":["labels","fractions","color_confidences","pixels","confidence","stainless","body_paint_only"]
            }
        }
        COLOR_INSTRUCTIONS = (
            "You are 'Color Detector AI' for VEHICLE BODY PAINT.\n\n"
            "SCOPE (IMPORTANT): analyze BODY PAINT ONLY (painted panels). EXCLUDE glass/lights/plates/chrome/wheels/tires/reflections.\n"
            "TASK: N∈{1,2,3}; labels=[Finish][Base][Lightness]; Base in {Red,Orange,Yellow,Green,Blue,Purple,Pink,White,Gray,Black,Silver,Gold,Brown,Beige,Maroon,Cyan}; fractions sum to 1.0\n"
            "OUTPUT JSON ONLY with keys: labels, fractions, color_confidences, pixels, confidence, stainless, body_paint_only.\n"
        )

        payload = {
            "model": model_id,
            "instructions": COLOR_INSTRUCTIONS,
            "input": [{
                "role": "user",
                "content": [
                    {"type":"input_text","text":"Analyze the BODY PAINT color(s). Return JSON only."},
                    {"type":"input_image","image_url": data_url}
                ],
            }],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": COLOR_SCHEMA["name"],
                    "strict": True,
                    "schema": COLOR_SCHEMA["schema"],
                }
            }
        }
        try:
            import requests, json as _j
            h = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            r = requests.post("https://api.openai.com/v1/responses", json=payload, headers=h, timeout=_http_timeout())
            if not r.ok:
                _log_json("WARN","color_openai_http_error", code=r.status_code)
                return []
            js = r.json() or {}
            parsed = None
            try:
                out = js.get("output") or []
                if isinstance(out, list) and out:
                    content = out[0].get("content") or []
                    if content and isinstance(content, list):
                        parsed = content[0].get("parsed")
                if parsed is None and "output_text" in js:
                    parsed = _j.loads(js.get("output_text","{}"))
            except Exception as e:
                _log_json("WARN","color_openai_parse_error", err=str(e))
            if not parsed:
                return []
            return _normalize(parsed)
        except Exception as e:
            _log_json("WARN","color_openai_error", err=str(e))
            return []

    _log_json("WARN","color_adapter_skipped", note="no endpoint and no OPENAI model/key")
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
            r = requests.post(ep, json=payload, headers=headers, timeout=_http_timeout())
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
            import requests
            buf = io.BytesIO(); crop.convert("RGB").save(buf, "JPEG", quality=95)
            files = {"upload": ("plate.jpg", buf.getvalue())}
            base = os.getenv("PLATE_API_BASE", "https://api.platerecognizer.com/v1")
            hdrs = {"Authorization": f"Token {token}"}
            data = {}
            regions = [r.strip() for r in (os.getenv("PLATE_REGIONS", "ph").split(",")) if r.strip()]
            if regions: data["regions"] = ",".join(regions)
            rr = requests.post(f"{base}/plate-reader/", headers=hdrs, data=data, files=files, timeout=_http_timeout())
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
