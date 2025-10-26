# >>> paste exactly the content you showed (it already says "api/analysis/color_plate_utils.py") <<<
# api/analysis/color_plate_utils.py
# Modern, env-driven utilities for Vehicle Color (ChatGPT/OpenAI or HTTP gateway)
# and Plate OCR (Plate Recognizer or HTTP gateway). No hardcoded secrets.
# Compatible with AWS EC2 workers and your existing inference runner.

from __future__ import annotations

import os
import io
import json
import time
import base64
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# ============================== JSON logger ==============================

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"

def _jlog(level: str, event: str, **fields) -> None:
    try:
        rec = {"ts": _now_iso(), "level": level, "svc": "cvitx-utils", "event": event}
        rec.update(fields)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    except Exception:
        # never crash due to logging
        pass

# ============================== common helpers ==============================

_BASE_COLORS = {
    "RED","ORANGE","YELLOW","GREEN","BLUE","PURPLE","PINK",
    "WHITE","GRAY","GREY","BLACK","SILVER","GOLD","BROWN","BEIGE","MAROON","CYAN"
}
_FINISH_WORDS = {"METALLIC","MATTE","GLOSSY","GLOSS"}
_LIGHTNESS_WORDS = {"LIGHT","DARK"}

def _pil_crop_xyxy(img: Image.Image, box: Tuple[float,float,float,float]) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    x1 = max(0, min(w-1, float(x1))); y1 = max(0, min(h-1, float(y1)))
    x2 = max(x1+1, min(w,   float(x2))); y2 = max(y1+1, min(h,   float(y2)))
    return img.crop((int(x1), int(y1), int(x2), int(y2)))

def _to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def _to_data_url_jpeg(img: Image.Image, quality: int = 92) -> str:
    b = _to_jpeg_bytes(img, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")

def _renorm(fs: List[float]) -> List[float]:
    s = float(sum(max(0.0, f) for f in fs))
    if s <= 1e-9:
        return [1.0] + [0.0]*(len(fs)-1)
    return [float(max(0.0, f))/s for f in fs]

def _parse_color_label(label: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a label like "Metallic Gray Dark" into (finish, base, lightness).
    If parts are missing, return None for that part.
    """
    if not label:
        return None, None, None
    toks = [t.strip().upper() for t in label.replace("/", " ").split() if t.strip()]
    finish = next((t for t in toks if t in _FINISH_WORDS), None)
    light  = next((t for t in toks if t in _LIGHTNESS_WORDS), None)
    base   = next((t for t in toks if t in _BASE_COLORS), None)
    # normalize variants
    if base == "GREY": base = "GRAY"
    if finish == "GLOSS": finish = "GLOSSY"
    return finish.title() if finish else None, base.title() if base else None, light.title() if light else None

# ============================== COLOR: provider selection ==============================

def _color_provider() -> str:
    # explicit selection wins
    p = (os.getenv("COLOR_PROVIDER") or "").strip().lower()
    if p in {"chatgpt","openai"}:   return "chatgpt"
    if p in {"gateway","internal"}: return "gateway"
    # auto-detect
    if os.getenv("OPENAI_API_KEY"):             return "chatgpt"
    if os.getenv("COLOR_UTILS_ENDPOINT"):       return "gateway"
    return "unset"

def _color_via_openai(image_pil: Image.Image,
                      veh_bbox_orig: Optional[Tuple[float,float,float,float]],
                      allow_multitone: bool,
                      timeout_s: int) -> List[Dict[str, Any]]:
    try:
        # lazy import to avoid hard dependency if you use gateway
        from openai import OpenAI
    except Exception as e:
        _jlog("WARN", "color_openai_import_error", error=str(e))
        return []

    model = os.getenv("COLOR_MODEL_ID", "gpt-5-mini")
    # Build instructions + schema (kept close to your Colab cell)
    _BASE = ["Red","Orange","Yellow","Green","Blue","Purple","Pink","White","Gray","Black","Silver","Gold","Brown","Beige","Maroon","Cyan"]
    instructions = (
        "You are 'Color Detector AI' for VEHICLE BODY PAINT.\n\n"
        "SCOPE (IMPORTANT):\n"
        "- Analyze the BODY PAINT ONLY: painted panels, roof, hood, trunk, doors, pillars, factory accents if painted.\n"
        "- EXCLUDE: glass/windows, lights, license plates, chrome/metal grille, wheels/tires, mirrors' reflective glass, sky/road/reflections/shadows/stickers/dirt.\n\n"
        "TASK:\n"
        "1) Decide the number of PROMINENT body-paint colors N ∈ {1,2,3}.\n"
        "   - Start with 1 (most common case).\n"
        "   - Include a 2nd color ONLY if you are ≥0.70 certain it’s actually part of the body paint (not reflections or non-paint parts).\n"
        "   - Include a 3rd color ONLY if you are ≥0.70 certain.\n"
        "   - Do NOT exceed 3.\n\n"
        "2) For each chosen color, produce a descriptor with up to 3 parts: [Finish] [Base] [Lightness]\n"
        "   - Finish ∈ {Metallic, Matte, Glossy} (omit if unclear)\n"
        f"   - Base   ∈ {{{', '.join(_BASE)}}} (REQUIRED)\n"
        "   - Lightness ∈ {Light, Dark} (omit if unclear)\n"
        '   - Examples: "Metallic Gray Dark", "White", "Glossy Red", "Beige Light", "Gray Dark"\n\n'
        "3) Provide fractional coverage per color (fractions sum to 1.0). Sort colors by coverage (largest first).\n\n"
        "CONFIDENCE:\n"
        "- Provide an overall confidence 0..1 for your BODY-PAINT color decision (not segmentation).\n"
        "- Also provide per-color confidences (0..1) indicating your certainty that each color is truly in the body paint.\n"
        "- Be conservative: lower confidence when heavy reflections, shadows, glare, poor lighting, compression artifacts, or ambiguous two-tone paint.\n\n"
        "OUTPUT RULES:\n"
        "- Return JSON ONLY using the provided schema.\n"
        "- If uncertain about additional colors (<0.70), collapse to a single dominant color."
    )
    schema = {
        "name": "VehicleBodyColorStructuredV2",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "labels": {"type":"array","minItems":1,"maxItems":3,"items":{"type":"string"}},
                "fractions": {"type":"array","minItems":1,"maxItems":3,"items":{"type":"number","minimum":0.0,"maximum":1.0}},
                "color_confidences": {"type":"array","minItems":1,"maxItems":3,"items":{"type":"number","minimum":0.0,"maximum":1.0}},
                "pixels": {"type":"integer","minimum":1},
                "confidence": {"type":"number","minimum":0.0,"maximum":1.0},
                "stainless": {"type":"boolean"},
                "body_paint_only": {"type":"boolean"}
            },
            "required": ["labels","fractions","color_confidences","pixels","confidence","stainless","body_paint_only"]
        }
    }

    img = _pil_crop_xyxy(image_pil, veh_bbox_orig) if veh_bbox_orig else image_pil
    data_url = _to_data_url_jpeg(img, quality=92)

    if not allow_multitone:
        instructions += "\nReturn exactly ONE dominant body-paint color (labels length = 1, fractions = [1.0])."

    client = OpenAI()
    try:
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze the BODY PAINT color(s). Return JSON only."},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            text={"format":{"type":"json_schema","name":schema["name"],"strict":schema["strict"],"schema":schema["schema"]}},
            timeout=timeout_s,
        )
    except Exception as e:
        _jlog("WARN","color_openai_call_error", error=str(e))
        return []

    # Parse structured output
    parsed = None
    try:
        parsed = resp.output[0].content[0].parsed
    except Exception:
        try:
            parsed = json.loads(resp.output_text)
        except Exception:
            try:
                parsed = json.loads(resp.output[0].content[0].text)
            except Exception as e:
                _jlog("WARN","color_openai_parse_error", error=str(e))
                return []

    labels = list(parsed.get("labels", []))
    fracs  = list(parsed.get("fractions", []))
    confs  = list(parsed.get("color_confidences", []))

    # alignment + conservative gating (≥0.70 for 2nd/3rd)
    if not labels:
        labels, fracs, confs = ["Unknown"], [1.0], [0.6]
    if len(fracs) != len(labels): fracs = [1.0] + [0.0]*(len(labels)-1)
    if len(confs) != len(labels): confs = [min(0.98, 0.6 + 0.4*f) for f in fracs]

    # sort by fraction
    order = list(range(len(labels)))
    order.sort(key=lambda i: fracs[i], reverse=True)
    labels = [labels[i] for i in order]
    fracs  = [fracs[i]  for i in order]
    confs  = [confs[i]  for i in order]

    keep = [0]
    if len(labels) > 1 and confs[1] >= 0.70: keep.append(1)
    if len(labels) > 2 and confs[2] >= 0.70: keep.append(2)
    labels = [labels[i] for i in keep][:3]
    fracs  = [fracs[i]  for i in keep][:3]
    confs  = [confs[i]  for i in keep][:3]
    fracs  = _renorm(fracs)

    # normalize output list
    out: List[Dict[str, Any]] = []
    for lbl, frac, c in zip(labels, fracs, confs):
        finish, base, light = _parse_color_label(lbl)
        item = {
            "label": lbl,                 # full label string (for display/debug)
            "base": base,                 # normalized components (for DB/UI)
            "finish": finish,
            "lightness": light,
            "fraction": float(frac),
            "conf": float(max(0.0, min(1.0, c))),
        }
        out.append(item)
    return out

def _color_via_gateway(image_pil: Image.Image,
                       veh_bbox_orig: Optional[Tuple[float,float,float,float]],
                       timeout_s: int) -> List[Dict[str, Any]]:
    endpoint = os.getenv("COLOR_UTILS_ENDPOINT")
    api_key  = os.getenv("COLOR_UTILS_API_KEY")
    if not endpoint or not api_key:
        _jlog("WARN","color_gateway_env_missing", endpoint=bool(endpoint), api_key=bool(api_key))
        return []
    try:
        import requests
    except Exception as e:
        _jlog("WARN","color_requests_import_error", error=str(e))
        return []

    img = _pil_crop_xyxy(image_pil, veh_bbox_orig) if veh_bbox_orig else image_pil
    payload = {
        "image_b64": base64.b64encode(_to_jpeg_bytes(img, quality=92)).decode("ascii"),
        "veh_box": [float(v) for v in veh_bbox_orig] if veh_bbox_orig else None,
        "max_colors": 3,
    }
    headers = {"X-API-Key": api_key}
    for attempt in range(2):
        try:
            r = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_s)
            if r.status_code == 200:
                js = r.json()
                out: List[Dict[str, Any]] = []
                for i in (js.get("colors") or [])[:3]:
                    # accept either your gateway format or generic
                    lbl = i.get("label") or i.get("base") or ""
                    finish, base, light = _parse_color_label(lbl)
                    out.append({
                        "label": lbl,
                        "base": i.get("base") or base,
                        "finish": i.get("finish") or finish,
                        "lightness": i.get("lightness") or light,
                        "fraction": float(i.get("fraction", 0.0)),
                        "conf": float(i.get("conf", 0.0)),
                    })
                return out
            time.sleep(0.25)
        except Exception:
            time.sleep(0.2)
    _jlog("WARN","color_gateway_error", endpoint=endpoint)
    return []

def get_colors(image_pil: Image.Image,
               veh_bbox_orig: Optional[Tuple[float,float,float,float]] = None,
               allow_multitone: bool = True,
               timeout_s: int = 20) -> List[Dict[str, Any]]:
    """
    Provider-aware color detection with normalized output.
    Returns: list[ {label, base, finish, lightness, fraction, conf} ]
    """
    prov = _color_provider()
    if prov == "chatgpt":
        return _color_via_openai(image_pil, veh_bbox_orig, allow_multitone, timeout_s)
    if prov == "gateway":
        return _color_via_gateway(image_pil, veh_bbox_orig, timeout_s)
    _jlog("WARN","color_provider_unset_or_missing_env")
    return []

# ============================== PLATE: provider selection ==============================

def _plate_provider() -> str:
    p = (os.getenv("PLATE_OCR_PROVIDER") or "").strip().lower()
    if p in {"platerecognizer","plate_recognizer"}: return "platerecognizer"
    if p in {"gateway","internal"}:                 return "gateway"
    # auto-detect
    if os.getenv("PLATERECOGNIZER_TOKEN") or os.getenv("PLATE_RECOGNITION_API_KEY") \
       or os.getenv("PLATE_API_KEY") or os.getenv("PLATE_RECOGNIZER_TOKEN"):
        return "platerecognizer"
    if os.getenv("PLATE_UTILS_ENDPOINT"):           return "gateway"
    return "unset"

def _plate_via_platerecognizer(image_pil: Image.Image,
                               plate_boxes_orig: Optional[List[Tuple[float,float,float,float]]],
                               regions: Optional[List[str]],
                               topk: int,
                               timeout_s: int) -> Dict[str, Any]:
    # token aliases
    token = (
        os.getenv("PLATERECOGNIZER_TOKEN") or
        os.getenv("PLATE_RECOGNITION_API_KEY") or
        os.getenv("PLATE_API_KEY") or
        os.getenv("PLATE_RECOGNIZER_TOKEN")
    )
    if not token:
        _jlog("WARN","plate_recognizer_token_missing")
        return {"text": None, "confidence": None, "bbox": None, "candidates": []}

    api_base = os.getenv("PLATE_API_BASE", "https://api.platerecognizer.com/v1")
    try:
        import requests
    except Exception as e:
        _jlog("WARN","plate_requests_import_error", error=str(e))
        return {"text": None, "confidence": None, "bbox": None, "candidates": []}

    session = requests.Session()
    headers = {"Authorization": f"Token {token}"}

    def _post(blob: bytes) -> dict:
        files = {"upload": ("img.jpg", blob)}
        data  = {}
        if regions:
            data["regions"] = regions
        r = session.post(f"{api_base}/plate-reader/", headers=headers, data=data, files=files, timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    def _best(js: dict) -> Dict[str, Any]:
        best_plate, best_score, best_box = None, None, None
        candidates: List[Tuple[str,float]] = []
        for r in (js.get("results") or []):
            plate = (r.get("plate") or "").upper()
            score = float(r.get("score", 0) or 0.0)
            box   = r.get("box") or {}
            bx = [float(box.get("xmin", 0.0)), float(box.get("ymin", 0.0)),
                  float(box.get("xmax", 0.0)), float(box.get("ymax", 0.0))]
            if plate:
                candidates.append((plate, score))
                if best_score is None or score > best_score:
                    best_plate, best_score, best_box = plate, score, bx
            for c in (r.get("candidates") or []):
                cp = (c.get("plate") or "").upper()
                if cp:
                    candidates.append((cp, float(c.get("score", 0) or 0.0)))
        # dedup & sort
        ded = {}
        for p, s in candidates:
            if p not in ded or s > ded[p]:
                ded[p] = s
        cand_list = [{"plate": p, "score": s} for p, s in sorted(ded.items(), key=lambda kv: -kv[1])]
        return {"text": best_plate, "confidence": best_score, "bbox": best_box, "candidates": cand_list}

    try:
        best = {"text": None, "confidence": None, "bbox": None, "candidates": []}
        if plate_boxes_orig:
            tried = 0
            for box in plate_boxes_orig:
                if tried >= max(1, int(topk)): break
                tried += 1
                crop = _pil_crop_xyxy(image_pil, box)
                js = _post(_to_jpeg_bytes(crop, 95))
                got = _best(js)
                # map crop-local box back to original if present
                if got.get("bbox"):
                    cx1, cy1, cx2, cy2 = got["bbox"]
                    ox1, oy1 = float(box[0]), float(box[1])
                    got["bbox"] = [ox1 + cx1, oy1 + cy1, ox1 + cx2, oy1 + cy2]
                else:
                    got["bbox"] = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                # merge candidates & keep best by confidence
                best["candidates"].extend(got.get("candidates", []))
                sc = got.get("confidence")
                if got.get("text") and (best["confidence"] is None or (sc is not None and sc > best["confidence"])):
                    best.update({"text": got["text"], "confidence": sc, "bbox": got["bbox"]})
        else:
            js = _post(_to_jpeg_bytes(image_pil, 95))
            got = _best(js)
            best.update(got)

        # final candidate dedup
        ded = {}
        for c in best["candidates"]:
            p = (c.get("plate") or "").upper()
            s = float(c.get("score", 0) or 0.0)
            if p not in ded or s > ded[p]:
                ded[p] = s
        best["candidates"] = [{"plate": p, "score": s} for p, s in sorted(ded.items(), key=lambda kv: -kv[1])]

        return best
    except Exception as e:
        msg = str(e)
        _jlog("WARN","plate_recognizer_error", error=msg[:300])
        return {"text": None, "confidence": None, "bbox": None, "candidates": [], "error": msg[:300]}

def _plate_via_gateway(image_pil: Image.Image,
                       plate_boxes_orig: Optional[List[Tuple[float,float,float,float]]],
                       timeout_s: int) -> Dict[str, Any]:
    endpoint = os.getenv("PLATE_UTILS_ENDPOINT")
    api_key  = os.getenv("PLATE_UTILS_API_KEY")
    if not endpoint or not api_key:
        _jlog("WARN","plate_gateway_env_missing", endpoint=bool(endpoint), api_key=bool(api_key))
        return {"text": None, "confidence": None, "bbox": None, "candidates": []}
    try:
        import requests
    except Exception as e:
        _jlog("WARN","plate_requests_import_error", error=str(e))
        return {"text": None, "confidence": None, "bbox": None, "candidates": []}

    def _call(img: Image.Image) -> Dict[str, Any]:
        payload = {"image_b64": base64.b64encode(_to_jpeg_bytes(img, 95)).decode("ascii")}
        headers = {"X-API-Key": api_key}
        for attempt in range(2):
            try:
                r = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_s)
                if r.status_code == 200:
                    return r.json()
                time.sleep(0.25)
            except Exception:
                time.sleep(0.2)
        return {"error": f"gateway error @ {endpoint}"}

    if plate_boxes_orig:
        best = {"text": None, "confidence": None, "bbox": None, "candidates": []}
        for box in plate_boxes_orig[:3]:
            crop = _pil_crop_xyxy(image_pil, box)
            js   = _call(crop)
            text = (js.get("text") or "").strip()
            conf = js.get("conf") or js.get("confidence")
            bbox = js.get("bbox")
            if bbox is None:
                bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            # merge
            if text and (best["confidence"] is None or (conf and conf > best["confidence"])):
                best.update({"text": text, "confidence": conf, "bbox": bbox})
            # candidates (optional)
            for c in (js.get("candidates") or []):
                if "plate" in c and "score" in c:
                    best["candidates"].append({"plate": str(c["plate"]).upper(), "score": float(c["score"])})
        return best
    else:
        js = _call(image_pil)
        return {
            "text": (js.get("text") or "").strip() or None,
            "confidence": js.get("conf") or js.get("confidence"),
            "bbox": js.get("bbox"),
            "candidates": js.get("candidates") or [],
            **({"error": js.get("error")} if js.get("error") else {})
        }

def get_plate(image_pil: Image.Image,
              plate_boxes_orig: Optional[List[Tuple[float,float,float,float]]] = None,
              regions: Optional[List[str]] = None,
              topk: int = 3,
              timeout_s: int = 20) -> Dict[str, Any]:
    """
    Provider-aware OCR for license plates.
    Returns: {"text","confidence","bbox","candidates","error?"}
    """
    prov = _plate_provider()
    if prov == "platerecognizer":
        # default regions from env if not provided
        if regions is None:
            env_r = [r.strip() for r in (os.getenv("PLATE_REGIONS") or "").split(",") if r.strip()]
            regions = env_r if env_r else None
        return _plate_via_platerecognizer(image_pil, plate_boxes_orig, regions, topk, timeout_s)
    if prov == "gateway":
        return _plate_via_gateway(image_pil, plate_boxes_orig, timeout_s)
    _jlog("WARN","plate_provider_unset_or_missing_env")
    return {"text": None, "confidence": None, "bbox": None, "candidates": []}

# ============================== public API ==============================

__all__ = ["get_colors", "get_plate"]


