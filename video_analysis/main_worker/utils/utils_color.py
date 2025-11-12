# api/analysis/utils_color.py
"""
CVITX · Color Utils (EC2 ↔ Colab parity, Phase-9 hardened)

Public API:
  - detect_vehicle_color(image_pil, veh_box=None, allow_multitone=True, timeout_s=20) -> List[FBL]
      Returns strict FBL only (≤3): [{"finish","base","lightness","conf"}]
      ALL returned colors have conf ≥ STRICT_MIN_CONF (default: 0.70).
      (If nothing passes, returns [] and the Engine will supply a null fallback.)
  - detect_vehicle_color_payload(image_pil, veh_box=None, allow_multitone=True, timeout_s=20) -> Dict
      Returns the Colab-shaped JSON (labels/fractions/..., + confidence_raw for transparency).

Environment (Colab-compatible):
  - COLOR_RECOGNITION_API_KEY  (preferred)  → sets OPENAI_API_KEY if empty
  - OPENAI_API_KEY             (fallback)   → used by OpenAI SDK
  - COLOR_MODEL_ID             (default: gpt-5-mini)
  - COLOR_TIMEOUT              (seconds; default: 30)

Optional microservice (kept; used first if present):
  - COLOR_UTILS_ENDPOINT
  - COLOR_UTILS_API_KEY
  - COLOR_PROVIDER             (e.g., "chatgpt")  # hint only

Phase-9 hardening (this module):
  • Never forward OPENAI secrets to non-OpenAI endpoints (microservice).
  • Strict FBL return: ≤3 entries, each with conf ≥ 0.70 (configurable).
  • Colab parity retained for payload shape and confidence accounting.
"""

from __future__ import annotations
import os, io, json, base64
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

# ------------------------- knobs & helpers -------------------------

STRICT_MIN_CONF = float(os.getenv("COLOR_MIN_CONF", "0.70"))  # for FBL gating

_BASE_COLORS = [
    "Red","Orange","Yellow","Green","Blue","Purple","Pink",
    "White","Gray","Black","Silver","Gold","Brown","Beige","Maroon","Cyan"
]
_FINISH = {"Metallic","Matte","Glossy"}
_LIGHT  = {"Light","Dark"}
_BASE   = set(_BASE_COLORS)

def _timeout_s() -> float:
    try:
        return float(os.getenv("COLOR_TIMEOUT", "30"))
    except Exception:
        return 30.0

def _pil_crop_xyxy(img: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, float(x1))); y1 = max(0, min(h - 1, float(y1)))
    x2 = max(x1 + 1, min(w, float(x2)));  y2 = max(y1 + 1, min(h, float(y2)))
    return img.crop((int(x1), int(y1), int(x2), int(y2)))

def _data_url_jpeg(img: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def _ensure_openai_key():
    # Prefer COLOR_RECOGNITION_API_KEY (Colab parity); allow OPENAI_API_KEY.
    key = os.getenv("COLOR_RECOGNITION_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key

# ------------------------- prompt & schema (Colab parity) -------------------------

_COLOR_INSTRUCTIONS = f"""
You are 'Color Detector AI' for VEHICLE BODY PAINT.

SCOPE (IMPORTANT):
- Analyze the BODY PAINT ONLY: painted panels, roof, hood, trunk, doors, pillars, factory accents if painted.
- EXCLUDE: glass/windows, lights, license plates, chrome/metal grille, wheels/tires, mirrors' reflective glass, sky/road/reflections/shadows/stickers/dirt.

TASK:
1) Decide the number of PROMINENT body-paint colors N ∈ {{1,2,3}}.
   - Start with 1 (most common case).
   - Include a 2nd color ONLY if you are ≥{STRICT_MIN_CONF:.2f} certain it’s part of the body paint (not reflections or non-paint parts).
   - Include a 3rd color ONLY if you are ≥{STRICT_MIN_CONF:.2f} certain.
   - Do NOT exceed 3.

2) For each chosen color, produce a descriptor with up to 3 parts: [Finish] [Base] [Lightness]
   - Finish ∈ {{Metallic, Matte, Glossy}} (omit if unclear)
   - Base   ∈ {{{", ".join(_BASE_COLORS)}}} (REQUIRED)
   - Lightness ∈ {{Light, Dark}} (omit if unclear)
   - Examples: "Metallic Gray Dark", "White", "Glossy Red", "Beige Light", "Gray Dark"

3) Provide fractional coverage per color (fractions sum to 1.0). Sort colors by coverage (largest first).

CONFIDENCE:
- Provide an overall confidence 0..1 for your BODY-PAINT color decision (not segmentation).
- Also provide per-color confidences (0..1) indicating your certainty that each color is truly in the body paint.
- Be conservative with strong glare/reflections/shadows/compression.

OUTPUT RULES:
- Return JSON ONLY using the provided schema.
- If uncertain about additional colors (<{STRICT_MIN_CONF:.2f}), collapse to a single dominant color.
"""

_COLOR_SCHEMA = {
    "name": "VehicleBodyColorStructuredV2",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "labels":             {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}},
            "fractions":          {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
            "color_confidences":  {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
            "pixels":             {"type": "integer", "minimum": 1},
            "confidence":         {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "stainless":          {"type": "boolean"},
            "body_paint_only":    {"type": "boolean"}
        },
        "required": ["labels","fractions","color_confidences","pixels","confidence","stainless","body_paint_only"]
    }
}

# ------------------------- payload → FBL mapping (strict) -------------------------

def _parse_label_to_fbl(label: str) -> Dict[str, Optional[str]]:
    """
    Parse a provider label like "Metallic Gray Dark" into FBL fields.
    Conservative: only recognize known finish/base/lightness tokens.
    """
    label = (label or "").strip()
    if not label:
        return {"finish": None, "base": None, "lightness": None}

    toks = [t.strip() for t in label.replace(",", " ").split() if t.strip()]
    finish = base = light = None

    # Finish
    for t in toks:
        if t in _FINISH and finish is None:
            finish = t

    # Lightness
    for t in toks:
        if t in _LIGHT and light is None:
            light = t

    # Base (must be one of canonical base colors)
    remaining = [t for t in toks if t not in _FINISH and t not in _LIGHT]
    for t in remaining:
        if t in _BASE:
            base = t
            break

    # Final conservative fallback: do NOT invent non-canonical bases
    return {"finish": finish, "base": base, "lightness": light}

def _payload_to_fbl(payload: Dict[str, Any], min_conf_for_keep: float = STRICT_MIN_CONF) -> List[Dict[str, Any]]:
    """
    Normalize provider payload to strict FBL list.
    Drop any entries with conf < min_conf_for_keep and cap to 3 items.
    Return [] if nothing qualifies (Engine provides null fallback).
    """
    labels   = list(payload.get("labels") or [])
    per_conf = list(payload.get("color_confidences") or [])
    out: List[Dict[str, Any]] = []

    k = min(len(labels), len(per_conf))
    for i in range(k):
        try:
            c = float(per_conf[i])
        except Exception:
            c = 0.0
        if c < float(min_conf_for_keep):
            continue
        fbl = _parse_label_to_fbl(labels[i])
        # keep only if we parsed a canonical base (enforces schema hygiene)
        if fbl.get("base") is None:
            continue
        fbl["conf"] = float(c)
        out.append(fbl)
        if len(out) == 3:
            break

    return out  # [] allowed (Engine supplies default)

# ------------------------- color via microservice (optional; hardened) -------------------------

def _call_color_microservice(
    image_pil: Image.Image,
    veh_box: Optional[Tuple[float,float,float,float]],
    allow_multitone: bool
) -> Optional[Dict[str, Any]]:
    """
    If COLOR_UTILS_ENDPOINT is set, call it first. DO NOT forward any OpenAI secrets.
    """
    ep = os.getenv("COLOR_UTILS_ENDPOINT")
    if not ep:
        return None

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("COLOR_UTILS_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    # IMPORTANT: never forward OPENAI_API_KEY/Authorization to non-OpenAI endpoints.

    img = image_pil if veh_box is None else _pil_crop_xyxy(image_pil, veh_box)
    payload: Dict[str, Any] = {"image_b64": _data_url_jpeg(img).split(",", 1)[1], "max_colors": 3}

    # Optional hints only (no secrets):
    provider = os.getenv("COLOR_PROVIDER") or "chatgpt"
    model_id = os.getenv("COLOR_MODEL_ID", "gpt-5-mini")
    payload.update({
        "mode": "dominant_colors",
        "provider": provider,
        "model": model_id,
        "prompt": _COLOR_INSTRUCTIONS,
        "response_shape": "colors_v1",
        "allow_multitone": bool(allow_multitone),
    })

    try:
        import requests
        r = requests.post(ep, json=payload, headers=headers, timeout=_timeout_s())
        if not r.ok:
            return None
        js = r.json() or {}
        # Normalize both possible service styles to Colab-like shape
        if "labels" in js and "color_confidences" in js:
            return js
        if "colors" in js and isinstance(js["colors"], list):
            labels = []
            percs  = []
            for c in js["colors"]:
                labels.append((c.get("label") or c.get("name") or "").strip())
                percs.append(float(c.get("confidence", c.get("conf", 0.0)) or 0.0))
            return {
                "labels": labels,
                "fractions": js.get("fractions", []),
                "color_confidences": percs,
                "pixels": js.get("pixels", 1),
                "confidence": js.get("confidence", max(percs) if percs else 0.0),
                "stainless": bool(js.get("stainless", False)),
                "body_paint_only": True,
            }
        return None
    except Exception:
        return None

# ------------------------- color via OpenAI (Colab-identical) -------------------------

def _call_color_openai(
    image_pil: Image.Image,
    veh_box: Optional[Tuple[float,float,float,float]],
    allow_multitone: bool
) -> Optional[Dict[str, Any]]:
    _ensure_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        return None

    from openai import OpenAI
    client = OpenAI()
    model = os.getenv("COLOR_MODEL_ID", "gpt-5-mini")

    img = image_pil if veh_box is None else _pil_crop_xyxy(image_pil, veh_box)
    data_url = _data_url_jpeg(img)
    fmt_payload = {
        "format": {
            "type": "json_schema",
            "name": _COLOR_SCHEMA["name"],
            "strict": _COLOR_SCHEMA["strict"],
            "schema": _COLOR_SCHEMA["schema"],
        }
    }
    instructions = _COLOR_INSTRUCTIONS
    if not allow_multitone:
        instructions += f"\nReturn exactly ONE dominant body-paint color (labels length = 1; fractions = [1.0])."

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
        text=fmt_payload,
        timeout=_timeout_s(),
    )

    # Parse like Colab (structured → output_text → raw text)
    try:
        parsed = resp.output[0].content[0].parsed  # type: ignore[attr-defined]
        result = parsed if isinstance(parsed, dict) else {}
    except Exception:
        try:
            result = json.loads(resp.output_text)  # type: ignore[attr-defined]
        except Exception:
            result = json.loads(resp.output[0].content[0].text)  # type: ignore[attr-defined]

    # --- sanitize / normalize (Colab parity + confidence_raw exposed) ---
    labels      = list(result.get("labels", [])) if isinstance(result, dict) else []
    fractions   = list(result.get("fractions", [])) if isinstance(result, dict) else []
    per_color   = list(result.get("color_confidences", [])) if isinstance(result, dict) else []
    conf_raw    = float(result.get("confidence", 0.0)) if isinstance(result, dict) else 0.0
    stainless   = bool(result.get("stainless", False)) if isinstance(result, dict) else False

    def _renorm(xs: List[float]) -> List[float]:
        s = float(sum(max(0.0, x) for x in xs))
        if s <= 1e-6:
            return [1.0] + [0.0] * (len(xs) - 1)
        return [float(max(0.0, x)) / s for x in xs]

    if not labels:
        labels = ["Unknown"]
    if not fractions or len(fractions) != len(labels):
        fractions = [1.0] + [0.0] * (len(labels) - 1)
    fractions = _renorm(fractions)

    if not per_color or len(per_color) != len(labels):
        per_color = [min(0.98, 0.6 + 0.4 * f) for f in fractions]  # proxy if missing

    # sort by coverage desc, align arrays
    order = list(range(len(labels)))
    order.sort(key=lambda i: fractions[i], reverse=True)
    labels    = [labels[i] for i in order]
    fractions = [fractions[i] for i in order]
    per_color = [per_color[i] for i in order]

    # Certainty filter for extras will be enforced later by _payload_to_fbl.
    # We still keep Colab-style shape here.
    k = len(labels)
    dom = float(fractions[0]) if fractions else 1.0
    complexity_penalty = [1.00, 0.92, 0.85][k - 1] if 1 <= k <= 3 else 0.80
    dominance_penalty  = 0.75 + 0.25 * dom
    conf_blended       = 0.55 + 0.45 * max(0.0, min(1.0, conf_raw))
    conf_honest        = max(0.0, min(1.0, conf_blended * complexity_penalty * dominance_penalty))

    return {
        "labels": labels,
        "fractions": fractions,
        "pixels": 1,  # placeholder for parity
        "confidence": float(conf_honest),
        "confidence_raw": float(max(0.0, min(1.0, conf_raw))),
        "color_confidences": [float(max(0.0, min(1.0, c))) for c in per_color],
        "stainless": bool(stainless),
        "body_paint_only": True,
    }

# ------------------------- Public API -------------------------

def detect_vehicle_color_payload(
    image_pil: Image.Image,
    veh_box: Optional[Tuple[float, float, float, float]] = None,
    allow_multitone: bool = True,
    timeout_s: int = 20,  # signature parity; actual timeout uses COLOR_TIMEOUT
) -> Dict[str, Any]:
    """
    Returns Colab-shaped payload dict:
      {"labels","fractions","color_confidences","pixels","confidence","stainless","body_paint_only", "confidence_raw?"}
      {} on failure.
    """
    js = _call_color_microservice(image_pil, veh_box, allow_multitone)
    if js is None:
        js = _call_color_openai(image_pil, veh_box, allow_multitone)
    return js or {}

def detect_vehicle_color(
    image_pil: Image.Image,
    veh_box: Optional[Tuple[float, float, float, float]] = None,
    allow_multitone: bool = True,
    timeout_s: int = 20,  # signature parity
) -> List[Dict[str, Any]]:
    """
    Returns strict FBL list (≤3), each with conf ≥ STRICT_MIN_CONF.
    Returns [] if nothing qualifies (Engine will add a null placeholder).
    """
    payload = detect_vehicle_color_payload(image_pil, veh_box, allow_multitone, timeout_s)
    return _payload_to_fbl(payload, min_conf_for_keep=STRICT_MIN_CONF)

__all__ = ["detect_vehicle_color", "detect_vehicle_color_payload"]
