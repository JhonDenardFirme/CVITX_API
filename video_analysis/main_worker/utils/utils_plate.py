# api/analysis/utils_plate.py
"""
CVITX · Plate Utils (EC2 ↔ Colab parity)

Public API:
  - recognize_plate(image_pil, plate_boxes_orig=None, regions=None, topk=3, timeout_s=20) -> Dict
      Returns:
        {
          "text": str|None,
          "confidence": float|None,
          "bbox": [x1,y1,x2,y2]|None,   # ORIGINAL image coords (clamped to bounds)
          "candidates": [{"plate":str,"score":float}, ...],
          "error": str?                  # present only when an error happens
        }
      • Handles its own CROPPING (per provided plate_boxes_orig), mapping crop-local bbox back to ORIGINAL coords.

  - read_plate_text(image_pil, plate_box=None, timeout_s=20) -> Dict
      Back-compat wrapper → {"text": str, "conf": float} or {}.

Environment:
  - PLATE_RECOGNITION_API_KEY (preferred) | PLATE_API_KEY | PLATE_RECOGNIZER_TOKEN
  - PLATE_API_BASE      (default: https://api.platerecognizer.com/v1)
  - PLATE_REGIONS       (default: "ph"; comma-separated accepted)
  - COLOR_TIMEOUT       (seconds; default: 30)  # reused for HTTP timeout

Optional microservice (if you deploy your own OCR service):
  - PLATE_UTILS_ENDPOINT
  - PLATE_UTILS_API_KEY  (# preferred for custom service authentication)
"""

from __future__ import annotations
import os, io, base64
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


# ------------------------- env & tiny helpers -------------------------
def _timeout_s() -> float:
    try:
        return float(os.getenv("COLOR_TIMEOUT", "30"))
    except Exception:
        return 30.0


def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _pil_crop_xyxy(img: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, float(x1))); y1 = max(0, min(h - 1, float(y1)))
    x2 = max(x1 + 1, min(w, float(x2))); y2 = max(y1 + 1, min(h, float(y2)))
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def _clamp_xyxy(xyxy: List[float], W: int, H: int) -> List[float]:
    """
    Clamp [x1,y1,x2,y2] to image bounds (float), ensuring the box lies inside [0..W-1]/[0..H-1].
    """
    if not xyxy or len(xyxy) != 4:
        return [0.0, 0.0, float(max(0, W - 1)), float(max(0, H - 1))]
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = max(0.0, min(x1, max(0, W - 1)))
    y1 = max(0.0, min(y1, max(0, H - 1)))
    x2 = max(0.0, min(x2, max(0, W - 1)))
    y2 = max(0.0, min(y2, max(0, H - 1)))
    return [float(x1), float(y1), float(x2), float(y2)]


# ------------------------- optional microservice path -------------------------
def _call_plate_microservice(image_pil: Image.Image) -> Optional[Dict[str, Any]]:
    ep  = os.getenv("PLATE_UTILS_ENDPOINT")
    key = os.getenv("PLATE_UTILS_API_KEY")  # do NOT fall back to OPENAI keys
    if not ep or not key:
        return None
    try:
        import requests
        payload = {
            "image_b64": base64.b64encode(_pil_to_jpeg_bytes(image_pil, 95)).decode("ascii")
        }
        # Hardened header policy: use only the intended microservice key.
        headers = {"X-API-Key": key, "Content-Type": "application/json"}
        r = requests.post(ep, json=payload, headers=headers, timeout=_timeout_s())
        if not r.ok:
            return None
        js = r.json() or {}
        text = (js.get("text") or js.get("plate") or "").strip().upper()
        conf = js.get("conf", js.get("confidence"))
        return {
            "text": (text or None),
            "confidence": (float(conf) if conf is not None else None),
            "bbox": None,
            "candidates": js.get("candidates", []),
        }
    except Exception:
        return None


# ------------------------- Plate Recognizer path (Colab-parity) -------------------------
def _plate_token() -> Optional[str]:
    return (
        os.getenv("PLATE_RECOGNITION_API_KEY")
        or os.getenv("PLATE_API_KEY")
        or os.getenv("PLATE_RECOGNIZER_TOKEN")
    )


def _plate_base() -> str:
    return os.getenv("PLATE_API_BASE", "https://api.platerecognizer.com/v1")


def _post_snapshot(blob: bytes, regions: Optional[List[str]]) -> Dict[str, Any]:
    import requests
    tok = _plate_token()
    if not tok:
        raise RuntimeError("No plate token configured")
    headers = {"Authorization": f"Token {tok}"}
    data: Dict[str, Any] = {}
    if regions:
        data["regions"] = regions  # list is accepted (Colab parity)
    r = requests.post(
        f"{_plate_base()}/plate-reader/",
        headers=headers,
        data=data,
        files={"upload": ("frame.jpg", blob)},
        timeout=_timeout_s(),
    )
    r.raise_for_status()
    return r.json()


def _pick_best(api_json: dict) -> Dict[str, Any]:
    best_plate, best_score, best_box = None, None, None
    cands: Dict[str, float] = {}
    for r in api_json.get("results", []) or []:
        plate = (r.get("plate") or "").upper()
        score = float(r.get("score", 0) or 0.0)
        box   = r.get("box") or {}
        xyxy  = [
            float(box.get("xmin", 0.0)),
            float(box.get("ymin", 0.0)),
            float(box.get("xmax", 0.0)),
            float(box.get("ymax", 0.0)),
        ]
        if plate:
            cands[plate] = max(cands.get(plate, 0.0), score)
            if best_score is None or score > best_score:
                best_plate, best_score, best_box = plate, score, xyxy
        for c in r.get("candidates") or []:
            cp = (c.get("plate") or "").upper()
            cs = float(c.get("score", 0) or 0.0)
            if cp:
                cands[cp] = max(cands.get(cp, 0.0), cs)
    cand_list = [{"plate": p, "score": s} for p, s in sorted(cands.items(), key=lambda kv: -kv[1])]
    return {"text": best_plate, "confidence": best_score, "bbox": best_box, "candidates": cand_list}


# ------------------------- Public API -------------------------
def recognize_plate(
    image_pil: Image.Image,
    plate_boxes_orig: Optional[List[Tuple[float, float, float, float]]] = None,
    regions: Optional[List[str]] = None,
    topk: int = 3,
    timeout_s: int = 20,  # signature parity; actual timeout uses COLOR_TIMEOUT
) -> Dict[str, Any]:
    """
    Strategy:
      • If PLATE_UTILS_ENDPOINT is configured → single full-frame call there.
      • Else → Plate Recognizer; try each provided crop (up to topk), then fall back to full frame.
    Returns normalized dict with text/conf/bbox/candidates (crop-local bbox is mapped back to ORIGINAL coords and clamped).
    """
    # (A) microservice
    ms = _call_plate_microservice(image_pil)
    if ms is not None:
        return ms

    # (B) Plate Recognizer direct REST (Colab parity)
    if not _plate_token():
        return {
            "text": None, "confidence": None, "bbox": None, "candidates": [],
            "error": "No plate token configured"
        }

    region_list = (
        regions
        if regions is not None
        else [r.strip() for r in os.getenv("PLATE_REGIONS", "ph").split(",") if r.strip()]
    )

    best: Dict[str, Any] = {"text": None, "confidence": None, "bbox": None, "candidates": []}
    W, H = image_pil.size

    try:
        import requests  # noqa: F401

        if plate_boxes_orig and len(plate_boxes_orig) > 0:
            tried = 0
            for box in plate_boxes_orig:
                if tried >= max(1, int(topk)):
                    break
                tried += 1

                crop = _pil_crop_xyxy(image_pil, box)
                js   = _post_snapshot(_pil_to_jpeg_bytes(crop, 95), region_list)
                got  = _pick_best(js)

                # map crop-local bbox back to original coords and clamp to bounds
                if got.get("bbox"):
                    cx1, cy1, cx2, cy2 = got["bbox"]
                    ox1, oy1 = float(box[0]), float(box[1])
                    remapped = [ox1 + float(cx1), oy1 + float(cy1), ox1 + float(cx2), oy1 + float(cy2)]
                    got["bbox"] = _clamp_xyxy(remapped, W, H)
                else:
                    got["bbox"] = _clamp_xyxy([float(box[0]), float(box[1]), float(box[2]), float(box[3])], W, H)

                # merge candidates & keep best
                best["candidates"].extend(got.get("candidates", []))
                sc = got.get("confidence")
                if got.get("text") and (best["confidence"] is None or (sc is not None and sc > best["confidence"])):
                    best.update({"text": got["text"], "confidence": sc, "bbox": got["bbox"]})
        else:
            js = _post_snapshot(_pil_to_jpeg_bytes(image_pil, 95), region_list)
            got = _pick_best(js)
            # full-frame path: if bbox exists, clamp to bounds
            if got.get("bbox"):
                got["bbox"] = _clamp_xyxy(got["bbox"], W, H)
            best.update(got)

        # dedupe candidates by max score
        dedup: Dict[str, float] = {}
        for c in best["candidates"]:
            p = (c.get("plate") or "").upper()
            s = float(c.get("score", 0) or 0.0)
            if p and (p not in dedup or s > dedup[p]):
                dedup[p] = s
        best["candidates"] = [{"plate": p, "score": s} for p, s in sorted(dedup.items(), key=lambda kv: -kv[1])]
        return best

    except requests.HTTPError as e:  # type: ignore
        err  = e.response.text if getattr(e, "response", None) is not None else str(e)
        code = getattr(getattr(e, "response", None), "status_code", "???")
        return {
            "text": None, "confidence": None, "bbox": None, "candidates": [],
            "error": f"HTTP {code}: {err[:300]}"
        }
    except Exception as ex:
        return {
            "text": None, "confidence": None, "bbox": None, "candidates": [],
            "error": f"{type(ex).__name__}: {ex}"
        }


def read_plate_text(
    image_pil: Image.Image,
    plate_box: Optional[Tuple[float, float, float, float]] = None,
    timeout_s: int = 20,
) -> Dict[str, Any]:
    """
    Back-compat wrapper.
    Returns {"text": str, "conf": float} or {}.
    """
    if plate_box is not None:
        got = recognize_plate(image_pil, plate_boxes_orig=[plate_box], topk=1, timeout_s=timeout_s)
    else:
        got = recognize_plate(image_pil, plate_boxes_orig=None, topk=1, timeout_s=timeout_s)
    if got.get("text"):
        return {"text": got["text"], "conf": float(got.get("confidence") or 0.0)}
    return {}


__all__ = ["recognize_plate", "read_plate_text"]
