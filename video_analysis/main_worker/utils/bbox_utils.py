# ── bbox_utils.py # api/analysis/bbox_utils.py
# Final unified version (V3) — Colab-aligned, engine-compatible
from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Iterable, Any

import torch
import torch.nn.functional as F

# ============================================================
# Small helpers
# ============================================================

def _cfg(CFG, name: str, default: Any):
    """Safe CFG fetch with default (keeps engine signature stable)."""
    return getattr(CFG, name, default) if hasattr(CFG, name) else default

def _box_area(b: Optional[Iterable[float]]) -> float:
    if not b: return 0.0
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _iou(a: Optional[Iterable[float]], b: Optional[Iterable[float]]) -> float:
    if not a or not b: return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    u = _box_area(a) + _box_area(b) - inter
    return inter / u if u > 0 else 0.0

# ============================================================
# Letterbox (Tensor-first; adapter wrapper stays in engine/Colab)
# ============================================================

def letterbox_fit(image: torch.Tensor, size: int):
    """
    Resize with aspect ratio into square 'size' canvas with equal padding.
    Accepts CHW, BCHW, or HWC torch Tensors. Returns:
      sq_image (same rank/layout), pad=(pad_w_left, pad_h_top), scale (float).
    """
    if not torch.is_tensor(image):
        raise TypeError("letterbox_fit expects a torch.Tensor")

    orig_dtype = image.dtype
    is_batched = image.dim() == 4
    is_chw     = (image.dim() == 3 and image.shape[0] in (1, 3))
    is_hwc     = (image.dim() == 3 and image.shape[-1] in (1, 3))

    if is_batched:
        x = image
    elif is_chw:
        x = image.unsqueeze(0)
    elif is_hwc:
        x = image.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError("Unsupported tensor rank")

    B, C, H, W = x.shape
    s = float(size)
    scale = min(s / max(1.0, W), s / max(1.0, H))
    new_w, new_h = int(round(W * scale)), int(round(H * scale))

    # Interpolate in float; cast back to original dtype at the end
    x_float = x.float()
    mode = "bilinear" if C > 1 else "nearest"
    x_resized = F.interpolate(
        x_float, size=(new_h, new_w), mode=mode,
        align_corners=False if mode == "bilinear" else None
    )

    pw_total, ph_total = max(0, size - new_w), max(0, size - new_h)
    left, right = pw_total // 2, pw_total - (pw_total // 2)
    top, bottom = ph_total // 2, ph_total - (ph_total // 2)

    x_padded = F.pad(x_resized, (left, right, top, bottom), value=0.0)

    # Cast back conservatively (do not clamp negatives for float pipelines)
    if orig_dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        x_padded = x_padded.clamp(0, 255).round().to(orig_dtype)
    else:
        x_padded = x_padded.to(orig_dtype)

    if is_batched:
        sq = x_padded
    elif is_chw:
        sq = x_padded.squeeze(0)
    else:
        sq = x_padded.squeeze(0).permute(1, 2, 0)

    return sq, (left, top), float(scale)

def invert_letterbox_box(box_on_sq, orig_w: int, orig_h: int, pad: tuple, scale: float):
    """
    Map a box from the letterboxed square back to original image coords.
    """
    if box_on_sq is None: return None
    x1, y1, x2, y2 = [float(v) for v in (box_on_sq.tolist() if isinstance(box_on_sq, torch.Tensor) else box_on_sq)]
    pw, ph = float(pad[0]), float(pad[1])
    s = max(1e-12, float(scale))
    xo1, yo1 = (x1 - pw) / s, (y1 - ph) / s
    xo2, yo2 = (x2 - pw) / s, (y2 - ph) / s
    xo1 = max(0.0, min(xo1, orig_w - 1.0)); yo1 = max(0.0, min(yo1, orig_h - 1.0))
    xo2 = max(0.0, min(xo2, orig_w - 1.0)); yo2 = max(0.0, min(yo2, orig_h - 1.0))
    return (xo1, yo1, xo2, yo2)

# ============================================================
# Decode helpers (Colab-aligned)
# ============================================================

def _unpack_bbox_to_BP4HW(raw: torch.Tensor, P_expected: int, H: int, W: int) -> torch.Tensor | None:
    """
    Accepts: [B,4P,H,W] or [B,P,4,H,W] or [B,4,P,H,W]
    Returns: [B,P_use,4,H,W] (P_use = min(P_expected, inferred_P))
    """
    if raw is None or not torch.is_tensor(raw) or raw.dim() not in (4, 5):
        return None

    if raw.dim() == 5 and raw.shape[2] == 4:
        B, P_in, _, Hh, Ww = raw.shape
        if (Hh, Ww) != (H, W): return None
        return raw[:, :min(P_expected, P_in)].contiguous()

    if raw.dim() == 5 and raw.shape[1] == 4:
        B, _, P_in, Hh, Ww = raw.shape
        if (Hh, Ww) != (H, W): return None
        P_use = min(P_expected, P_in)
        return raw[:, :, :P_use].permute(0, 2, 1, 3, 4).contiguous()

    if raw.dim() == 4:
        B, C, Hh, Ww = raw.shape
        if (Hh, Ww) != (H, W) or C % 4 != 0: return None
        P_use = min(P_expected, C // 4)
        return raw[:, : 4 * P_use].view(B, P_use, 4, H, W).contiguous()

    return None

def _decode_dxdy_logwh_cell(BP4: torch.Tensor, H: int, W: int, img_size: int) -> torch.Tensor | None:
    """
    Input  BP4: [B,P,4,H,W] with (dx,dy,logw,logh) in CELL units
    Output XYXY: [B,P,4,H,W] in pixels, clamped to [0, img_size-1]
    """
    if BP4 is None or BP4.dim() != 5 or BP4.shape[2] != 4: return None
    B, P, _, Hh, Ww = BP4.shape
    if (Hh, Ww) != (H, W): return None

    cw, ch = img_size / W, img_size / H
    i = torch.arange(W, device=BP4.device, dtype=torch.float32).view(1, 1, 1, W)
    j = torch.arange(H, device=BP4.device, dtype=torch.float32).view(1, 1, H, 1)
    cx0, cy0 = (i + 0.5) * cw, (j + 0.5) * ch

    dx, dy = BP4[:, :, 0].float(), BP4[:, :, 1].float()
    lw, lh = BP4[:, :, 2].float().clamp(-8, 8), BP4[:, :, 3].float().clamp(-8, 8)

    cx, cy = cx0 + dx * cw, cy0 + dy * ch
    w,  h  = torch.exp(lw) * cw, torch.exp(lh) * ch
    x1, y1 = (cx - 0.5 * w).clamp(0, img_size - 1), (cy - 0.5 * h).clamp(0, img_size - 1)
    x2, y2 = (cx + 0.5 * w).clamp(0, img_size - 1), (cy + 0.5 * h).clamp(0, img_size - 1)
    return torch.stack([x1, y1, x2, y2], dim=2)

def _ensure_min_box(x1: float, y1: float, x2: float, y2: float,
                    img_size: int, min_frac: float = 0.04):
    """
    Enforce a minimum box size as a fraction of the image and clamp to bounds.
    """
    min_w, min_h = img_size * min_frac, img_size * min_frac
    if (x2 - x1) < min_w:
        cx = 0.5 * (x1 + x2); x1, x2 = cx - 0.5 * min_w, cx + 0.5 * min_w
    if (y2 - y1) < min_h:
        cy = 0.5 * (y1 + y2); y1, y2 = cy - 0.5 * min_h, cy + 0.5 * min_h
    return max(0, x1), max(0, y1), min(img_size - 1, x2), min(img_size - 1, y2)

def _box_from_heatmap(logit_hw: torch.Tensor, img_size: int,
                      gamma: float = 3.0, temperature: float = 0.5):
    """
    Heatmap → Gaussian-ish box by moment matching (Colab parity).
    """
    H, W = logit_hw.shape
    cw, ch = img_size / W, img_size / H
    prob = torch.softmax((logit_hw.float() / max(1e-6, temperature)).flatten(), dim=0).view(H, W)

    r = torch.arange(H, device=logit_hw.device, dtype=torch.float32)
    c = torch.arange(W, device=logit_hw.device, dtype=torch.float32)
    pr, pc = prob.sum(1), prob.sum(0)

    mu_r, mu_c = (pr * r).sum(), (pc * c).sum()
    var_r = (pr * (r - mu_r).pow(2)).sum().clamp_min(1e-6)
    var_c = (pc * (c - mu_c).pow(2)).sum().clamp_min(1e-6)
    std_r, std_c = torch.sqrt(var_r), torch.sqrt(var_c)

    cx, cy = (mu_c + 0.5) * cw, (mu_r + 0.5) * ch
    w,  h  = (2.0 * gamma * std_c * cw).item(), (2.0 * gamma * std_r * ch).item()

    x1, y1 = cx - 0.5 * w, cy - 0.5 * h
    x2, y2 = cx + 0.5 * w, cy + 0.5 * h
    return max(0, x1), max(0, y1), min(img_size - 1, x2), min(img_size - 1, y2)

# ============================================================
# Vehicle box decoding (supports both head formats)
# ============================================================

def _veh_box_from_any(veh_pred: torch.Tensor, img_size: int):
    """
    Supports:
      [B,4] normalized (cx,cy,w,h)  OR
      [B,4,H,W] (dx,dy,logw,logh) pick best cell via crude size proxy
    """
    if veh_pred is None or not torch.is_tensor(veh_pred): return None

    # [B,4] case — assume normalized in 0..1 (like Colab-head export)
    if veh_pred.dim() == 2 and veh_pred.shape[1] == 4:
        sig = torch.sigmoid(veh_pred.float())
        cx, cy, w, h = sig[:, 0] * img_size, sig[:, 1] * img_size, sig[:, 2] * img_size, sig[:, 3] * img_size
        x1, y1 = (cx - 0.5 * w).clamp(0, img_size - 1), (cy - 0.5 * h).clamp(0, img_size - 1)
        x2, y2 = (cx + 0.5 * w).clamp(0, img_size - 1), (cy + 0.5 * h).clamp(0, img_size - 1)
        return torch.stack([x1, y1, x2, y2], dim=1)

    # [B,4,H,W] case — choose the best anchor cell using size proxy
    if veh_pred.dim() == 4 and veh_pred.shape[1] == 4:
        B, _, H, W = veh_pred.shape
        cw, ch = img_size / W, img_size / H
        outs = []
        for b in range(B):
            dx, dy = veh_pred[b, 0].float(), veh_pred[b, 1].float()
            lw, lh = veh_pred[b, 2].float().clamp(-8, 8), veh_pred[b, 3].float().clamp(-8, 8)
            score = lw + lh
            gi, gj = torch.nonzero(score == score.max(), as_tuple=False)[0].tolist()
            cx0, cy0 = (gj + 0.5) * cw, (gi + 0.5) * ch
            cx, cy = cx0 + float(dx[gi, gj]) * cw, cy0 + float(dy[gi, gj]) * ch
            wpx, hpx = float(torch.exp(lw[gi, gj])) * cw, float(torch.exp(lh[gi, gj])) * ch
            x1, y1, x2, y2 = cx - 0.5 * wpx, cy - 0.5 * hpx, cx + 0.5 * wpx, cy + 0.5 * hpx
            outs.append(_ensure_min_box(x1, y1, x2, y2, img_size, min_frac=0.10))
        return torch.tensor(outs, device=veh_pred.device, dtype=torch.float32)

    return None

# ============================================================
# Vehicle box from parts (weighted quantiles) + selection helper
# ============================================================

def _derive_vehicle_box_quantile(parts_boxes: List[Iterable[float]],
                                 parts_scores: List[float],
                                 img_size: int,
                                 trim: float = 0.10):
    """
    Robust vehicle box from part boxes using weighted quantiles (Colab parity).
    Returns (x1,y1,x2,y2) or None.
    """
    if not parts_boxes: return None

    xs1 = torch.tensor([b[0] for b in parts_boxes], dtype=torch.float32)
    ys1 = torch.tensor([b[1] for b in parts_boxes], dtype=torch.float32)
    xs2 = torch.tensor([b[2] for b in parts_boxes], dtype=torch.float32)
    ys2 = torch.tensor([b[3] for b in parts_boxes], dtype=torch.float32)
    wts = torch.tensor(parts_scores, dtype=torch.float32).clamp_min(1e-6)
    wts = wts / wts.sum()

    def _wq(v: torch.Tensor, q: float) -> float:
        s, idx = torch.sort(v)
        cw = torch.cumsum(wts[idx], dim=0)
        j = torch.searchsorted(cw, v.new_tensor([q]), right=True).clamp_max(len(s) - 1)[0]
        return float(s[int(j)].item())

    x1 = _wq(xs1, trim);      y1 = _wq(ys1, trim)
    x2 = _wq(xs2, 1.0 - trim); y2 = _wq(ys2, 1.0 - trim)
    return _ensure_min_box(x1, y1, x2, y2, img_size, min_frac=0.10)

def select_vehicle_box(
    *,
    img_size: int,
    veh_box_pred: Optional[torch.Tensor] = None,   # [B,4] or [B,4,H,W]
    parts_boxes: Optional[List[Iterable[float]]] = None,
    parts_scores: Optional[List[float]] = None,
    any_part_heat: Optional[torch.Tensor] = None,  # [H,W] (max over parts)
    gamma_vehicle: float = 1.4,
    temp_vehicle: float = 0.9,
    min_frac: float = 0.10,
    max_frac: float = 0.95,
    prefer: Optional[Dict[str, int]] = None        # lower is more preferred
) -> Tuple[Optional[Tuple[float,float,float,float]], str]:
    """
    Multi-source candidate policy (Colab): veh-head → parts-quantile → any-part-heat.
    Returns (veh_box, source_tag). Non-breaking: engine can keep using _veh_box_from_any if desired.
    """
    prefer = prefer or {"veh-head": 0, "parts-quantile": 1, "any-part-heat": 2}

    candidates: List[Tuple[str, Tuple[float,float,float,float], bool, float]] = []

    # 1) Vehicle head
    veh_from_head = None
    if isinstance(veh_box_pred, torch.Tensor):
        vb = _veh_box_from_any(veh_box_pred, img_size)
        if isinstance(vb, torch.Tensor) and vb.dim() == 2 and vb.shape[1] == 4:
            x1, y1, x2, y2 = [float(v) for v in vb[0].tolist()]
            veh_from_head = (x1, y1, x2, y2)

    # 2) Parts-quantile (weighted)
    veh_from_parts = None
    if parts_boxes:
        veh_from_parts = _derive_vehicle_box_quantile(parts_boxes, parts_scores or [1.0]*len(parts_boxes), img_size)

    # 3) Any-part heat → box
    veh_from_heat = None
    if isinstance(any_part_heat, torch.Tensor):
        x1, y1, x2, y2 = _box_from_heatmap(any_part_heat, img_size, gamma=gamma_vehicle, temperature=temp_vehicle)
        veh_from_heat = _ensure_min_box(x1, y1, x2, y2, img_size, min_frac=min_frac)

    # Validate + score each candidate: prefer ~35% area of the frame and good coverage of parts envelope
    def _validate(b: Optional[Tuple[float,float,float,float]]) -> Tuple[bool, float]:
        if b is None: return False, 0.0
        area_frac = _box_area(b) / float(img_size * img_size)
        if area_frac < min_frac or area_frac > max_frac: return False, 0.0
        cov, score = 0.0, 0.0
        if parts_boxes:
            xs1 = min(x1 for x1,_,_,_ in parts_boxes); ys1 = min(y1 for _,y1,_,_ in parts_boxes)
            xs2 = max(x2 for _,_,x2,_ in parts_boxes); ys2 = max(y2 for _,_,_,y2 in parts_boxes)
            env = (xs1, ys1, xs2, ys2)
            cov = _iou(b, env)  # “coverage” of the parts envelope
        score = (1.0 - abs(area_frac - 0.35)) + 2.0 * cov
        return (cov >= 0.2), score

    if veh_from_head is not None:
        ok, sc = _validate(veh_from_head); candidates.append(("veh-head", veh_from_head, ok, sc))
    if veh_from_parts is not None:
        ok, sc = _validate(veh_from_parts); candidates.append(("parts-quantile", veh_from_parts, ok, sc))
    if veh_from_heat is not None:
        ok, sc = _validate(veh_from_heat); candidates.append(("any-part-heat", veh_from_heat, ok, sc))

    if not candidates:
        return None, "none"

    # Prefer valid first, then higher score; tie-break by preference order
    candidates.sort(key=lambda t: (not t[2], -t[3], prefer.get(t[0], 99)))
    src, box, _, _ = candidates[0]
    x1, y1, x2, y2 = _ensure_min_box(*box, img_size, min_frac=min_frac)
    return (x1, y1, x2, y2), src

# ============================================================
# Logit blending + cascade (engine signatures kept)
# ============================================================

def _safe_blend(voter: torch.Tensor | None, head: torch.Tensor | None, alpha: float):
    if voter is None and head is None: return None
    if voter is None: return head
    if head is None: return voter
    a = float(max(0.0, min(1.0, alpha)))
    return a * voter + (1.0 - a) * head

def blend_logits(out: dict, CFG):
    lt = _safe_blend(out.get("voter_type"),  out.get("type_logits"),  float(_cfg(CFG, "alpha_type",  0.5)))
    lm = _safe_blend(out.get("voter_make"),  out.get("make_logits"),  float(_cfg(CFG, "alpha_make",  0.5)))
    lk = _safe_blend(out.get("voter_model"), out.get("model_logits"), float(_cfg(CFG, "alpha_model", 0.5)))
    return lt, lm, lk

def cascade_infer(
    lt: torch.Tensor | None,
    lm: torch.Tensor | None,
    lk: torch.Tensor | None,
    CFG,
    allowed_m_by_t: dict[int, List[int]] | None = None,
    allowed_k_by_m: dict[int, List[int]] | None = None,
):
    Tt, Tm, Tk     = float(_cfg(CFG, "temp_type", 1.0)), float(_cfg(CFG, "temp_make", 1.0)), float(_cfg(CFG, "temp_model", 1.0))
    tau_t, tau_m, tau_k = float(_cfg(CFG, "tau_type", 0.70)), float(_cfg(CFG, "tau_make", 0.70)), float(_cfg(CFG, "tau_model", 0.70))
    use_cascade    = bool(_cfg(CFG, "use_cascade", True))

    if lt is None:
        return {"type": None, "make": None, "model": None, "confs": (), "stop": "no-type"}

    pt = torch.softmax(lt.float() / max(1e-6, Tt), dim=-1)
    t  = int(pt.argmax(dim=-1)[0]); ct = float(pt[0, t])
    if (not use_cascade) or (lm is None) or (ct < tau_t):
        return {"type": t, "make": None, "model": None, "confs": (ct,), "stop": "type" if ct < tau_t else "nocascade"}

    allowed_m = (allowed_m_by_t or {}).get(t, None)
    if allowed_m is not None:
        if len(allowed_m) == 0:
            return {"type": t, "make": None, "model": None, "confs": (ct,), "stop": "type-nomakes"}
        neg_inf = torch.finfo(lm.dtype).min
        mask_m = torch.full_like(lm, neg_inf)
        mask_m[:, allowed_m] = 0.0
        pm = torch.softmax((lm.float() + mask_m) / max(1e-6, Tm), dim=-1)
    else:
        pm = torch.softmax(lm.float() / max(1e-6, Tm), dim=-1)

    m  = int(pm.argmax(dim=-1)[0]); cm = float(pm[0, m])
    if (lk is None) or (cm < tau_m):
        return {"type": t, "make": None, "model": None, "confs": (ct, cm), "stop": "make"}

    allowed_k = (allowed_k_by_m or {}).get(m, [])
    if allowed_k:
        neg_inf_k = torch.finfo(lk.dtype).min
        mask_k = torch.full_like(lk, neg_inf_k)
        mask_k[:, allowed_k] = 0.0
        pk = torch.softmax((lk.float() + mask_k) / max(1e-6, Tk), dim=-1)
    else:
        pk = torch.softmax(lk.float() / max(1e-6, Tk), dim=-1)

    k  = int(pk.argmax(dim=-1)[0]); ck = float(pk[0, k])
    if ck < tau_k:
        return {"type": t, "make": m, "model": None, "confs": (ct, cm, ck), "stop": "model"}

    return {"type": t, "make": m, "model": k, "confs": (ct, cm, ck), "stop": "ok"}

# ============================================================
# Grid assert (unchanged engine contract)
# ============================================================

def assert_head_grid_matches(CFG, part_logits: torch.Tensor | None = None, model=None):
    if torch.is_tensor(part_logits):
        H, W = int(part_logits.shape[-2]), int(part_logits.shape[-1])
        if getattr(CFG, "_head_grid", None) is not None and (H, W) != tuple(CFG._head_grid):
            print(f"[warn] head grid drift: part_logits {(H, W)} vs CFG._head_grid {CFG._head_grid}")
        else:
            CFG._head_grid = (H, W)
        return (H, W)

    if getattr(CFG, "_head_grid", None) is not None:
        return tuple(CFG._head_grid)

    if hasattr(model, "head_grid"):
        CFG._head_grid = tuple(model.head_grid(int(_cfg(CFG, "img_size", 640))))
        return CFG._head_grid

    s = int(_cfg(CFG, "img_size", 640))
    CFG._head_grid = (s // 32, s // 32)
    return CFG._head_grid

# ============================================================
# Extra small utilities (non-breaking, used by FE/ALPR/Color)
# ============================================================

def crop_by_xyxy(img_pil, box_xyxy: Iterable[float]):
    """
    Convenience crop helper for PIL images (kept simple for engine import sites).
    """
    from PIL import Image
    if img_pil is None or box_xyxy is None: return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    return img_pil.crop((x1, y1, x2, y2))
