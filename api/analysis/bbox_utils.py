from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn.functional as F

# ---------- CFG helper ----------
def _cfg(CFG, name, default):
    return getattr(CFG, name, default) if hasattr(CFG, name) else default

# ---------- Letterbox (Tensor-first; accepts PIL via engine wrapper) ----------
def letterbox_fit(image: torch.Tensor, size: int):
    """
    Resize with aspect ratio into square 'size' canvas with equal padding.
    Accepts CHW, BCHW, or HWC. Returns (sq_image, (pad_w_left, pad_h_top), scale).
    """
    if not torch.is_tensor(image):
        raise TypeError("letterbox_fit expects a torch.Tensor")

    is_batched = image.dim() == 4
    is_chw     = (image.dim() == 3 and image.shape[0] in (1,3))
    is_hwc     = (image.dim() == 3 and image.shape[-1] in (1,3))

    if is_batched: x = image
    elif is_chw:   x = image.unsqueeze(0)
    elif is_hwc:   x = image.permute(2,0,1).unsqueeze(0)
    else:          raise ValueError("Unsupported tensor rank")

    B, C, H, W = x.shape
    s = float(size)
    scale = min(s/max(1.0,W), s/max(1.0,H))
    new_w, new_h = int(round(W*scale)), int(round(H*scale))

    mode = "bilinear" if C > 1 else "nearest"
    x_resized = F.interpolate(x.float(), size=(new_h, new_w), mode=mode, align_corners=False if mode=="bilinear" else None)

    pw_total, ph_total = max(0, size-new_w), max(0, size-new_h)
    left, right = pw_total//2, pw_total - pw_total//2
    top,  bottom= ph_total//2, ph_total - ph_total//2
    x_padded = F.pad(x_resized, (left, right, top, bottom), value=0.0)

    if is_batched: sq = x_padded
    elif is_chw:   sq = x_padded.squeeze(0)
    else:          sq = x_padded.squeeze(0).permute(1,2,0)
    return sq, (left, top), float(scale)

def invert_letterbox_box(box_on_sq, orig_w: int, orig_h: int, pad: tuple, scale: float):
    if box_on_sq is None: return None
    x1,y1,x2,y2 = [float(v) for v in (box_on_sq.tolist() if isinstance(box_on_sq, torch.Tensor) else box_on_sq)]
    pw, ph = float(pad[0]), float(pad[1]); s = max(1e-12, float(scale))
    xo1, yo1 = (x1-pw)/s, (y1-ph)/s
    xo2, yo2 = (x2-pw)/s, (y2-ph)/s
    xo1 = max(0.0, min(xo1, orig_w-1.0)); yo1 = max(0.0, min(yo1, orig_h-1.0))
    xo2 = max(0.0, min(xo2, orig_w-1.0)); yo2 = max(0.0, min(yo2, orig_h-1.0))
    return (xo1, yo1, xo2, yo2)

# ---------- Decode helpers ----------
def _unpack_bbox_to_BP4HW(raw: torch.Tensor, P_expected: int, H: int, W: int) -> torch.Tensor | None:
    if raw is None or not torch.is_tensor(raw) or raw.dim() not in (4,5): return None
    if raw.dim()==5 and raw.shape[2]==4:
        B,P_in,_,Hh,Ww = raw.shape
        if (Hh,Ww)!=(H,W): return None
        return raw[:,:min(P_expected,P_in)].contiguous()
    if raw.dim()==5 and raw.shape[1]==4:
        B,_,P_in,Hh,Ww = raw.shape
        if (Hh,Ww)!=(H,W): return None
        P_use = min(P_expected, P_in)
        return raw[:,:,:P_use].permute(0,2,1,3,4).contiguous()
    if raw.dim()==4:
        B,C,Hh,Ww = raw.shape
        if (Hh,Ww)!=(H,W) or C%4!=0: return None
        P_use = min(P_expected, C//4)
        return raw[:, :4*P_use].view(B, P_use, 4, H, W).contiguous()
    return None

def _decode_dxdy_logwh_cell(BP4: torch.Tensor, H: int, W: int, img_size: int) -> torch.Tensor | None:
    if BP4 is None or BP4.dim()!=5 or BP4.shape[2]!=4: return None
    B,P,_,Hh,Ww = BP4.shape
    if (Hh,Ww)!=(H,W): return None
    cw, ch = img_size/W, img_size/H
    i = torch.arange(W, device=BP4.device, dtype=torch.float32).view(1,1,1,W)
    j = torch.arange(H, device=BP4.device, dtype=torch.float32).view(1,1,H,1)
    cx0, cy0 = (i+0.5)*cw, (j+0.5)*ch
    dx, dy = BP4[:,:,0].float(), BP4[:,:,1].float()
    lw, lh = BP4[:,:,2].float().clamp(-8,8), BP4[:,:,3].float().clamp(-8,8)
    cx, cy = cx0 + dx*cw, cy0 + dy*ch
    w,  h  = torch.exp(lw)*cw, torch.exp(lh)*ch
    x1, y1 = (cx-0.5*w).clamp(0, img_size-1), (cy-0.5*h).clamp(0, img_size-1)
    x2, y2 = (cx+0.5*w).clamp(0, img_size-1), (cy+0.5*h).clamp(0, img_size-1)
    return torch.stack([x1,y1,x2,y2], dim=2)

def _ensure_min_box(x1, y1, x2, y2, img_size: int, min_frac: float=0.04):
    min_w, min_h = img_size*min_frac, img_size*min_frac
    if (x2-x1)<min_w:
        cx = 0.5*(x1+x2); x1, x2 = cx-0.5*min_w, cx+0.5*min_w
    if (y2-y1)<min_h:
        cy = 0.5*(y1+y2); y1, y2 = cy-0.5*min_h, cy+0.5*min_h
    return max(0,x1), max(0,y1), min(img_size-1,x2), min(img_size-1,y2)

def _box_from_heatmap(logit_hw: torch.Tensor, img_size: int, gamma: float=3.0, temperature: float=0.5):
    H,W = logit_hw.shape
    cw, ch = img_size/W, img_size/H
    prob = torch.softmax((logit_hw.float()/max(1e-6,temperature)).flatten(), dim=0).view(H,W)
    r = torch.arange(H, device=logit_hw.device, dtype=torch.float32)
    c = torch.arange(W, device=logit_hw.device, dtype=torch.float32)
    pr, pc = prob.sum(1), prob.sum(0)
    mu_r, mu_c = (pr*r).sum(), (pc*c).sum()
    var_r = (pr*(r-mu_r).pow(2)).sum().clamp_min(1e-6)
    var_c = (pc*(c-mu_c).pow(2)).sum().clamp_min(1e-6)
    std_r, std_c = torch.sqrt(var_r), torch.sqrt(var_c)
    cx, cy = (mu_c+0.5)*cw, (mu_r+0.5)*ch
    w, h = (2.0*gamma*std_c*cw).item(), (2.0*gamma*std_r*ch).item()
    x1,y1,x2,y2 = cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h
    return max(0,x1), max(0,y1), min(img_size-1,x2), min(img_size-1,y2)

def _veh_box_from_any(veh_pred: torch.Tensor, img_size: int):
    if veh_pred is None or not torch.is_tensor(veh_pred): return None
    if veh_pred.dim()==2 and veh_pred.shape[1]==4:
        sig = torch.sigmoid(veh_pred.float())
        cx,cy,w,h = sig[:,0]*img_size, sig[:,1]*img_size, sig[:,2]*img_size, sig[:,3]*img_size
        x1,y1 = (cx-0.5*w).clamp(0,img_size-1), (cy-0.5*h).clamp(0,img_size-1)
        x2,y2 = (cx+0.5*w).clamp(0,img_size-1), (cy+0.5*h).clamp(0,img_size-1)
        return torch.stack([x1,y1,x2,y2], dim=1)
    if veh_pred.dim()==4 and veh_pred.shape[1]==4:
        B,_,H,W = veh_pred.shape
        cw, ch = img_size/W, img_size/H
        outs=[]
        for b in range(B):
            dx, dy = veh_pred[b,0].float(), veh_pred[b,1].float()
            lw, lh = veh_pred[b,2].float().clamp(-8,8), veh_pred[b,3].float().clamp(-8,8)
            score = lw+lh
            gi, gj = torch.nonzero(score==score.max(), as_tuple=False)[0].tolist()
            cx0, cy0 = (gj+0.5)*cw, (gi+0.5)*ch
            cx, cy = cx0+float(dx[gi,gj])*cw, cy0+float(dy[gi,gj])*ch
            wpx, hpx = float(torch.exp(lw[gi,gj]))*cw, float(torch.exp(lh[gi,gj]))*ch
            x1,y1,x2,y2 = cx-0.5*wpx, cy-0.5*hpx, cx+0.5*wpx, cy+0.5*hpx
            outs.append(_ensure_min_box(x1,y1,x2,y2,img_size, min_frac=0.10))
        return torch.tensor(outs, device=veh_pred.device, dtype=torch.float32)
    return None

# ---------- Logit blending + cascade ----------
def _safe_blend(voter, head, alpha: float):
    if voter is None and head is None: return None
    if voter is None: return head
    if head  is None: return voter
    a = float(max(0.0, min(1.0, alpha))); return a*voter + (1.0-a)*head

def blend_logits(out: dict, CFG):
    lt = _safe_blend(out.get("voter_type"),  out.get("type_logits"),  float(_cfg(CFG,"alpha_type",  0.5)))
    lm = _safe_blend(out.get("voter_make"),  out.get("make_logits"),  float(_cfg(CFG,"alpha_make",  0.5)))
    lk = _safe_blend(out.get("voter_model"), out.get("model_logits"), float(_cfg(CFG,"alpha_model", 0.5)))
    return lt, lm, lk

def cascade_infer(lt, lm, lk, CFG, allowed_m_by_t: dict[int,List[int]]|None=None, allowed_k_by_m: dict[int,List[int]]|None=None):
    Tt, Tm, Tk = float(_cfg(CFG,"temp_type",1.0)), float(_cfg(CFG,"temp_make",1.0)), float(_cfg(CFG,"temp_model",1.0))
    tau_t, tau_m, tau_k = float(_cfg(CFG,"tau_type",0.70)), float(_cfg(CFG,"tau_make",0.70)), float(_cfg(CFG,"tau_model",0.70))
    use_cascade = bool(_cfg(CFG,"use_cascade",True))
    if lt is None: return {"type":None,"make":None,"model":None,"confs":(), "stop":"no-type"}

    pt = torch.softmax(lt.float()/max(1e-6,Tt), dim=-1)
    t  = int(pt.argmax(dim=-1)[0]); ct = float(pt[0,t])
    if (not use_cascade) or (lm is None) or (ct < tau_t):
        return {"type":t, "make":None, "model":None, "confs":(ct,), "stop":"type" if ct<tau_t else "nocascade"}

    allowed_m = (allowed_m_by_t or {}).get(t, None)
    if allowed_m is not None:
        neg_inf = torch.finfo(lm.dtype).min
        mask_m = torch.full_like(lm, neg_inf)
        if len(allowed_m)>0: mask_m[:, allowed_m] = 0.0
        pm = torch.softmax((lm.float()+mask_m)/max(1e-6,Tm), dim=-1)
    else:
        pm = torch.softmax(lm.float()/max(1e-6,Tm), dim=-1)
    m  = int(pm.argmax(dim=-1)[0]); cm = float(pm[0,m])
    if (lk is None) or (cm < tau_m): return {"type":t,"make":None,"model":None,"confs":(ct,cm), "stop":"make"}

    allowed_k = (allowed_k_by_m or {}).get(m, [])
    if allowed_k:
        neg_inf_k = torch.finfo(lk.dtype).min
        mask_k = torch.full_like(lk, neg_inf_k)
        mask_k[:, allowed_k] = 0.0
        pk = torch.softmax((lk.float()+mask_k)/max(1e-6,Tk), dim=-1)
    else:
        pk = torch.softmax(lk.float()/max(1e-6,Tk), dim=-1)
    k  = int(pk.argmax(dim=-1)[0]); ck = float(pk[0,k])
    if ck < tau_k: return {"type":t,"make":m,"model":None,"confs":(ct,cm,ck), "stop":"model"}
    return {"type":t,"make":m,"model":k,"confs":(ct,cm,ck), "stop":"ok"}

# ---------- Grid assert ----------
def assert_head_grid_matches(CFG, part_logits: torch.Tensor | None=None, model=None):
    if torch.is_tensor(part_logits):
        H,W = int(part_logits.shape[-2]), int(part_logits.shape[-1])
        if getattr(CFG,"_head_grid",None) is not None and (H,W)!=tuple(CFG._head_grid):
            print(f"[warn] head grid drift: part_logits {(H,W)} vs CFG._head_grid {CFG._head_grid}")
        else:
            CFG._head_grid = (H,W)
        return (H,W)
    if getattr(CFG,"_head_grid",None) is not None:
        return tuple(CFG._head_grid)
    if hasattr(model,"head_grid"):
        CFG._head_grid = tuple(model.head_grid(int(_cfg(CFG,"img_size",640))))
        return CFG._head_grid
    s = int(_cfg(CFG,"img_size",640)); CFG._head_grid=(s//32,s//32); return CFG._head_grid
