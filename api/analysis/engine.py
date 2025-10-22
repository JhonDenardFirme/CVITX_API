import os, io, json, time, hashlib, zipfile
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
from io import BytesIO

# Light deps (installed in step 3)
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image

# ---- Local logger (kept) ------------------------------------------------------
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"

class JsonLogger:
    def __init__(self, name="cvitx-engine"):
        self.name = name
    def _emit(self, level, event, **fields):
        print(json.dumps({"ts":_now_iso(),"level":level,"svc":self.name,"event":event,**fields}), flush=True)
    def info(self, event, **fields):  self._emit("INFO", event, **fields)
    def warn(self, event, **fields):  self._emit("WARN", event, **fields)
    def error(self, event, **fields): self._emit("ERROR", event, **fields)

LOG = JsonLogger("cvitx-engine")

# ==============================================================================
# PHASE 2 — SSOT + label_maps adoption (+ allow-lists, regions/parts)
# ==============================================================================

# --- Regions & SSOT hierarchy (trimmed to keep file compact; add freely later) ---
REGIONS: List[str] = ["Front","Side","Rear","Roof"]

vehicle_hierarchy: Dict[str, Dict[str, List[str]]] = {
    "Car": {
        "Toyota": ["Wigo","Vios","CorollaAltis","Camry","GRYaris"],
        "Honda":  ["Brio","City","Civic","CivicTypeR"],
        "Ford":   ["Mustang"],
        "Nissan": ["Almera","GT-R"],
    },
    "SUV": {
        "Toyota": ["Raize","YarisCross","CorollaCross","Rush","Fortuner","Innova"],
        "Ford":   ["Everest","Territory"],
        "Honda":  ["BR-V","CR-V","HR-V"],
    },
    "Pickup": {
        "Toyota": ["Hilux"],
        "Ford":   ["RangerWildtrak","RangerRaptor"],
    },
    "Van": { "Toyota": ["HiAce"], "Hyundai": ["Staria"] },
    "Utility": { "Mitsubishi":["L300"] },
    "Motorcycle": { "HondaMC":["BeAT","Click","PCX"] },
    "Bus": {}, "LightTruck": {}, "ContainerTruck": {}, "SpecialVehicle": {},
    "Bicycle": {}, "E-Bike": {}, "Pedicab": {}, "Tricycle": {}, "Jeepney": {}, "E-Jeepney": {}, "CarouselBus": {}
}

# Part classes by type (subset for compactness; OK to expand later without code changes)
part_classes_by_type: Dict[str, List[Tuple[str,str]]] = {
    "Car": [
        ("Bumper","Front"),("Grille","Front"),("Hood","Front"),
        ("LeftHeadlight","Front"),("RightHeadlight","Front"),("Plate","Front"),
        ("Windshield","Front"),("Roof","Roof"),
        ("LeftTaillight","Rear"),("RightTaillight","Rear"),("Plate","Rear"),
        ("FrontDoor","Side"),("RearDoor","Side"),("FrontWheel","Side"),("RearWheel","Side"),
    ],
    "SUV": [
        ("Bumper","Front"),("Grille","Front"),("Hood","Front"),
        ("LeftHeadlight","Front"),("RightHeadlight","Front"),("Plate","Front"),
        ("Roof","Roof"),
        ("LeftTaillight","Rear"),("RightTaillight","Rear"),("Plate","Rear"),
        ("FrontDoor","Side"),("RearDoor","Side"),("FrontWheel","Side"),("RearWheel","Side"),
    ],
    "Pickup": [
        ("Bumper","Front"),("Grille","Front"),("Hood","Front"),
        ("LeftHeadlight","Front"),("RightHeadlight","Front"),("Plate","Front"),
        ("Roof","Roof"),
        ("LeftTaillight","Rear"),("RightTaillight","Rear"),("Plate","Rear"),
        ("FrontDoor","Side"),("RearWheel","Side"),("FrontWheel","Side"),
    ],
    "Motorcycle": [
        ("MainHeadlight","Front"),("Plate","Rear"),("FrontWheel","Side"),("RearWheel","Side"),("Seat","Side")
    ],
}

# Canonical TYPE list (fixed order)
TYPES: List[str] = [
    "Car","SUV","Van","Pickup","Utility","Motorcycle","Bicycle",
    "E-Bike","Pedicab","Tricycle","Jeepney","E-Jeepney",
    "Bus","CarouselBus","LightTruck","ContainerTruck","SpecialVehicle"
]

# Simple canonicalizations for light/taillight
_LIGHT_FIX = {
    "HeadLight":"Headlight","LeftHeadLight":"LeftHeadlight","RightHeadLight":"RightHeadlight",
    "TailLight":"Taillight","LeftTailLight":"LeftTaillight","RightTailLight":"RightTaillight",
}
def _canon_part(p:str)->str: return _LIGHT_FIX.get(p,p)

def compute_region_part_maps(per_type_pairs: Dict[str, List[Tuple[str, str]]]):
    type_region_to_parts, type_part_to_regions = {}, {}
    for vtype, pairs in (per_type_pairs or {}).items():
        reg2parts, part2regs = {}, {}
        for part, region in pairs:
            pp = _canon_part(part)
            reg2parts.setdefault(region, set()).add(pp)
            part2regs.setdefault(pp, set()).add(region)
        type_region_to_parts[vtype] = reg2parts
        type_part_to_regions[vtype] = part2regs
    return type_region_to_parts, type_part_to_regions

TYPE_REGION_TO_PARTS, TYPE_PART_TO_REGIONS = compute_region_part_maps(part_classes_by_type)

def _build_vocab(names: List[str]) -> "OrderedDict[str,int]":
    return OrderedDict((k, i) for i, k in enumerate(list(names)))

def _derive_makes_models_from_hierarchy(h: dict) -> Tuple[List[str], List[str]]:
    seen_makes, seen_models = OrderedDict(), OrderedDict()
    for _t, mkdict in (h or {}).items():
        for mk, models in (mkdict or {}).items():
            if mk not in seen_makes: seen_makes[mk] = True
            for md in (models or []):
                md2 = str(md).replace(" ","")
                if md2 not in seen_models: seen_models[md2]=True
    return list(seen_makes.keys()), list(seen_models.keys())

def build_vocab_from_ssot() -> Dict[str, Any]:
    _seen_parts = OrderedDict()
    for vtype, pairs in part_classes_by_type.items():
        for part,_r in pairs: _seen_parts[_canon_part(part)] = True
    parts = list(_seen_parts.keys())
    makes, models = _derive_makes_models_from_hierarchy(vehicle_hierarchy)
    return {
        "TYPES": TYPES, "MAKES": makes, "MODELS": models, "PARTS": parts, "REGIONS": REGIONS,
        "type2idx": _build_vocab(TYPES),
        "make2idx": _build_vocab(makes),
        "model2idx": _build_vocab(models),
        "part2idx": _build_vocab(parts),
        "region2idx": _build_vocab(REGIONS),
    }

def _coerce_idx_map(maybe)->Dict[str,int]:
    if maybe is None: return {}
    if isinstance(maybe, dict) and all(isinstance(v,(int,float)) for v in maybe.values()):
        return {str(k):int(v) for k,v in maybe.items()}
    if isinstance(maybe, dict) and "idx2name" in maybe and isinstance(maybe["idx2name"],(list,tuple)):
        return {str(name):i for i,name in enumerate(list(maybe["idx2name"]))}
    if isinstance(maybe, dict) and "names" in maybe and isinstance(maybe["names"],(list,tuple)):
        return {str(name):i for i,name in enumerate(list(maybe["names"]))}
    if isinstance(maybe,(list,tuple)):
        return {str(name):i for i,name in enumerate(list(maybe))}
    return {}

def _idx_map_to_order(idx_map: Dict[str,int]) -> List[str]:
    idx2name={}
    for k,i in idx_map.items():
        if i not in idx2name: idx2name[i]=k
    return [idx2name[i] for i in sorted(idx2name.keys())]

def sanitize_maps_lights(maps: dict) -> dict:
    out = dict(maps or {})
    if isinstance(out.get("parts"), list):
        out["parts"] = [_canon_part(p) for p in out["parts"]]
    if isinstance(out.get("part2idx"), dict):
        out["part2idx"] = {_canon_part(k): int(v) for k,v in out["part2idx"].items()}
    return out

def build_vocab_from_label_maps(maps: dict) -> Dict[str, Any]:
    if not isinstance(maps, dict): raise ValueError("label_maps must be a dict")
    maps = sanitize_maps_lights(maps)

    t_map = _coerce_idx_map(maps.get("type2idx")  or maps.get("types"))
    m_map = _coerce_idx_map(maps.get("make2idx")  or maps.get("makes"))
    d_map = _coerce_idx_map(maps.get("model2idx") or maps.get("models"))
    p_map = _coerce_idx_map(maps.get("part2idx")  or maps.get("parts"))

    types  = _idx_map_to_order(t_map) if t_map else list(TYPES)
    makes  = _idx_map_to_order(m_map)
    models = _idx_map_to_order(d_map)
    parts  = _idx_map_to_order(p_map)

    return {
        "TYPES": types, "MAKES": makes, "MODELS": models, "PARTS": parts, "REGIONS": REGIONS,
        "type2idx": (t_map or _build_vocab(types)),
        "make2idx": (m_map or _build_vocab(makes)),
        "model2idx": (d_map or _build_vocab(models)),
        "part2idx": (p_map or _build_vocab(parts)),
        "region2idx": _build_vocab(REGIONS),
    }

# Globals adopted by SSOT or bundle
_ssot = build_vocab_from_ssot()
TYPES, MAKES, MODELS, PARTS, REGIONS = (_ssot[k] for k in ("TYPES","MAKES","MODELS","PARTS","REGIONS"))
type2idx, make2idx, model2idx, part2idx, region2idx = (_ssot[k] for k in ("type2idx","make2idx","model2idx","part2idx","region2idx"))

def build_allowlists(type2idx: Dict[str,int], make2idx: Dict[str,int], model2idx: Dict[str,int],
                     ssot_hierarchy: Dict[str, Dict[str, List[str]]]):
    type_to_makes: Dict[str,List[str]] = {}
    make_to_models: Dict[str,List[str]] = {}
    for t_name, mk_dict in (ssot_hierarchy or {}).items():
        mks = list((mk_dict or {}).keys()); type_to_makes[t_name]=mks
        for mk, model_list in (mk_dict or {}).items():
            toks=[str(md).replace(" ","") for md in (model_list or [])]
            make_to_models.setdefault(mk, [])
            for tok in toks:
                if tok not in make_to_models[mk]: make_to_models[mk].append(tok)

    allowed_makes_by_type_idx: Dict[int,List[int]] = {}
    for t_name, m_list in type_to_makes.items():
        if t_name not in type2idx: continue
        allowed_makes_by_type_idx[type2idx[t_name]] = [make2idx[m] for m in m_list if m in make2idx]

    allowed_models_by_make_idx: Dict[int,List[int]] = {}
    for mk, toks in make_to_models.items():
        if mk not in make2idx: continue
        allowed_models_by_make_idx[make2idx[mk]] = [model2idx[t] for t in toks if t in model2idx]

    allowed_models_by_type_idx: Dict[int,List[int]] = {}
    for t_name, mk_list in type_to_makes.items():
        if t_name not in type2idx: continue
        t_id = type2idx[t_name]; seen, mids = set(), []
        for mk in mk_list:
            if mk not in make2idx: continue
            m_id = make2idx[mk]
            for mid in allowed_models_by_make_idx.get(m_id, []):
                if mid not in seen: seen.add(mid); mids.append(mid)
        allowed_models_by_type_idx[t_id] = mids

    return allowed_makes_by_type_idx, allowed_models_by_make_idx, allowed_models_by_type_idx

allowed_makes_by_type_idx, allowed_models_by_make_idx, allowed_models_by_type_idx = build_allowlists(
    type2idx, make2idx, model2idx, vehicle_hierarchy
)

def adopt_label_maps_into_globals(maps: dict, per_type_parts: Optional[Dict[str,List[Tuple[str,str]]]] = None):
    global TYPES, MAKES, MODELS, PARTS, REGIONS
    global type2idx, make2idx, model2idx, part2idx, region2idx
    global TYPE_REGION_TO_PARTS, TYPE_PART_TO_REGIONS
    global allowed_makes_by_type_idx, allowed_models_by_make_idx, allowed_models_by_type_idx

    bundle = build_vocab_from_label_maps(maps)
    TYPES, MAKES, MODELS, PARTS, REGIONS = (bundle[k] for k in ("TYPES","MAKES","MODELS","PARTS","REGIONS"))
    type2idx, make2idx, model2idx, part2idx, region2idx = (bundle[k] for k in ("type2idx","make2idx","model2idx","part2idx","region2idx"))

    if isinstance(per_type_parts, dict) and per_type_parts:
        TYPE_REGION_TO_PARTS, TYPE_PART_TO_REGIONS = compute_region_part_maps(per_type_parts)
    else:
        TYPE_REGION_TO_PARTS, TYPE_PART_TO_REGIONS = compute_region_part_maps(part_classes_by_type)

    allowed_makes_by_type_idx, allowed_models_by_make_idx, allowed_models_by_type_idx = build_allowlists(
        type2idx, make2idx, model2idx, vehicle_hierarchy
    )

    LOG.info("labelmaps_adopted",
             types=len(TYPES), makes=len(MAKES), models=len(MODELS), parts=len(PARTS),
             types_first=TYPES[:3], types_last=TYPES[-3:], aliases_applied=0)

# ==============================================================================
# PHASE 4 — Model builder (backbone + heads sized by vocab)
# ==============================================================================

def _best_gn_groups(C:int)->int:
    for g in (32,16,8,4,2,1):
        if C%g==0: return g
    return 1

class Conv1x1_GN_ReLU(nn.Module):
    def __init__(self,in_ch:int,out_ch:int):
        super().__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        self.gn=nn.GroupNorm(_best_gn_groups(out_ch),out_ch)
        self.act=nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.gn(self.conv(x)))

class GlobalHead(nn.Module):
    def __init__(self,in_ch:int,out_classes:int,hidden_ch:int=None,p_drop:float=0.0):
        super().__init__()
        self.enabled = int(out_classes)>0
        self.out_classes = int(out_classes)
        if not self.enabled:
            self.pre=self.drop=self.fc=None; return
        hidden_ch = hidden_ch or in_ch
        self.pre  = Conv1x1_GN_ReLU(in_ch, hidden_ch)
        self.drop = nn.Dropout(p_drop) if p_drop>0 else nn.Identity()
        self.fc   = nn.Linear(hidden_ch, self.out_classes)
    def forward(self, feats: torch.Tensor):
        if not self.enabled: return None
        x = self.pre(feats); x = F.adaptive_avg_pool2d(x,1).flatten(1); x = self.drop(x)
        return self.fc(x)

class PartBoxHead(nn.Module):
    def __init__(self,in_ch:int,n_parts:int,inter:int=256):
        super().__init__()
        self.n_parts = int(n_parts)
        if self.n_parts<=0:
            self.stem=nn.Identity(); self.part=self.box=None
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, inter, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_best_gn_groups(inter), inter),
                nn.ReLU(inplace=True),
            )
            self.part = nn.Conv2d(inter, self.n_parts,   kernel_size=1)
            self.box  = nn.Conv2d(inter, 4*self.n_parts, kernel_size=1)
    def forward(self, x):
        if self.n_parts<=0: return None, None
        f = self.stem(x); return self.part(f), self.box(f)

class TimmLastFeature(nn.Module):
    def __init__(self, model_name="mobilevitv2_200", pretrained=True, out_index=-1):
        super().__init__()
        self.net = timm.create_model(model_name, features_only=True, pretrained=pretrained, out_indices=(out_index,))
        self.out_ch = self.net.feature_info.channels()[out_index]
    def forward(self, x): return self.net(x)[0]

class CompositionalMobileViT(nn.Module):
    def __init__(self, backbone: nn.Module, feat_ch:int,
                 n_types:int, n_parts:int, n_makes:int=0, n_models:int=0):
        super().__init__()
        self.backbone = backbone
        self.feat_ch  = int(feat_ch)
        self.meta = {"n_types":int(n_types), "n_parts":int(n_parts),
                     "n_makes":int(n_makes), "n_models":int(n_models)}
        self.type_head  = GlobalHead(self.feat_ch, self.meta["n_types"])
        self.make_head  = GlobalHead(self.feat_ch, self.meta["n_makes"])
        self.model_head = GlobalHead(self.feat_ch, self.meta["n_models"])
        self.part_box   = PartBoxHead(self.feat_ch, self.meta["n_parts"])
        self.pres_head  = nn.Conv2d(self.feat_ch, self.meta["n_parts"], kernel_size=1) if self.meta["n_parts"]>0 else None
        self.veh_box_fc = nn.Linear(self.feat_ch, 4)

        self.veh_box_head = nn.Sequential(
            nn.Conv2d(self.feat_ch, self.feat_ch, kernel_size=1, bias=True),
            nn.GroupNorm(_best_gn_groups(self.feat_ch), self.feat_ch),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def head_signature(self)->dict:
        P = self.meta["n_parts"]
        return {"type_head":self.meta["n_types"], "make_head":self.meta["n_makes"],
                "model_head":self.meta["n_models"], "part_head":P, "bbox_head":4*P, "presence":P}

    @torch.no_grad()
    def head_grid(self, img_size:int)->Tuple[int,int]:
        self.eval(); d=next(self.parameters()).device
        feats=self.backbone(torch.zeros(1,3,img_size,img_size, device=d))
        H,W=feats.shape[-2:]; return int(H),int(W)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        type_logits  = self.type_head(feats)
        make_logits  = self.make_head(feats)
        model_logits = self.model_head(feats)
        part_logits, bbox_preds = (None,None)
        if self.meta["n_parts"]>0:
            part_logits, bbox_preds = self.part_box(feats)
        pres_logits = self.pres_head(feats) if self.pres_head is not None else None
        g = self.veh_box_head(feats).flatten(1); veh_box_preds = self.veh_box_fc(g).sigmoid()
        return {"type_logits":type_logits,"make_logits":make_logits,"model_logits":model_logits,
                "part_logits":part_logits,"bbox_preds":bbox_preds,"part_present_logits":pres_logits,
                "veh_box_preds":veh_box_preds}

def _head_rows(head)->Optional[int]:
    if not isinstance(head, nn.Module): return None
    fc = getattr(head, "fc", None)
    if fc is None: return None
    return int(fc.weight.shape[0])

def _model_vocab_signature(model)->Dict[str,Optional[int]]:
    P = getattr(model,"meta",{}).get("n_parts",None)
    return {"type_head":_head_rows(getattr(model,"type_head",None)),
            "make_head":_head_rows(getattr(model,"make_head",None)),
            "model_head":_head_rows(getattr(model,"model_head",None)),
            "part_head":P, "bbox_head":(4*P if P is not None else None), "presence":P}

def _assure_alignment_counts(model, types, makes, models, parts)->bool:
    sig = _model_vocab_signature(model)
    checks = [
        ("types",  sig.get("type_head"),  len(types)),
        ("makes",  sig.get("make_head"),  len(makes) if makes is not None else None),
        ("models", sig.get("model_head"), len(models) if models is not None else None),
        ("parts",  sig.get("part_head"),  len(parts)),
        ("bbox",   sig.get("bbox_head"),  (4*len(parts)) if parts is not None else None),
        ("pres",   sig.get("presence"),   len(parts) if parts is not None else None),
    ]
    ok=True
    for name, have, exp in checks:
        if exp is None or have is None: continue
        good = (int(have)==int(exp)); ok = ok and good
        LOG.info("align_check", head=name, have=int(have), expected=int(exp), ok=good)
    LOG.info("align_summary", ok=ok)
    return ok

def build_model(backbone_name:str)->CompositionalMobileViT:
    n_types, n_parts, n_makes, n_models = len(TYPES), len(PARTS), len(MAKES), len(MODELS)
    bb = TimmLastFeature(model_name=backbone_name, pretrained=True)
    model = CompositionalMobileViT(bb, bb.out_ch, n_types=n_types, n_parts=n_parts,
                                   n_makes=n_makes, n_models=n_models)
    LOG.info("build_model",
             backbone=backbone_name, head_sizes=_model_vocab_signature(model),
             grid=list(model.head_grid(640)))
    return model

# ==============================================================================
# PHASE 3 — Bundle loader (zip/dir) + alignment audits + sha256
# ==============================================================================

def _glob_suffixes(glob_str: str)->List[str]:
    raw = (glob_str or "*.pt|*.pth|*.ckpt").split("|")
    out=[]
    for pat in raw:
        s = pat.strip().lower()
        if s.endswith(".pt"): out.append(".pt")
        if s.endswith(".pth"): out.append(".pth")
        if s.endswith(".ckpt"): out.append(".ckpt")
    return list(dict.fromkeys(out)) or [".pt",".pth",".ckpt"]

def _sha256_bytes(b: bytes)->str: return hashlib.sha256(b).hexdigest()

def _strip_module_prefix(sd: dict)->dict:
    if not isinstance(sd, dict): return sd
    out={}
    for k,v in sd.items():
        k2 = k[7:] if k.startswith("module.") else (k[6:] if k.startswith("model.") else k)
        out[k2]=v
    return out

def _expected_rows_from_vocab()->dict:
    P=len(PARTS)
    return {
        "type_head.fc.weight":  len(TYPES),
        "make_head.fc.weight":  len(MAKES),
        "model_head.fc.weight": len(MODELS),
        "part_box.part.weight": P,
        "part_box.box.weight":  4*P,
        "pres_head.weight":     P,
    }

def _audit_state_vs_vocab(state_dict: dict, strict_shapes: bool)->List[str]:
    errs=[]
    exp=_expected_rows_from_vocab()
    for key, exp_rows in exp.items():
        if key in state_dict and isinstance(state_dict[key], torch.Tensor):
            have=int(state_dict[key].shape[0])
            if have!=int(exp_rows): errs.append(f"{key}: rows={have} != expected={exp_rows}")
    if errs and strict_shapes:
        for e in errs: LOG.error("audit_shape_mismatch", detail=e)
    return errs

def _is_label_maps_dict(d:dict)->int:
    if not isinstance(d, dict): return 0
    keys=set(k.lower() for k in d.keys())
    score=0
    for a,b in [("type2idx","types"),("make2idx","makes"),("model2idx","models"),("part2idx","parts")]:
        if a in keys or b in keys: score+=2
    if sum(1 for k in ("type2idx","types","make2idx","makes","model2idx","models","part2idx","parts") if k in keys)>=3:
        score+=2
    return score

def _dir_find_members(root:str, weight_suffixes:List[str])->Tuple[Optional[str],Optional[str],int,int]:
    json_files, weight_files = [], []
    for r,_d,fs in os.walk(root):
        for f in fs:
            fp=os.path.join(r,f); fl=f.lower()
            if fl.endswith(".json"): json_files.append(fp)
            if any(fl.endswith(sfx) for sfx in weight_suffixes): weight_files.append(fp)
    best_json, best_score = None, -1
    for j in json_files:
        try:
            with open(j,"r",encoding="utf-8") as fh: d=json.load(fh)
            sc=_is_label_maps_dict(d)
            if sc>best_score: best_score, best_json = sc, j
        except Exception: continue
    best_weight=None
    if weight_files:
        cand=[(w, os.path.getsize(w), os.path.basename(w).lower()) for w in weight_files]
        cand.sort(key=lambda x: (not ("best" in x[2] or "ema" in x[2]), -x[1]))
        best_weight=cand[0][0]
    return best_json, best_weight, len(json_files), len(weight_files)

def load_bundle_dir(dir_path:str, cfg:Optional[dict]=None, backbone_name:str="mobilevitv2_200"):
    strict_shapes    = bool((cfg or {}).get("strict_state_shapes", True))
    auto_remap_parts = bool((cfg or {}).get("auto_remap_parts", True))
    weight_suffixes  = _glob_suffixes(str((cfg or {}).get("bundle_weights_glob", "*.pt|*.pth|*.ckpt")))
    report = {"sha256": None, "weights_name": None, "perm_hash": None, "audits": [], "aligned": False, "trained": False, "load_notes": []}

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    LOG.info("bundle_scan_dir", root=dir_path)
    label_path, weight_path, n_json, n_w = _dir_find_members(dir_path, weight_suffixes)
    LOG.info("bundle_scan_result", json_files=n_json, weight_files=n_w, label_path=label_path, weight_path=weight_path)

    if label_path is None:
        raise FileNotFoundError("No usable label_maps.json found in the directory.")
    if weight_path is None:
        raise FileNotFoundError(f"No weights with suffix in {weight_suffixes} found in the directory.")

    with open(label_path,"r",encoding="utf-8") as fh:
        maps=json.load(fh)
    adopt_label_maps_into_globals(maps, per_type_parts=None)
    LOG.info("bundle_adopted", types=len(TYPES), makes=len(MAKES), models=len(MODELS), parts=len(PARTS))

    with open(weight_path,"rb") as fh:
        buf=fh.read()
    report["sha256"]=_sha256_bytes(buf); report["weights_name"]=os.path.basename(weight_path)
    LOG.info("bundle_sha256", sha256=report["sha256"][:32], weights=report["weights_name"])

    model = build_model(backbone_name=backbone_name)
    blob = torch.load(io.BytesIO(buf), map_location="cpu")
    state_dict = blob.get("state_dict", None) if isinstance(blob, dict) else None
    if state_dict is None: state_dict = blob.get("model", None) if isinstance(blob, dict) else blob
    if not isinstance(state_dict, dict): raise ValueError("Weights payload is not a valid state_dict container.")
    state_dict = _strip_module_prefix(state_dict)

    errs = _audit_state_vs_vocab(state_dict, strict_shapes)
    if errs and strict_shapes: raise RuntimeError("Strict shape audit failed.")
    if errs: report["audits"]=errs

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    cov = (sum(model.state_dict()[k].numel() for k in (set(state_dict.keys())-set(unexpected)) if k in model.state_dict())
           / max(1, sum(p.numel() for p in model.state_dict().values())))
    aligned = _assure_alignment_counts(model, TYPES, MAKES, MODELS, PARTS)
    heads_present = any(k.startswith("type_head.fc.") for k in state_dict.keys())
    heads_loaded  = all(k in (set(state_dict.keys())-set(unexpected)) for k in
                        {"type_head.fc.weight","type_head.fc.bias"} & set(model.state_dict().keys()))
    trained = bool(heads_present and heads_loaded and cov>0.50)
    report.update({"aligned":bool(aligned),"trained":trained,"load_report":{"missing":missing,"unexpected":unexpected}})
    LOG.info("bundle_load_summary", aligned=bool(aligned), trained=trained, coverage=f"{cov*100:.1f}%")
    return model, report

def load_bundle_zip(zip_path:str, cfg:Optional[dict]=None, backbone_name:str="mobilevitv2_200"):
    # Zip variant preserved (same semantics as dir)
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Bundle zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path,"r") as zf:
        members=[n for n in zf.namelist() if not n.endswith("/")]
        # find label_maps.json and best weights inside zip
        label_entry=None; weight_entry=None; weight_suffixes=_glob_suffixes(str((cfg or {}).get("bundle_weights_glob","*.pt|*.pth|*.ckpt")))
        cands=[]
        for n in members:
            base=os.path.basename(n).lower()
            if base=="label_maps.json": label_entry=n
            if any(base.endswith(sfx) for sfx in weight_suffixes): cands.append(n)
        if cands:
            cands.sort(key=lambda n: (not ("best" in n.lower() or "ema" in n.lower()), -zf.getinfo(n).file_size))
            weight_entry=cands[0]
        if label_entry is None or weight_entry is None:
            raise FileNotFoundError("Zip must contain label_maps.json and a *.pt|*.pth|*.ckpt file.")
        with zf.open(label_entry) as fh: maps=json.load(fh)
        adopt_label_maps_into_globals(maps, per_type_parts=None)
        buf=zf.read(weight_entry)
    # then same as dir path from here:
    report={"sha256":_sha256_bytes(buf),"weights_name":os.path.basename(weight_entry),"aligned":False,"trained":False,"audits":[]}
    model = build_model(backbone_name=backbone_name)
    blob = torch.load(io.BytesIO(buf), map_location="cpu")
    sd = blob.get("state_dict", None) if isinstance(blob, dict) else None
    if sd is None: sd = blob.get("model", None) if isinstance(blob, dict) else blob
    if not isinstance(sd, dict): raise ValueError("Weights payload is not a valid state_dict container.")
    sd = _strip_module_prefix(sd)
    errs=_audit_state_vs_vocab(sd, bool((cfg or {}).get("strict_state_shapes", True)))
    if errs and bool((cfg or {}).get("strict_state_shapes", True)): raise RuntimeError("Strict shape audit failed.")
    if errs: report["audits"]=errs
    missing, unexpected = model.load_state_dict(sd, strict=False)
    cov = (sum(model.state_dict()[k].numel() for k in (set(sd.keys())-set(unexpected)) if k in model.state_dict())
           / max(1, sum(p.numel() for p in model.state_dict().values())))
    aligned=_assure_alignment_counts(model, TYPES, MAKES, MODELS, PARTS)
    heads_present = any(k.startswith("type_head.fc.") for k in sd.keys())
    heads_loaded  = all(k in (set(sd.keys())-set(unexpected)) for k in
                        {"type_head.fc.weight","type_head.fc.bias"} & set(model.state_dict().keys()))
    trained = bool(heads_present and heads_loaded and cov>0.50)
    report.update({"aligned":bool(aligned),"trained":trained,"load_report":{"missing":missing,"unexpected":unexpected}})
    LOG.info("bundle_load_summary", aligned=bool(aligned), trained=trained, coverage=f"{cov*100:.1f}%")
    return model, report

# ==============================================================================
# Public API
# ==============================================================================

def _backbone_name_for_variant(variant:str)->str:
    if variant=="baseline": return os.getenv("BACKBONE_NAME_BASELINE","mobilevitv2_200")
    if variant=="cmt":      return os.getenv("BACKBONE_NAME_CMT","mobilevitv2_200")
    return os.getenv("BACKBONE_NAME_DEFAULT","mobilevitv2_200")

def load_model(variant:str)->Tuple[nn.Module, Dict[str,Any]]:
    # Decide 2-file directory path first; zip fallback allowed.
    bundle_dir = os.getenv("BASELINE_BUNDLE_PATH") if variant=="baseline" else os.getenv("CMT_BUNDLE_PATH")
    bundle_zip = None  # optional: allow ZIP via *_BUNDLE_ZIP later
    bb_name = _backbone_name_for_variant(variant)
    cfg = {"bundle_weights_glob":"*.pt|*.pth|*.ckpt","strict_state_shapes":True,"auto_remap_parts":True}

    if bundle_dir and os.path.isdir(bundle_dir):
        try:
            model, report = load_bundle_dir(bundle_dir, cfg, backbone_name=bb_name)
            LOG.info("model_loaded", variant=variant, from_dir=bundle_dir, aligned=report.get("aligned"), trained=report.get("trained"))
            return model, report
        except Exception as e:
            LOG.warn("bundle_dir_load_failed", variant=variant, dir=bundle_dir, error=str(e))

    if bundle_zip and os.path.isfile(bundle_zip):
        try:
            model, report = load_bundle_zip(bundle_zip, cfg, backbone_name=bb_name)
            LOG.info("model_loaded", variant=variant, from_zip=bundle_zip, aligned=report.get("aligned"), trained=report.get("trained"))
            return model, report
        except Exception as e:
            LOG.warn("bundle_zip_load_failed", variant=variant, zip=bundle_zip, error=str(e))

    # Fallback: SSOT build (untrained heads)
    LOG.warn("bundle_missing", variant=variant, note="Building base model from SSOT (no checkpoint).")
    model = build_model(backbone_name=bb_name)
    report = {"aligned": _assure_alignment_counts(model, TYPES, MAKES, MODELS, PARTS),
              "trained": False, "load_notes":["no_bundle_found"]}
    return model, report

# Keep Phase 1 self-test runner for easy smoke
def run_inference(image_bytes: bytes, variant: str, enable_color: bool=False, enable_plate: bool=False) -> Dict[str, Any]:
    # For now: we only demonstrate model build + heads; actual vision pipeline comes later.
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    model, rep = load_model(variant)
    _ = (img.width, img.height)  # placeholder
    return {"type":"CAR","make":"TOYOTA","model":"FORTUNER","veh_box":(10,10,200,120),
            "boxes":[(10,10,200,120)], "_timing":{"total_ms":1,"per_stage":[]}, "_loader_report":rep}

if __name__ == "__main__":
    # Quick SSOT smoke (no bundles)
    LOG.info("selftest_begin")
    m, rep = load_model(os.getenv("VARIANT","baseline"))
    LOG.info("selftest_model", head_sig=_model_vocab_signature(m), aligned=rep.get("aligned"), trained=rep.get("trained"))
    LOG.info("selftest_ok")
