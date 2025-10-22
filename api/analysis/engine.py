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

# ==============================================================================
# PHASE 5 — Inference runner + Colab-style stage timings (MS-computers feel)
# ==============================================================================
import math, base64, random
import numpy as _np
from time import perf_counter
from contextlib import nullcontext
from io import BytesIO

# ---------- ENV knobs (safe defaults) ----------
def _env_bool(name:str, default:int=0)->bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1","true","yes","on")
def _env_int(name:str, default:int)->int:
    try: return int(os.getenv(name, str(default)))
    except Exception: return int(default)
def _env_float(name:str, default:float)->float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return float(default)

IMG_SIZE     = _env_int("IMG_SIZE", 640)
AMP          = _env_bool("AMP", 1)  # autocast on CUDA
WARMUP_STEPS = _env_int("WARMUP_STEPS", 2)

# Cascade thresholds / temps (can be tuned later or set via ENV)
TAU_TYPE  = _env_float("TAU_TYPE", 0.70)
TAU_MAKE  = _env_float("TAU_MAKE", 0.70)
TAU_MODEL = _env_float("TAU_MODEL", 0.70)
TEMP_TYPE  = _env_float("TEMP_TYPE", 1.00)
TEMP_MAKE  = _env_float("TEMP_MAKE", 1.00)
TEMP_MODEL = _env_float("TEMP_MODEL", 1.00)

PART_TAU   = _env_float("PART_TAU", 0.30)

ENABLE_COLOR = _env_bool("ENABLE_COLOR", 0)
ENABLE_PLATE = _env_bool("ENABLE_PLATE", 0)

# ---------- StageTimer ----------
class StageTimer:
    ORDER = ["setup","vocab","bundle","preproc","forward","decode","postproc","color","plate","s3","db"]
    def __init__(self, analysis_id:str, variant:str, logger:JsonLogger=LOG):
        self.analysis_id = analysis_id
        self.variant     = variant
        self.ms          = {}
        self._logger     = logger
        self._t0         = perf_counter()
        self._last       = self._t0
    def tick(self, name:str):
        now = perf_counter()
        self.ms[name] = int(round(1000.0*(now - self._last)))
        self._last = now
    def done(self)->dict:
        total = int(round(1000.0*(perf_counter() - self._t0)))
        out = {k: int(self.ms.get(k, 0)) for k in self.ORDER}
        out["total"] = total
        self._logger.info("stages", analysis_id=self.analysis_id, variant=self.variant, ms=out)
        return out

# ---------- Letterbox + inverse mapping ----------
def _letterbox_fit_chw(x: torch.Tensor, size:int):
    # x: CHW float in [0,1] or uint8 0..255
    assert x.dim()==3 and x.shape[0] in (1,3), "expect CHW"
    B = x.unsqueeze(0)
    _, C, H, W = B.shape
    s = float(size); sc = min(s/max(1.0,W), s/max(1.0,H))
    new_w, new_h = int(round(W*sc)), int(round(H*sc))
    mode = "bilinear" if C>1 else "nearest"
    Bf = B.float()
    Bi = F.interpolate(Bf, size=(new_h,new_w), mode=mode, align_corners=False if mode=="bilinear" else None)
    pw_tot, ph_tot = max(0, size-new_w), max(0, size-new_h)
    left, right, top, bottom = pw_tot//2, pw_tot-pw_tot//2, ph_tot//2, ph_tot-ph_tot//2
    padB = F.pad(Bi, (left,right,top,bottom), value=0.0)
    return padB.squeeze(0), (left, top), sc

def _inv_letterbox_xyxy(box, orig_w:int, orig_h:int, pad:tuple, scale:float):
    if box is None: return None
    x1,y1,x2,y2 = [float(v) for v in (box.tolist() if torch.is_tensor(box) else box)]
    pw, ph = float(pad[0]), float(pad[1]); s = max(1e-12, float(scale))
    xo1, yo1 = (x1-pw)/s, (y1-ph)/s
    xo2, yo2 = (x2-pw)/s, (y2-ph)/s
    xo1 = max(0.0, min(xo1, orig_w-1.0)); yo1 = max(0.0, min(yo1, orig_h-1.0))
    xo2 = max(0.0, min(xo2, orig_w-1.0)); yo2 = max(0.0, min(yo2, orig_h-1.0))
    return (xo1,yo1,xo2,yo2)

# ---------- BBox helpers ----------
def _unpack_bbox_to_BP4HW(raw: torch.Tensor, P_expected:int, H:int, W:int):
    if raw is None or not torch.is_tensor(raw): return None
    if raw.dim()==4:
        B,C,Hh,Ww = raw.shape
        if Hh!=H or Ww!=W or (C%4)!=0: return None
        P=min(P_expected, C//4)
        return raw[:,:4*P].view(B,P,4,H,W)
    if raw.dim()==5 and raw.shape[2]==4:
        B,P_in,_,Hh,Ww = raw.shape
        if Hh!=H or Ww!=W: return None
        return raw[:,:min(P_expected,P_in)]
    if raw.dim()==5 and raw.shape[1]==4:
        B,_,P_in,Hh,Ww = raw.shape
        if Hh!=H or Ww!=W: return None
        P=min(P_expected,P_in)
        return raw[:,: ,:P].permute(0,2,1,3,4)
    return None

def _decode_dxdy_logwh_cell(BP4: torch.Tensor, H:int, W:int, img_size:int):
    if BP4 is None: return None
    B,P,_4,Hh,Ww = BP4.shape
    if Hh!=H or Ww!=W or _4!=4: return None
    cw, ch = img_size/float(W), img_size/float(H)
    i = torch.arange(W, device=BP4.device, dtype=torch.float32).view(1,1,1,W)
    j = torch.arange(H, device=BP4.device, dtype=torch.float32).view(1,1,H,1)
    cx0, cy0 = (i+0.5)*cw, (j+0.5)*ch
    dx, dy = BP4[:,:,0], BP4[:,:,1]
    lw, lh = BP4[:,:,2].clamp(-8,8), BP4[:,:,3].clamp(-8,8)
    cx, cy = cx0 + dx*cw, cy0 + dy*ch
    w,  h  = torch.exp(lw)*cw, torch.exp(lh)*ch
    x1, y1 = (cx-0.5*w).clamp(0, img_size-1), (cy-0.5*h).clamp(0, img_size-1)
    x2, y2 = (cx+0.5*w).clamp(0, img_size-1), (cy+0.5*h).clamp(0, img_size-1)
    return torch.stack([x1,y1,x2,y2], dim=2)  # [B,P,4,H,W]

def _veh_box_from_any(veh_pred: torch.Tensor, img_size:int):
    if veh_pred is None or not torch.is_tensor(veh_pred): return None
    if veh_pred.dim()==2 and veh_pred.shape[1]==4:
        sig = veh_pred.sigmoid().float()
        cx,cy,w,h = sig[:,0]*img_size, sig[:,1]*img_size, sig[:,2]*img_size, sig[:,3]*img_size
        x1,y1 = (cx-0.5*w).clamp(0,img_size-1), (cy-0.5*h).clamp(0,img_size-1)
        x2,y2 = (cx+0.5*w).clamp(0,img_size-1), (cy+0.5*h).clamp(0,img_size-1)
        return torch.stack([x1,y1,x2,y2], dim=1)
    return None

def _derive_vehicle_box_from_parts(part_logits: torch.Tensor, BP4: torch.Tensor, img_size:int):
    if part_logits is None or BP4 is None: return None
    # pick best cell per part from logits, then take weighted quantiles of boxes
    B,P,H,W = part_logits.shape
    probs = torch.sigmoid(part_logits).reshape(B,P,-1)
    idx = probs.argmax(dim=-1)  # [B,P]
    ys, xs = (idx // W), (idx % W)
    boxes_sq = []
    scores   = []
    decoded = _decode_dxdy_logwh_cell(BP4, H, W, img_size)  # [B,P,4,H,W]
    if decoded is None: return None
    for p in range(P):
        y, x = int(ys[0,p]), int(xs[0,p])
        b = decoded[0,p,:,y,x]  # [4]
        boxes_sq.append([float(b[0]),float(b[1]),float(b[2]),float(b[3])])
        scores.append(float(probs[0,p, y*W+x]))
    if not boxes_sq: return None
    # weighted middle 80%
    xs1 = torch.tensor([b[0] for b in boxes_sq]); ys1 = torch.tensor([b[1] for b in boxes_sq])
    xs2 = torch.tensor([b[2] for b in boxes_sq]); ys2 = torch.tensor([b[3] for b in boxes_sq])
    wts = torch.tensor(scores).clamp_min(1e-6); cw = wts.cumsum(0)/wts.sum()

    def _wq(vals, q):
        s, idx = torch.sort(vals); ww = wts[idx]; c = ww.cumsum(0)/ww.sum()
        j = int(torch.searchsorted(c, torch.tensor([q]), right=True).clamp(max=len(s)-1))
        return float(s[j])
    x1 = _wq(xs1, 0.10); y1 = _wq(ys1, 0.10)
    x2 = _wq(xs2, 0.90); y2 = _wq(ys2, 0.90)

    # ensure min size
    minw = 0.10*img_size; minh = 0.10*img_size
    if (x2-x1)<minw:
        cx = 0.5*(x1+x2); x1, x2 = cx-0.5*minw, cx+0.5*minw
    if (y2-y1)<minh:
        cy = 0.5*(y1+y2); y1, y2 = cy-0.5*minh, cy+0.5*minh
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_size-1, x2); y2 = min(img_size-1, y2)
    return (x1,y1,x2,y2)

# ---------- Cascade (Type → Make → Model) ----------
def _cascade_type_make_model(lt:torch.Tensor|None, lm:torch.Tensor|None, lk:torch.Tensor|None):
    if lt is None: return {"type":None,"make":None,"model":None,"confs":(),"stop":"no-type"}
    pt = torch.softmax(lt.float()/max(1e-6,TEMP_TYPE), dim=-1)
    t  = int(pt.argmax(dim=-1)[0]); ct = float(pt[0,t])
    if lm is None or ct < TAU_TYPE:
        return {"type":t,"make":None,"model":None,"confs":(ct,), "stop":"type" if ct<TAU_TYPE else "nocascade"}
    # Mask makes by allowed types
    allowed_m = (allowed_makes_by_type_idx or {}).get(t, None)
    if allowed_m is not None:
        mask = torch.full_like(lm, -1e9); mask[:, allowed_m] = 0.0
        pm = torch.softmax((lm.float()+mask)/max(1e-6,TEMP_MAKE), dim=-1)
    else:
        pm = torch.softmax(lm.float()/max(1e-6,TEMP_MAKE), dim=-1)
    m  = int(pm.argmax(dim=-1)[0]); cm = float(pm[0,m])
    if lk is None or cm < TAU_MAKE:
        return {"type":t,"make":None,"model":None,"confs":(ct,cm),"stop":"make"}
    allowed_k = (allowed_models_by_make_idx or {}).get(m, [])
    if allowed_k:
        maskk = torch.full_like(lk, -1e9); maskk[:, allowed_k] = 0.0
        pk = torch.softmax((lk.float()+maskk)/max(1e-6,TEMP_MODEL), dim=-1)
    else:
        pk = torch.softmax(lk.float()/max(1e-6,TEMP_MODEL), dim=-1)
    k  = int(pk.argmax(dim=-1)[0]); ck = float(pk[0,k])
    if ck < TAU_MODEL:
        return {"type":t,"make":m,"model":None,"confs":(ct,cm,ck),"stop":"model"}
    return {"type":t,"make":m,"model":k,"confs":(ct,cm,ck),"stop":"ok"}

# ---------- Model cache + warm-up ----------
_MODEL_CACHE: dict[str, Tuple[nn.Module, dict]] = {}
_WARMED: set[str] = set()
def _get_model_for_variant(variant:str)->Tuple[nn.Module, dict]:
    key = variant.strip().lower()
    if key in _MODEL_CACHE: return _MODEL_CACHE[key]
    try:
        m, rep = load_model(key)  # existing helper from earlier phases
    except Exception as e:
        LOG.warn("load_model_error", variant=key, error=str(e))
        m, rep = build_model(backbone_name=os.getenv("BACKBONE_NAME_BASELINE","mobilevitv2_200")), {"aligned":True,"trained":False,"load_notes":["fallback_build"]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev).eval()
    _MODEL_CACHE[key] = (m, rep or {})
    return _MODEL_CACHE[key]

def _maybe_warmup(model: nn.Module, device:str, img_size:int, steps:int=WARMUP_STEPS):
    key = f"{id(model)}:{device}:{img_size}"
    if key in _WARMED or steps<=0: return
    x = torch.zeros(1,3,img_size,img_size, device=device)
    use_amp = bool(AMP and device.startswith("cuda") and torch.cuda.is_available())
    ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
    with torch.no_grad():
        with ctx:
            for _ in range(int(steps)):
                _ = model(x)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
    _WARMED.add(key)
    LOG.info("warmup_done", steps=int(steps), device=device)

# ---------- Inference runner ----------
def run_inference(img_bytes: bytes, variant: str="baseline", analysis_id: Optional[str]=None):
    """
    Returns: (dets, timings, metrics)
      dets: {
        "type": name|None, "type_conf": float|0,
        "make": name|None, "make_conf": float|0,
        "model": name|None, "model_conf": float|0,
        "parts": [{"name":str,"conf":float}, ...],
        "colors": [{"name":str,"fraction":float,"conf":float}, ...],
        "plate_text": str, "plate_conf": float|0,
        "veh_box": (x1,y1,x2,y2) in original image coords (if available),
        "plate_box": (x1,y1,x2,y2) (if available)
      }
    """
    aid = analysis_id or f"an_{int(time.time()*1000)}_{random.randint(100,999)}"
    timer = StageTimer(aid, variant)

    # setup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    ow, oh = img.size
    timer.tick("setup")

    # vocab (touch globals to ensure they’re built/adopted)
    _ = (len(TYPES), len(MAKES), len(MODELS), len(PARTS), len(REGIONS))
    timer.tick("vocab")

    # bundle/model
    model, rep = _get_model_for_variant(variant)
    timer.tick("bundle")

    # preproc (PIL -> torch CHW float [0,1] -> letterbox -> BCHW)
    arr = _np.asarray(img)  # HWC uint8
    tCHW = torch.from_numpy(arr).permute(2,0,1)  # CHW uint8
    tCHW = tCHW.float()/255.0
    sq, pad, scale = _letterbox_fit_chw(tCHW, int(IMG_SIZE))  # CHW
    X = sq.unsqueeze(0).to(dev)  # BCHW
    timer.tick("preproc")

    # forward (with one-time warm-up)
    _maybe_warmup(model, dev, int(IMG_SIZE))
    use_amp = bool(AMP and dev.startswith("cuda") and torch.cuda.is_available())
    ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
    with torch.no_grad():
        with ctx:
            out = model(X)
            if dev.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
    timer.tick("forward")

    # decode: types/makes/models
    lt = out.get("type_logits")
    lm = out.get("make_logits")
    lk = out.get("model_logits")
    cascade = _cascade_type_make_model(lt, lm, lk)
    t_id, m_id, k_id = cascade["type"], cascade["make"], cascade["model"]
    confs = cascade["confs"] + ((0.0,)*(3-len(cascade["confs"])))
    type_name  = (TYPES[t_id]  if t_id is not None and 0<=t_id<len(TYPES)  else None)
    make_name  = (MAKES[m_id]  if m_id is not None and 0<=m_id<len(MAKES)  else None)
    model_name = (MODELS[k_id] if k_id is not None and 0<=k_id<len(MODELS) else None)

    # parts
    P = len(PARTS)
    part_logits = out.get("part_logits")  # [B,P,H,W] or None
    bbox_preds  = out.get("bbox_preds")   # [B,4P,H,W] or compatible
    parts_list  = []
    veh_box_sq  = None

    if torch.is_tensor(part_logits):
        B,Pp,H,W = part_logits.shape
        probs = torch.sigmoid(part_logits)[0]  # [P,H,W]
        scores = probs.amax(dim=(-1,-2))       # [P]
        # pick parts above PART_TAU (sane default)
        for p_idx, sc in enumerate(scores.tolist()):
            if float(sc) >= float(PART_TAU):
                parts_list.append({"name": PARTS[p_idx] if p_idx < len(PARTS) else f"part_{p_idx}",
                                   "conf": float(sc)})
        # veh box: prefer explicit veh head, else derive from parts+bbox
        veh_box_pred = out.get("veh_box_preds")
        if veh_box_pred is not None:
            v = _veh_box_from_any(veh_box_pred, int(IMG_SIZE))
            if v is not None: veh_box_sq = tuple(float(v[0,i].item()) for i in range(4))
        if veh_box_sq is None and torch.is_tensor(bbox_preds):
            BP4 = _unpack_bbox_to_BP4HW(bbox_preds, P, H, W)
            veh_box_sq = _derive_vehicle_box_from_parts(part_logits[0], BP4, int(IMG_SIZE))
    else:
        # still try veh head
        veh_box_pred = out.get("veh_box_preds")
        if veh_box_pred is not None:
            v = _veh_box_from_any(veh_box_pred, int(IMG_SIZE))
            if v is not None: veh_box_sq = tuple(float(v[0,i].item()) for i in range(4))

    # back to original coords
    veh_box = _inv_letterbox_xyxy(veh_box_sq, ow, oh, pad, scale) if veh_box_sq is not None else None

    timer.tick("decode")

    # postproc: assemble dets
    dets = {
        "type": type_name,   "type_conf":  float(confs[0]) if len(confs)>0 else 0.0,
        "make": make_name,   "make_conf":  float(confs[1]) if len(confs)>1 else 0.0,
        "model": model_name, "model_conf": float(confs[2]) if len(confs)>2 else 0.0,
        "parts": parts_list,
        "colors": [],
        "plate_text": "",
        "plate_conf": 0.0,
        "veh_box": veh_box,
        "plate_box": None,
    }
    timer.tick("postproc")

    # color (optional; warn once if disabled/missing)
    global _COLOR_WARNED; 
    try: _COLOR_WARNED
    except NameError: _COLOR_WARNED = False

    if ENABLE_COLOR:
        try:
            from analysis import utils as _U
            if hasattr(_U, "detect_vehicle_color") and veh_box is not None:
                # crop and call
                crop = _U.crop_by_xyxy(img, veh_box) if hasattr(_U, "crop_by_xyxy") else img.crop(tuple(map(int,veh_box)))
                colors = _U.detect_vehicle_color(crop)
                # Expected: list of dicts {name,fraction,conf} or similar (be tolerant)
                if isinstance(colors, (list,tuple)):
                    out_colors=[]
                    for c in colors[:3]:
                        if isinstance(c, dict) and "name" in c:
                            out_colors.append({"name":str(c["name"]),
                                               "fraction": float(c.get("fraction", 0.0)),
                                               "conf": float(c.get("conf", c.get("confidence", 0.0)))})
                        elif isinstance(c, (list,tuple)) and len(c)>=1:
                            out_colors.append({"name":str(c[0]),
                                               "fraction": float(c[1]) if len(c)>1 else 0.0,
                                               "conf": float(c[2]) if len(c)>2 else 0.0})
                    dets["colors"] = out_colors
            else:
                if not _COLOR_WARNED:
                    LOG.warn("color_missing_impl", note="ENABLE_COLOR=1 but utils.detect_vehicle_color not found or no veh_box")
                    _COLOR_WARNED = True
        except Exception as e:
            if not _COLOR_WARNED:
                LOG.warn("color_error", error=str(e)); _COLOR_WARNED=True
    else:
        if not _COLOR_WARNED:
            LOG.warn("color_disabled", note="ENABLE_COLOR=0 → color stage skipped")
            _COLOR_WARNED = True
    timer.tick("color")

    # plate (optional; warn once if disabled/missing)
    global _PLATE_WARNED;
    try: _PLATE_WARNED
    except NameError: _PLATE_WARNED=False

    if ENABLE_PLATE and not dets.get("plate_text"):
        try:
            from analysis import utils as _U
            if hasattr(_U, "read_plate_text") and veh_box is not None:
                crop = _U.crop_by_xyxy(img, veh_box) if hasattr(_U, "crop_by_xyxy") else img.crop(tuple(map(int,veh_box)))
                resp = _U.read_plate_text(crop)
                if isinstance(resp, dict):
                    dets["plate_text"] = resp.get("text","") or resp.get("plate","") or ""
                    dets["plate_conf"] = float(resp.get("conf", resp.get("confidence", 0.0)))
                elif isinstance(resp, (list,tuple)) and resp:
                    dets["plate_text"] = str(resp[0]); dets["plate_conf"] = float(resp[1]) if len(resp)>1 else 0.0
                elif isinstance(resp, str):
                    dets["plate_text"] = resp
            else:
                if not _PLATE_WARNED:
                    LOG.warn("plate_missing_impl", note="ENABLE_PLATE=1 but utils.read_plate_text not found or no veh_box")
                    _PLATE_WARNED = True
        except Exception as e:
            if not _PLATE_WARNED:
                LOG.warn("plate_error", error=str(e)); _PLATE_WARNED=True
    else:
        if not _PLATE_WARNED:
            LOG.warn("plate_disabled", note="ENABLE_PLATE=0 → plate stage skipped")
            _PLATE_WARNED = True
    timer.tick("plate")

    # (placeholders for S3 / DB; leave zero if unused)
    timer.tick("s3")
    timer.tick("db")

    timings = timer.done()

    # metrics (align E2E with StageTimer total)
    mem_gb = None
    if dev.startswith("cuda") and torch.cuda.is_available():
        try: mem_gb = torch.cuda.memory_allocated()/1e9
        except Exception: mem_gb = None
    metrics = {
        "latency_ms": float(timings["total"]),
        "gflops": None,  # can be filled later if we wire thop here
        "mem_gb": float(mem_gb) if mem_gb is not None else None,
        "device": dev,
        "trained": bool(rep.get("trained", False)),
    }
    return dets, timings, metrics

