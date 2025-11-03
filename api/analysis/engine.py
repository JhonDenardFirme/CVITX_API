
# ======================= START OF EC2 CODE ENGINE (REWRITTEN — Phase-A Stable, Phase-9 Keys, SSOT-Aligned + Alias Hooks) =======================
# Rewriter notes:
# - Full replacement retaining original structure & public API (surgical changes only).
# - Change labels: [SSOT] completed vocabs/maps, [ALIAS] alias stubs + reapplication, [LIGHT] TailLight→Taillight fixes,
#                  [REGION] CFG.regions support, [KEEP] preserved behavior.

import os, io, json, time, hashlib, zipfile
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any, Sequence
from io import BytesIO

# Light deps
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image, ImageDraw, ImageOps

# === Shared geometry/decoding utils (alias as s_*) ===
from .bbox_utils import (
    letterbox_fit            as s_letterbox_fit,
    invert_letterbox_box     as s_invert_letterbox_box,
    _unpack_bbox_to_BP4HW    as s_unpack_BP4HW,
    _decode_dxdy_logwh_cell  as s_decode_dxdy,
    _veh_box_from_any        as s_veh_box_from_any,
    _box_from_heatmap        as s_box_from_heatmap,
    _ensure_min_box          as s_ensure_min_box,     # not used but kept for parity
    blend_logits             as s_blend_logits,
    cascade_infer            as s_cascade_infer,
    assert_head_grid_matches as s_assert_head_grid_matches,
)

# --- Shared utils (runtime, color, plate) -------------------------------------
try:
    from api.analysis.utils_runtime import warmup_model, estimate_gflops, per_infer_memory_metrics
except Exception:
    warmup_model = estimate_gflops = per_infer_memory_metrics = None

try:
    from api.analysis import utils_color as _COLOR
except Exception:
    _COLOR = None

try:
    from api.analysis import utils_plate as _PLATE
except Exception:
    _PLATE = None

# ---- Local logger (kept)
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

# --- Runtime metrics helpers (GFLOPs + NVML) --- (kept as fallbacks)
_GFLOPS_CACHE: dict[tuple, float] = {}
_NVML_TOTAL_GB: float | None = None

def _nvml_totals_gb() -> float | None:
    global _NVML_TOTAL_GB
    if _NVML_TOTAL_GB is not None:
        return _NVML_TOTAL_GB
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        _NVML_TOTAL_GB = float(info.total) / 1e9
    except Exception:
        _NVML_TOTAL_GB = None
    return _NVML_TOTAL_GB

def _estimate_gflops(model: nn.Module, img_size: int) -> float | None:
    key = (id(model), int(img_size))
    if key in _GFLOPS_CACHE:
        return _GFLOPS_CACHE[key]
    try:
        from thop import profile
        device = next(model.parameters()).device
        x = torch.zeros(1, 3, int(img_size), int(img_size), device=device)
        with torch.no_grad():
            macs, _ = profile(model, inputs=(x,), verbose=False)
        gf = float(macs) / 1e9
        _GFLOPS_CACHE[key] = gf
        return gf
    except Exception:
        _GFLOPS_CACHE[key] = None
        return None


# ==============================================================================
# PHASE 2 — SSOT + label_maps adoption (+ allow-lists, regions/parts) [SSOT][ALIAS][LIGHT][REGION]
# ==============================================================================

# ------------------------ [ALIAS] alias helpers (Colab-compatible stubs) ------------------------
# We will call apply_all_label_aliases(type2idx, make2idx, model2idx, part2idx, cfg),
# if provided elsewhere. Otherwise we supply a no-op stub and sanitized alias dicts on CFG.
if "apply_all_label_aliases" not in globals():
    def apply_all_label_aliases(type2idx=None, make2idx=None, model2idx=None, part2idx=None, cfg=None, verbose=False):
        # no-op if user didn't define real aliasing; keep maps unchanged
        return type2idx, make2idx, model2idx, part2idx

def _sanitize_alias_dict(d):
    return {k: v for k, v in (d or {}).items() if isinstance(v, str) and v.strip()}

# Ensure CFG exists and carries alias fields used by the Colab reference.
if "CFG" not in globals():
    class _CFGStub:
        regions: Tuple[str, ...] = tuple(["Front", "Side", "Rear", "Roof"])
        alias_type:  Dict[str, str] = {}
        alias_make:  Dict[str, str] = {}
        alias_model: Dict[str, str] = {}
        alias_part:  Dict[str, str] = {}
    CFG = _CFGStub()  # type: ignore

# sanitize any pre-provided alias dicts
CFG.alias_type  = _sanitize_alias_dict(getattr(CFG, "alias_type",  {}))
CFG.alias_make  = _sanitize_alias_dict(getattr(CFG, "alias_make",  {}))
CFG.alias_model = _sanitize_alias_dict(getattr(CFG, "alias_model", {}))
CFG.alias_part  = _sanitize_alias_dict(getattr(CFG, "alias_part",  {}))

# ------------------------ SSOT: Vehicle Hierarchy (expanded, unchanged content) -----------------
# Rules mirrored from Colab:
# • Preserve dashes in TYPE names.
# • Conjoin multi-word models (e.g., "Ranger Wildtrak"→"RangerWildtrak").
# • Canonicalize TailLight→Taillight.
vehicle_hierarchy: Dict[str, Dict[str, List[str]]] = {
    "Car": {
        "Toyota":      ["Wigo", "Vios", "CorollaAltis", "Camry", "GRYaris"],
        "Mitsubishi":  ["Mirage", "MirageG4"],
        "Ford":        ["Mustang"],
        "Nissan":      ["Almera", "GT-R"],
        "Honda":       ["Brio", "City", "Civic", "CivicTypeR"],
        "Suzuki":      ["Dzire"],
        "Hyundai":     ["Elantra"],
        "Geely":       ["Emgrand"],
        "Chevrolet":   ["Camaro"]
    },
    "SUV": {
        "Toyota":      ["Raize", "YarisCross", "CorollaCross", "Rush", "Fortuner",
                        "LandCruiserPrado", "LandCruiserLC300", "Avanza", "Veloz", "Innova"],
        "Mitsubishi":  ["Xpander", "MonteroSport"],
        "Ford":        ["Everest", "Territory", "Explorer"],
        "Nissan":      ["Terra", "Patrol"],
        "Honda":       ["BR-V", "CR-V", "HR-V"],
        "Suzuki":      ["Jimny", "Ertiga"],
        "Hyundai":     ["Stargazer", "Tucson", "SantaFe"],
        "Isuzu":       ["Mu-X"],
        "Geely":       ["Coolray", "GX3Pro", "Okavango"],
        "Chevrolet":   ["Trailblazer", "Suburban"]
    },
    "Van": {
        "Toyota":      ["Alphard", "Coaster", "HiAce"],
        "Hyundai":     ["Staria"],
        "Nissan":      ["Urvan"]
    },
    "Pickup": {
        "Toyota":      ["Hilux", "HiluxTamaraw"],
        "Mitsubishi":  ["Strada", "Triton"],
        "Ford":        ["RangerWildtrak", "RangerRaptor"],
        "Nissan":      ["Navara"],
        "Isuzu":       ["D-Max"]
    },
    "Utility": {
        "Toyota":      ["LiteAce"],
        "Mitsubishi":  ["L300"],
        "Hyundai":     ["H-100"],
        "Isuzu":       ["Traviz"]
    },
    "Motorcycle": {
        "HondaMC":    ["BeAT", "Click", "PCX", "WaveRSX", "TMX", "XRM"],
        "YamahaMC":   ["Mio", "NMAX", "Aerox"],
        "SuzukiMC":   ["Raider", "Smash", "BurgmanStreet"]
    },
    "Bicycle": {},
    "E-Bike": {},
    "Pedicab": {},
    "Tricycle": {},
    "Jeepney": {},
    "E-Jeepney": {},
    "Bus": {},
    "CarouselBus": {},
    "LightTruck": {},
    "ContainerTruck": {},
    "SpecialVehicle": {}
}

# ------------------------ SSOT: Part Classes by Type (expanded, unchanged content) --------------
part_classes_by_type: Dict[str, List[Tuple[str, str]]] = {
    "Car": [
        ("Bumper","Front"), ("Grille","Front"), ("Hood","Front"),
        ("LeftHeadlight","Front"), ("Logo","Front"), ("Plate","Front"),
        ("RightHeadlight","Front"), ("Windshield","Front"),
        ("Roof","Roof"),
        ("LeftTaillight","Rear"), ("Logo","Rear"), ("Plate","Rear"),
        ("Rear","Rear"), ("RightTaillight","Rear"), ("Trunk","Rear"),
        ("Window","Rear"), ("ModelText","Rear"),
        ("FrontDoor","Side"), ("FrontWheel","Side"),
        ("RearDoor","Side"), ("RearWheel","Side"),
    ],
    "SUV": [
        ("Bumper","Front"), ("Grille","Front"), ("Hood","Front"),
        ("LeftHeadlight","Front"), ("Logo","Front"), ("Plate","Front"),
        ("RightHeadlight","Front"), ("Windshield","Front"),
        ("Roof","Roof"),
        ("LeftTaillight","Rear"), ("Logo","Rear"), ("Plate","Rear"),
        ("Rear","Rear"), ("RightTaillight","Rear"), ("Trunk","Rear"),
        ("Window","Rear"), ("ModelText","Rear"),
        ("FrontDoor","Side"), ("FrontWheel","Side"),
        ("RearDoor","Side"), ("RearWheel","Side"), ("RearSection","Side"),
    ],
    "Pickup": [
        ("Bumper","Front"), ("Grille","Front"), ("Hood","Front"),
        ("LeftHeadlight","Front"), ("Logo","Front"), ("Plate","Front"),
        ("RightHeadlight","Front"), ("Windshield","Front"),
        ("Roof","Roof"),
        ("LeftTaillight","Rear"), ("Logo","Rear"), ("Plate","Rear"),
        ("Rear","Rear"), ("RightTaillight","Rear"), ("CargoRear","Rear"),
        ("Window","Rear"), ("ModelText","Rear"),
        ("CargoPanel","Side"), ("RearDoor","Side"), ("FrontDoor","Side"),
        ("RearWheel","Side"), ("FrontWheel","Side"),
    ],
    "Van": [
        ("Bumper","Front"), ("Grille","Front"),
        ("LeftHeadlight","Front"), ("Logo","Front"), ("Plate","Front"),
        ("RightHeadlight","Front"), ("Windshield","Front"),
        ("Roof","Roof"),
        ("LeftTaillight","Rear"), ("Logo","Rear"), ("Plate","Rear"),
        ("Rear","Rear"), ("RightTaillight","Rear"), ("RearDoor","Rear"),
        ("Window","Rear"), ("ModelText","Rear"),
        ("SlidingDoor","Side"), ("FrontDoor","Side"),
        ("RearWheel","Side"), ("FrontWheel","Side"), ("RearSection","Side"),
    ],
    "Utility": [
        ("Bumper","Front"), ("Grille","Front"),
        ("LeftHeadlight","Front"), ("Logo","Front"), ("Plate","Front"),
        ("RightHeadlight","Front"), ("Windshield","Front"),
        ("CabRoof","Roof"), ("RearRoof","Roof"),
        ("LeftTaillight","Rear"), ("Logo","Rear"), ("Plate","Rear"),
        ("Rear","Rear"), ("RightTaillight","Rear"), ("RearDoor","Rear"),
        ("Window","Rear"), ("ModelText","Rear"),
        ("FrontDoor","Side"), ("CabPanel","Side"),
        ("RearWheel","Side"), ("FrontWheel","Side"),
    ],
    "Motorcycle": [
        ("LeftHeadlight","Front"), ("MainHeadlight","Front"),
        ("RightHeadlight","Front"), ("FrontFairing","Front"), ("Handlebar","Front"),
        ("Plate","Rear"), ("Taillight","Rear"),
        ("RearWheel","Side"), ("FrontWheel","Side"),
        ("Seat","Side"), ("Body","Side"),
    ],
    "Bicycle": [
        ("FrontWheel","Front"), ("Handlebars","Front"),
        ("RearWheel","Rear"), ("RearFrame","Rear"),
        ("Chain","Side"), ("Seat","Side"), ("Frame","Side"),
    ],
    "E-Bike": [
        ("FrontWheel","Front"), ("Handlebars","Front"),
        ("RearWheel","Rear"), ("RearFrame","Rear"),
        ("Chain","Side"), ("Seat","Side"), ("Frame","Side"),
    ],
    "Pedicab": [
        ("FrontWheel","Front"), ("Handlebars","Front"), ("SidecarFront","Front"),
        ("SidecarRoof","Roof"), ("Driversideroof","Roof"),
        ("RearWheel","Rear"), ("SidecarRear","Rear"),
        ("Chain","Side"), ("Seat","Side"), ("Frame","Side"),
        ("SidecarPanel","Side"), ("SidecarWheel","Side"), ("PassengerSeat","Side"),
    ],
    "Tricycle": [
        ("Headlight","Front"), ("MotorWindshield","Front"), ("MotorHandlebar","Front"), ("SidecarFront","Front"),
        ("MotorRoof","Roof"), ("SidecarRoof","Roof"),
        ("Plate","Rear"), ("SidecarRear","Rear"), ("MotorRear","Rear"),
        ("RearWheel","Side"), ("FrontWheel","Side"), ("MotorSide","Side"), ("SidecarPanel","Side"),
    ],
    "Jeepney": [
        ("Windshield","Front"), ("Plate","Front"), ("Bumper","Front"),
        ("LeftHeadlight","Front"), ("RightHeadlight","Front"), ("Grille","Front"), ("Hood","Front"),
        ("Roof","Roof"),
        ("Plate","Rear"), ("RearDoor","Rear"), ("LeftTaillight","Rear"), ("RightTaillight","Rear"),
        ("DriverDoor","Side"), ("SidePanel","Side"), ("RearWheel","Side"), ("FrontWheel","Side"),
    ],
    "E-Jeepney": [
        ("Windshield","Front"), ("Plate","Front"), ("Bumper","Front"),
        ("LeftHeadlight","Front"), ("RightHeadlight","Front"), ("Grille","Front"),
        ("Roof","Roof"),
        ("Plate","Rear"), ("RearPanel","Rear"), ("LeftTaillight","Rear"), ("RightTaillight","Rear"),
        ("Door","Side"), ("SidePanel","Side"), ("FrontWheel","Side"), ("RearWheel","Side"),
    ],
    "CarouselBus": [
        ("Windshield","Front"), ("Plate","Front"), ("Bumper","Front"),
        ("LeftHeadlight","Front"), ("RightHeadlight","Front"), ("Grille","Front"),
        ("Roof","Roof"),
        ("Plate","Rear"), ("RearPanel","Rear"), ("LeftTaillight","Rear"), ("RightTaillight","Rear"), ("Window","Rear"),
        ("PassengerDoor","Side"), ("SidePanel","Side"), ("RearWheel","Side"), ("FrontWheel","Side"),
    ],
    "Bus": [
        ("Windshield","Front"), ("Plate","Front"), ("Bumper","Front"),
        ("LeftHeadlight","Front"), ("RightHeadlight","Front"), ("Grille","Front"),
        ("Roof","Roof"),
        ("Plate","Rear"), ("RearPanel","Rear"), ("LeftTaillight","Rear"), ("RightTaillight","Rear"), ("Window","Rear"),
        ("PassengerDoor","Side"), ("SidePanel","Side"), ("RearWheel","Side"), ("FrontWheel","Side"),
    ],
    "LightTruck": [
        ("Windshield","Front"), ("Plate","Front"), ("Bumper","Front"),
        ("LeftHeadlight","Front"), ("RightHeadlight","Front"), ("Grille","Front"), ("Logo","Front"),
        ("Roof","Roof"),
        ("Plate","Rear"), ("Flatbed","Rear"), ("LeftTaillight","Rear"), ("RightTaillight","Rear"),
        ("DriverDoor","Side"), ("CargoPanel","Side"), ("RearWheel","Side"), ("FrontWheel","Side"),
    ],
    "ContainerTruck": [
        ("Windshield","Front"), ("Plate","Front"), ("ContainerPanel","Front"),
        ("Bumper","Front"), ("LeftHeadlight","Front"), ("RightHeadlight","Front"),
        ("Grille","Front"), ("Hood","Front"), ("Logo","Front"),
        ("FrontRoof","Roof"), ("ContainerRoof","Roof"),
        ("Rear","Rear"), ("Plate","Rear"), ("ContainerDoor","Rear"),
        ("LeftTaillight","Rear"), ("RightTaillight","Rear"),
        ("FrontDoor","Side"), ("ContainerPanel","Side"), ("RearWheel","Side"), ("FrontWheel","Side"),
    ],
    "SpecialVehicle": [
        ("Windshield","Front"), ("Plate","Front"), ("Bumper","Front"),
        ("LeftHeadlight","Front"), ("RightHeadlight","Front"), ("Grille","Front"), ("Logo","Front"),
        ("Roof","Roof"),
        ("Plate","Rear"), ("RearDoor","Rear"), ("LeftTaillight","Rear"), ("RightTaillight","Rear"),
        ("SidePanel","Side"), ("RearWheel","Side"), ("FrontWheel","Side"), ("SpecialMark","Side"),
    ],
}

# Canonical TYPE list (fixed order) — matches canon (no spaces in special classes)
TYPES: List[str] = [
    "Car","SUV","Van","Pickup","Utility","Motorcycle","Bicycle",
    "E-Bike","Pedicab","Tricycle","Jeepney","E-Jeepney",
    "Bus","CarouselBus","LightTruck","ContainerTruck","SpecialVehicle"
]

# ------------------------ [LIGHT] simple canonicalizer for HeadLight/TailLight variants ----------
_LIGHT_FIX = {
    "HeadLight":"Headlight","LeftHeadLight":"LeftHeadlight","RightHeadLight":"RightHeadlight",
    "TailLight":"Taillight","LeftTailLight":"LeftTaillight","RightTailLight":"RightTaillight",
}
def _canon_part(p:str)->str: return _LIGHT_FIX.get(p,p)

def canonicalize_label_lights_simple(label: str) -> str:
    chunks = str(label or "").split("_", 2)
    if len(chunks) < 3:
        return label
    vtype, region, part = chunks
    return f"{vtype}_{region}_{_canon_part(part)}"

def sanitize_maps_lights(maps: dict) -> dict:
    out = dict(maps or {})
    if isinstance(out.get("parts"), list):
        out["parts"] = [_canon_part(p) for p in out["parts"]]
    if isinstance(out.get("part2idx"), dict):
        out["part2idx"] = {_canon_part(k): int(v) for k,v in out["part2idx"].items()}
    return out

# ------------------------ [REGION] Regions (prefer CFG.regions if provided) ---------------------
try:
    _regions_from_cfg = list(getattr(CFG, "regions", ())) if "CFG" in globals() else []
except Exception:
    _regions_from_cfg = []
REGIONS: List[str] = _regions_from_cfg or ["Front","Side","Rear","Roof"]

# ------------------------ Region–Part maps (supports bundle or SSOT) ----------------------------
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

# ------------------------ Vocab builders (SSOT & bundle) ---------------------------------------
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

# local TYPE alias lookups (do not alter canonical TYPES order)
_type_alias_local = {
    "CAR":"Car","car":"Car","Suv":"SUV","suv":"SUV","van":"Van","VAN":"Van",
    "pickup":"Pickup","PICKUP":"Pickup",
    "Utility Vehicles":"Utility","Utility Vehicle":"Utility","Utilityvehicle":"Utility",
    "Carousel Bus":"CarouselBus","Light Truck":"LightTruck","Container Truck":"ContainerTruck","Special Vehicle":"SpecialVehicle",
}

def build_vocab_from_ssot() -> Dict[str, Any]:
    # parts union (canonicalized)
    _seen_parts = OrderedDict()
    for vtype, pairs in part_classes_by_type.items():
        for part,_r in pairs: _seen_parts[_canon_part(part)] = True
    parts = list(_seen_parts.keys())

    makes, models = _derive_makes_models_from_hierarchy(vehicle_hierarchy)

    type2idx  = _build_vocab(TYPES)
    # add local aliases as extra keys pointing to the same indices (order stays canonical)
    for alias, canon in _type_alias_local.items():
        if canon in type2idx:
            type2idx[alias] = type2idx[canon]

    make2idx   = _build_vocab(makes)
    model2idx  = _build_vocab(models)
    part2idx   = _build_vocab(parts)
    region2idx = _build_vocab(REGIONS)

    # re-apply user/global aliases if any (no mutation to canonical order lists)
    try:
        apply_all_label_aliases(type2idx, make2idx, model2idx, part2idx, cfg=CFG, verbose=False)
    except Exception as _e:
        LOG.warn("alias_apply_failed", error=str(_e))

    return {
        "TYPES": TYPES, "MAKES": makes, "MODELS": models, "PARTS": parts, "REGIONS": REGIONS,
        "type2idx": type2idx, "make2idx": make2idx, "model2idx": model2idx, "part2idx": part2idx, "region2idx": region2idx,
    }

def _coerce_idx_map(maybe)->Dict[str,int]:
    if maybe is None: return {}
    if isinstance(maybe, dict) and all(isinstance(v,(int,float)) for v in maybe.values()):
        return {str(k):int(v) for k,v in maybe.items()}
    if isinstance(maybe, dict) and "idx2name" in maybe and isinstance(maybe["idx2name"],(list,tuple)):
        return {str(name):i for i,name in enumerate(list(maybe["idx2name"]))}
    if isinstance(maybe, dict) and "names" in maybe and isinstance(maybe["names"], (list,tuple)):
        return {str(name):i for i,name in enumerate(list(maybe["names"]))}
    if isinstance(maybe,(list,tuple)):
        return {str(name):i for i,name in enumerate(list(maybe))}
    return {}

def _idx_map_to_order(idx_map: Dict[str,int]) -> List[str]:
    idx2name={}
    for k,i in idx_map.items():
        if i not in idx2name: idx2name[i]=k
    return [idx2name[i] for i in sorted(idx2name.keys())]

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

    # exact bundle order respected; after maps are built, re-apply alias lookups
    type2idx_out  = (t_map or _build_vocab(types))
    make2idx_out  = (m_map or _build_vocab(makes))
    model2idx_out = (d_map or _build_vocab(models))
    part2idx_out  = (p_map or _build_vocab(parts))
    try:
        apply_all_label_aliases(type2idx_out, make2idx_out, model2idx_out, part2idx_out, cfg=CFG, verbose=False)
    except Exception as _e:
        LOG.warn("alias_apply_failed", error=str(_e))

    return {
        "TYPES": types, "MAKES": makes, "MODELS": models, "PARTS": parts, "REGIONS": REGIONS,
        "type2idx": type2idx_out, "make2idx": make2idx_out, "model2idx": model2idx_out, "part2idx": part2idx_out, "region2idx": _build_vocab(REGIONS),
    }

# Globals adopted by SSOT or bundle
_ssot = build_vocab_from_ssot()
TYPES, MAKES, MODELS, PARTS, REGIONS = (_ssot[k] for k in ("TYPES","MAKES","MODELS","PARTS","REGIONS"))
type2idx, make2idx, model2idx, part2idx, region2idx = (_ssot[k] for k in ("type2idx","make2idx","model2idx","part2idx","region2idx"))

# ------------------------ Allow-lists (Type→Make, Make→Model, Type→Model) ----------------------
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

    # sanity: warn if a model token maps to multiple makes in SSOT & exists in vocab
    rev = {}
    for mk, toks in make_to_models.items():
        for tok in toks:
            rev.setdefault(tok, set()).add(mk)
    multi = [md for md, makes in rev.items() if len(makes) > 1 and md in model2idx]
    if multi:
        LOG.warn("ssot_model_token_multi_make", tokens=sorted(multi))

    return allowed_makes_by_type_idx, allowed_models_by_make_idx, allowed_models_by_type_idx

allowed_makes_by_type_idx, allowed_models_by_make_idx, allowed_models_by_type_idx = build_allowlists(
    type2idx, make2idx, model2idx, vehicle_hierarchy
)

# ------------------------ [ALIAS] Bundle adoption helper (re-applies aliases) ------------------
def adopt_label_maps_into_globals(maps: dict, per_type_parts: Optional[Dict[str,List[Tuple[str,str]]]] = None):
    global TYPES, MAKES, MODELS, PARTS, REGIONS
    global type2idx, make2idx, model2idx, part2idx, region2idx
    global TYPE_REGION_TO_PARTS, TYPE_PART_TO_REGIONS
    global allowed_makes_by_type_idx, allowed_models_by_make_idx, allowed_models_by_type_idx

    bundle = build_vocab_from_label_maps(maps)

    # adopt bundle vocab (exact order), then re-apply aliases to lookup maps
    TYPES, MAKES, MODELS, PARTS, REGIONS = (bundle[k] for k in ("TYPES","MAKES","MODELS","PARTS","REGIONS"))
    type2idx, make2idx, model2idx, part2idx, region2idx = (bundle[k] for k in ("type2idx","make2idx","model2idx","part2idx","region2idx"))
    try:
        apply_all_label_aliases(type2idx, make2idx, model2idx, part2idx, cfg=CFG, verbose=False)
    except Exception as _e:
        LOG.warn("alias_apply_failed", error=str(_e))

    # region maps
    if isinstance(per_type_parts, dict) and per_type_parts:
        TYPE_REGION_TO_PARTS, TYPE_PART_TO_REGIONS = compute_region_part_maps(per_type_parts)
    else:
        TYPE_REGION_TO_PARTS, TYPE_PART_TO_REGIONS = compute_region_part_maps(part_classes_by_type)

    # allow-lists against the *current* vocab
    allowed_makes_by_type_idx, allowed_models_by_make_idx, allowed_models_by_type_idx = build_allowlists(
        type2idx, make2idx, model2idx, vehicle_hierarchy
    )

    LOG.info("labelmaps_adopted",
             types=len(TYPES), makes=len(MAKES), models=len(MODELS), parts=len(PARTS),
             types_first=TYPES[:3], types_last=TYPES[-3:], aliases_applied=True)

# ==============================================================================
# PHASE 4 — Model builder (backbone + heads sized by vocab) [KEEP]
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
        g = self.veh_box_head(feats).flatten(1)
        veh_box_preds = self.veh_box_fc(g)   # raw; sigmoid applied once in decoder
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
# PHASE 3 — Bundle loader (zip/dir) + alignment audits + sha256 [KEEP]
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

# --- Part-channel remap helpers (align checkpoint rows to bundle parts order)
def _compute_part_perm(bundle_maps: dict, engine_part2idx: dict) -> Optional[List[int]]:
    try:
        old_names: List[str] = []
        if isinstance(bundle_maps.get("part2idx"), dict):
            old_names = [k for k,_ in sorted(bundle_maps["part2idx"].items(), key=lambda kv: int(kv[1]))]
        elif isinstance(bundle_maps.get("parts"), list):
            old_names = list(bundle_maps["parts"])
        if not old_names:
            return None
        perm: List[int] = []
        for nm in old_names:
            canon = _canon_part(nm)
            if canon not in engine_part2idx:
                return None
            perm.append(int(engine_part2idx[canon]))
        if perm == list(range(len(perm))):
            return None
        return perm
    except Exception:
        return None

def _apply_part_perm_to_state_dict(sd: dict, perm: List[int]) -> dict:
    if not perm: return sd
    try:
        P = len(perm)
        idx = torch.as_tensor(perm, dtype=torch.long)
        blk = torch.stack([4*idx + i for i in range(4)], dim=1).flatten()

        def pick0(t: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
            return t.index_select(0, i)

        for k in list(sd.keys()):
            t = sd[k]
            if not torch.is_tensor(t): continue
            if k.endswith("pres_head.weight") and t.dim() >= 1 and t.shape[0] == P:
                sd[k] = pick0(t, idx)
            if k.endswith("part_box.part.weight") and t.dim()==4 and t.shape[0]==P:
                sd[k] = pick0(t, idx)
            if k.endswith("part_box.part.bias") and t.dim()==1 and t.shape[0]==P:
                sd[k] = pick0(t, idx)
            if k.endswith("part_box.box.weight") and t.dim()==4 and t.shape[0]==4*P:
                sd[k] = pick0(t, blk)
            if k.endswith("part_box.box.bias") and t.dim()==1 and t.shape[0]==4*P:
                sd[k] = pick0(t, blk)
        return sd
    except Exception:
        return sd

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

    try:
        perm = _compute_part_perm(maps, part2idx) if auto_remap_parts else None
        if perm:
            state_dict = _apply_part_perm_to_state_dict(state_dict, perm)
            report["perm_hash"] = hashlib.sha256((",".join(str(i) for i in perm)).encode("utf-8")).hexdigest()[:16]
            LOG.info("part_perm_applied", P=len(perm), perm_hash=report["perm_hash"])
    except Exception as _e:
        LOG.warn("part_perm_apply_failed", error=str(_e))

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
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Bundle zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path,"r") as zf:
        members=[n for n in zf.namelist() if not n.endswith("/") ]
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
    report={"sha256":_sha256_bytes(buf),"weights_name":os.path.basename(weight_entry),"aligned":False,"trained":False,"audits":[],"perm_hash":None}
    model = build_model(backbone_name=backbone_name)
    blob = torch.load(io.BytesIO(buf), map_location="cpu")
    sd = blob.get("state_dict", None) if isinstance(blob, dict) else None
    if sd is None: sd = blob.get("model", None) if isinstance(blob, dict) else blob
    if not isinstance(sd, dict): raise ValueError("Weights payload is not a valid state_dict container.")
    sd = _strip_module_prefix(sd)

    try:
        auto_remap_parts = bool((cfg or {}).get("auto_remap_parts", True))
        perm = _compute_part_perm(maps, part2idx) if auto_remap_parts else None
        if perm:
            sd = _apply_part_perm_to_state_dict(sd, perm)
            report["perm_hash"] = hashlib.sha256((",".join(str(i) for i in perm)).encode("utf-8")).hexdigest()[:16]
            LOG.info("part_perm_applied", P=len(perm), perm_hash=report["perm_hash"])
    except Exception as _e:
        LOG.warn("part_perm_apply_failed", error=str(_e))

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

# ======================================================================
# Public API — loader facade [KEEP]
# ======================================================================

def _backbone_name_for_variant(variant:str)->str:
    if variant=="baseline": return os.getenv("BACKBONE_NAME_BASELINE","mobilevitv2_200")
    if variant=="cmt":      return os.getenv("BACKBONE_NAME_CMT","mobilevitv2_200")
    return os.getenv("BACKBONE_NAME_DEFAULT","mobilevitv2_200")

def load_model(variant:str)->Tuple[nn.Module, Dict[str,Any]]:
    bundle_dir = os.getenv("BASELINE_BUNDLE_PATH") if variant=="baseline" else os.getenv("CMT_BUNDLE_PATH")
    bundle_zip = None
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

    LOG.warn("bundle_missing", variant=variant, note="Building base model from SSOT (no checkpoint).")
    model = build_model(backbone_name=bb_name)
    report = {"aligned": _assure_alignment_counts(model, TYPES, MAKES, MODELS, PARTS),
              "trained": False, "load_notes":["no_bundle_found"]}
    return model, report

# ======================================================================
# PHASE 5 — Inference runner + timings [KEEP]
# ======================================================================

import math, base64, random
import numpy as _np
from time import perf_counter
from contextlib import nullcontext

# ---------- ENV knobs ----------
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
AMP          = _env_bool("AMP", 1)
WARMUP_STEPS = _env_int("WARMUP_STEPS", 2)

# Cascade thresholds / temps
TAU_TYPE  = _env_float("TAU_TYPE", 0.70)
TAU_MAKE  = _env_float("TAU_MAKE", 0.70)
TAU_MODEL = _env_float("TAU_MODEL", 0.70)
TEMP_TYPE  = _env_float("TEMP_TYPE", 1.00)
TEMP_MAKE  = _env_float("TEMP_MAKE", 1.00)
TEMP_MODEL = _env_float("TEMP_MODEL", 1.00)

PART_TAU   = _env_float("PART_TAU", 0.30)
AREA_MIN_FRAC = _env_float("PART_BOX_AREA_MIN_FRAC", 0.0009)
AREA_MAX_FRAC = _env_float("PART_BOX_AREA_MAX_FRAC", 0.60)
ALLOWLIST_EN  = _env_bool("PART_ALLOWLIST_EN", 0)

ENABLE_COLOR = _env_bool("ENABLE_COLOR", 0)
ENABLE_PLATE = _env_bool("ENABLE_PLATE", 0)

# Cascade CFG injection
class _CFG: pass
CFG_INFER = _CFG()
CFG_INFER.tau_type  = float(TAU_TYPE);  CFG_INFER.tau_make  = float(TAU_MAKE);  CFG_INFER.tau_model  = float(TAU_MODEL)
CFG_INFER.temp_type = float(TEMP_TYPE); CFG_INFER.temp_make = float(TEMP_MAKE); CFG_INFER.temp_model = float(TEMP_MODEL)
try:
    LOG.info("cascade_cfg", tau=[CFG_INFER.tau_type, CFG_INFER.tau_make, CFG_INFER.tau_model], temp=[CFG_INFER.temp_type, CFG_INFER.temp_make, CFG_INFER.temp_model])
except Exception:
    pass

# Deterministic numerics & eval lock
try:
    import torch.backends.cudnn as _cudnn
    _cudnn.benchmark = False
    _cudnn.deterministic = True
except Exception:
    pass
def _lock_eval_mode(model: nn.Module) -> nn.Module:
    model.eval()
    for m in model.modules():
        if getattr(m, "training", False):
            m.training = False
    return model

# ---------- StageTimer ----------
class StageTimer:
    ORDER = ["setup","vocab","bundle","preproc","warmup","forward","decode","postproc","color","plate","s3","db"]
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

# ---------- Helpers (legacy kept + authorized additions) ----------
def crop_by_xyxy(img: Image.Image, box: Tuple[float,float,float,float]) -> Image.Image:
    x1,y1,x2,y2 = [int(round(v)) for v in box]
    x1 = max(0, min(img.width-1,  x1))
    y1 = max(0, min(img.height-1, y1))
    x2 = max(x1+1, min(img.width,  x2))
    y2 = max(y1+1, min(img.height, y2))
    return img.crop((x1,y1,x2,y2))

def _color_name_from_rgb(r:int,g:int,b:int)->str:
    mx = max(r,g,b); mn = min(r,g,b)
    if mx < 40: return "Black"
    if mn > 200: return "White"
    if abs(r-g)<15 and abs(g-b)<15: return "Gray"
    if r>g and r>b: return "Red" if g<b else "Orange"
    if g>r and g>b: return "Green"
    if b>r and b>g: return "Blue"
    if r>200 and g>200 and b<80: return "Yellow"
    if r>160 and b>160 and g<120: return "Purple"
    if g>160 and b>160 and r<120: return "Cyan"
    return "Color"

def detect_vehicle_color(image_pil: Image.Image) -> List[Dict[str, float]]:
    im = image_pil.convert("RGB")
    small = im.resize((96,96), Image.BILINEAR)
    pal = small.convert("P", palette=Image.ADAPTIVE, colors=8).convert("RGB")
    data = _np.asarray(pal).reshape(-1,3)
    uniq, counts = _np.unique(data, axis=0, return_counts=True)
    total = counts.sum()
    rows = []
    for (r,g,b), c in sorted(zip(uniq, counts), key=lambda t:-t[1]):
        frac = float(c)/float(total)
        rows.append({"name": _color_name_from_rgb(int(r),int(g),int(b)),
                     "fraction": frac,
                     "conf": min(0.99, 0.5 + 0.5*frac)})
    return rows[:3]

def read_plate_text(image_pil: Image.Image) -> Dict[str, Any]:
    try:
        import pytesseract
    except Exception:
        return {"text":"", "conf":0.0}
    im = ImageOps.autocontrast(image_pil.convert("L"))
    try:
        data = pytesseract.image_to_data(im, output_type='dict')
        words = [w for w,cnf in zip(data.get("text",[]), data.get("conf",[])) if (w and str(cnf).isdigit())]
        confs = [float(c) for c in data.get("conf",[]) if str(c).isdigit()]
        text = " ".join(words).strip()
        conf = (sum(confs)/max(1,len(confs)))/100.0 if confs else 0.0
        return {"text": text, "conf": conf}
    except Exception:
        txt = pytesseract.image_to_string(im).strip()
        return {"text": txt, "conf": 0.0}

# ---------- Model cache + warm-up ----------
_MODEL_CACHE: dict[str, Tuple[nn.Module, dict]] = {}
_WARMED: set[str] = set()

def _get_model_for_variant(variant:str)->Tuple[nn.Module, dict]:
    key = variant.strip().lower()
    if key in _MODEL_CACHE: return _MODEL_CACHE[key]
    try:
        m, rep = load_model(key)
    except Exception as e:
        LOG.warn("load_model_error", variant=key, error=str(e))
        m, rep = build_model(backbone_name=os.getenv("BACKBONE_NAME_BASELINE","mobilevitv2_200")), {"aligned":True,"trained":False,"load_notes":["fallback_build"]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _lock_eval_mode(m.to(dev))
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

# ---------- Decoding helpers ----------
def _veh_head_to_xyxy(veh_pred, img_size:int):
    if not (isinstance(veh_pred, torch.Tensor) and veh_pred.dim()==2 and veh_pred.shape[1]==4):
        return None
    t = veh_pred[0].detach().float().cpu()
    cx,cy,w,h = torch.sigmoid(t[0])*img_size, torch.sigmoid(t[1])*img_size, torch.sigmoid(t[2])*img_size, torch.sigmoid(t[3])*img_size
    x1,y1 = max(0.0, float(cx-0.5*w)), max(0.0, float(cy-0.5*h))
    x2,y2 = min(img_size-1.0, float(cx+0.5*w)), min(img_size-1.0, float(cy+0.5*h))
    return (x1,y1,x2,y2)

def _veh_from_parts_quantile(parts_boxes, parts_scores, img_size:int, trim=0.10):
    if not parts_boxes: return None
    xs1=_np.array([b[0] for b in parts_boxes],dtype=_np.float32)
    ys1=_np.array([b[1] for b in parts_boxes],dtype=_np.float32)
    xs2=_np.array([b[2] for b in parts_boxes],dtype=_np.float32)
    ys2=_np.array([b[3] for b in parts_boxes],dtype=_np.float32)
    wq=lambda arr,q: float(_np.quantile(arr,q))
    x1=wq(xs1,trim); y1=wq(ys1,trim); x2=wq(xs2,1.0-trim); y2=wq(ys2,1.0-trim)
    minw=0.10*img_size; minh=0.10*img_size
    if (x2-x1)<minw: x2=x1+minw
    if (y2-y1)<minh: y2=y1+minh
    return (max(0.0,x1), max(0.0,y1), min(img_size-1.0,x2), min(img_size-1.0,y2))

def _veh_from_any_part_heat(hm, img_size:int):
    if hm is None: return None
    flat = torch.softmax(hm.view(-1), dim=0)
    idx = int(torch.argmax(flat).item()); H,W = hm.shape
    gi, gj = idx // W, idx % W
    cell_w, cell_h = (img_size/W), (img_size/H)
    cx=(gj+0.5)*cell_w; cy=(gi+0.5)*cell_h
    w=max(cell_w,0.10*img_size); h=max(cell_h,0.10*img_size)
    x1,y1 = cx-0.5*w, cy-0.5*h; x2,y2 = cx+0.5*w, cy+0.5*h
    return (max(0.0,x1), max(0.0,y1), min(img_size-1.0,x2), min(img_size-1.0,y2))

def _filter_parts_debug_list(parts_debug_sq, img_size=IMG_SIZE,
                             area_min_frac=AREA_MIN_FRAC,
                             area_max_frac=AREA_MAX_FRAC,
                             allowlist_en=ALLOWLIST_EN,
                             type_name=None):
    try:
        sqA = float(img_size) * float(img_size)
        minA = float(area_min_frac) * sqA
        maxA = float(area_max_frac) * sqA

        def ok_area(b):
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                return False
            x1, y1, x2, y2 = [float(v) for v in b]
            w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
            a = w * h
            return (minA <= a <= maxA)

        out = []
        for p in (parts_debug_sq or []):
            b = p.get("box_sq") or []
            if not ok_area(b):
                continue
            out.append(p)
        LOG.info("parts_debug_hardening", kept=len(out),
                 area_min=area_min_frac, area_max=area_max_frac,
                 allowlist=bool(allowlist_en))
        return out
    except Exception as _e:
        LOG.warn("parts_debug_hardening_failed", error=str(_e))
        return parts_debug_sq or []

# ---------- Inference runner ----------
def run_inference(img_bytes: bytes, variant: str="baseline", analysis_id: Optional[str]=None):
    """
    Returns: (dets, timings, metrics)
    dets: {
      "type","type_conf","make","make_conf","model","model_conf",
      "thresholds": {...}, "below_threshold": {...},
      "parts": [...],
      "colors": [ {finish, base, lightness, conf} × 1..3 ],
      "plate_text","plate_conf",
      "veh_box","veh_box_src","plate_box",
      "_debug_parts_sq":[...], "_debug_pad_scale":{...}
    }
    """
    TOPK     = _env_int("PARTS_DEBUG_TOPK", 12)
    DRAW_ALL = _env_bool("PARTS_DRAW_ALL", 0)

    aid = analysis_id or f"an_{int(time.time()*1000)}_{random.randint(100,999)}"
    timer = StageTimer(aid, variant)

    # setup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    ow, oh = img.size
    timer.tick("setup")

    # vocab touch
    _ = (len(TYPES), len(MAKES), len(MODELS), len(PARTS), len(REGIONS))
    timer.tick("vocab")

    # bundle/model
    model, rep = _get_model_for_variant(variant)
    timer.tick("bundle")

    # preproc
    arr = _np.asarray(img)
    tCHW = torch.from_numpy(arr.copy()).permute(2,0,1).float()/255.0
    sq, pad, scale = s_letterbox_fit(tCHW, int(IMG_SIZE))  # CHW
    X = sq.unsqueeze(0).to(dev)  # BCHW
    timer.tick("preproc")

    # warmup
    if warmup_model is not None:
        warm_ms = warmup_model(model, dev, int(IMG_SIZE), steps=WARMUP_STEPS)
    else:
        _maybe_warmup(model, dev, int(IMG_SIZE), steps=WARMUP_STEPS)
        warm_ms = 0
    timer.tick("warmup")

    # forward
    use_amp = bool(AMP and dev.startswith("cuda") and torch.cuda.is_available())
    ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
    with torch.no_grad():
        with ctx:
            out = model(X)
            if dev.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
    timer.tick("forward")

    # ---- decode: type/make/model cascade (+ voter blend) ----
    try:
        lt, lm, lk = s_blend_logits(out)
        LOG.info("voter_blend_used", used=True)
    except Exception:
        lt = out.get("type_logits"); lm = out.get("make_logits"); lk = out.get("model_logits")
        LOG.info("voter_blend_used", used=False)

    cascade = s_cascade_infer(
         lt, lm, lk,
         CFG_INFER,
         allowed_makes_by_type_idx if "allowed_makes_by_type_idx" in globals() else None,
         allowed_models_by_make_idx if "allowed_models_by_make_idx" in globals() else None
    )

    t_id, m_id, k_id = cascade["type"], cascade["make"], cascade["model"]
    confs = list(cascade["confs"]) + [0.0]*(3-len(cascade["confs"]))

    type_name  = (TYPES[t_id]  if t_id is not None and 0<=t_id<len(TYPES)  else None)
    make_name  = (MAKES[m_id]  if m_id is not None and 0<=m_id<len(MAKES)  else None)
    model_name = (MODELS[k_id] if k_id is not None and 0<=k_id<len(MODELS) else None)

    # persist-always best-effort if any None (masked top-1)
    make_mask_idx  = allowed_makes_by_type_idx.get(int(t_id)) if ('allowed_makes_by_type_idx' in globals() and t_id is not None) else None
    model_mask_idx = allowed_models_by_make_idx.get(int(m_id)) if ('allowed_models_by_make_idx' in globals() and m_id is not None) else None

    def _masked_top1(logits, mask_idx, temp):
        if logits is None or not torch.is_tensor(logits) or logits.numel() == 0: return None, 0.0
        z = logits.float().clone()
        if mask_idx:
            mask = torch.zeros(z.shape[-1], dtype=torch.bool, device=z.device)
            mask[torch.as_tensor(list(mask_idx), device=z.device, dtype=torch.long)] = True
            z[..., ~mask] = -1e9
        p = torch.softmax(z / max(1e-6, float(temp)), dim=-1)
        idx = int(p.argmax(dim=-1)[0].item()) if p.dim() == 2 else int(p.argmax().item())
        conf = float(p.view(-1)[idx].item())
        return idx, conf

    def _top1(logits, temp):
        if logits is None or not torch.is_tensor(logits) or logits.numel() == 0: return None, 0.0
        p = torch.softmax(logits.float() / max(1e-6, float(temp)), dim=-1)
        idx = int(p.argmax(dim=-1)[0].item()) if p.dim() == 2 else int(p.argmax().item())
        return idx, float(p.view(-1)[idx].item())

    if type_name is None:
        t1, c1 = _top1(lt, CFG_INFER.temp_type)
        type_name = (TYPES[t1] if t1 is not None and 0<=t1<len(TYPES) else None)
        confs[0] = float(c1)
    if make_name is None:
        m1, c2 = _masked_top1(lm, make_mask_idx, CFG_INFER.temp_make)
        make_name = (MAKES[m1] if m1 is not None and 0<=m1<len(MAKES) else None)
        confs[1] = float(c2)
    if model_name is None:
        k1, c3 = _masked_top1(lk, model_mask_idx, CFG_INFER.temp_model)
        model_name = (MODELS[k1] if k1 is not None and 0<=k1<len(MODELS) else None)
        confs[2] = float(c3)

    # ---- parts + vehicle box ----
    P = len(PARTS)
    part_logits = out.get("part_logits")
    s_assert_head_grid_matches(type("CFG", (), {"img_size": int(IMG_SIZE)})(), part_logits, model)
    bbox_preds  = out.get("bbox_preds")
    parts_list  = []
    parts_debug_sq = []
    veh_box_sq  = None

    if torch.is_tensor(part_logits):
        B,Pp,H,W = part_logits.shape
        probs = torch.sigmoid(part_logits)[0]
        scores = probs.amax(dim=(-1,-2))
        for p_idx, sc in enumerate(scores.tolist()):
            if float(sc) >= float(PART_TAU):
                parts_list.append({"name": PARTS[p_idx] if p_idx < len(PARTS) else f"part_{p_idx}",
                                   "conf": float(sc)})

        veh_box_pred = out.get("veh_box_preds")
        vh = None
        if veh_box_pred is not None:
            v = s_veh_box_from_any(veh_box_pred, int(IMG_SIZE))
            if v is not None: vh = tuple(float(v[0,i].item()) for i in range(4))

        heat_map = torch.sigmoid(part_logits)[0]
        heat_max = heat_map.amax(dim=(-1, -2))
        pres_logits = out.get("part_present_logits")
        pres_max = None
        if torch.is_tensor(pres_logits):
            pl = pres_logits
            if pl.dim() == 4 and tuple(pl.shape[-2:]) == (H, W):
                pres_max = torch.sigmoid(pl)[0].amax(dim=(-1, -2))
            else:
                pres_max = torch.sigmoid(pl).view(B, -1)[0][:Pp]
        conf_vec = 0.5 * heat_max + 0.5 * pres_max if pres_max is not None and pres_max.shape[0]==heat_max.shape[0] else heat_max

        allowed_parts = None
        if type_name is not None:
            allowed_parts = set(TYPE_PART_TO_REGIONS.get(type_name, {}).keys())

        kept = []
        for p_idx, sc in enumerate(conf_vec.tolist()):
            if p_idx >= P: break
            p_name = PARTS[p_idx]
            if allowed_parts is not None and p_name not in allowed_parts:
                continue
            if float(sc) < float(PART_TAU):
                continue
            kept.append((p_idx, p_name, float(sc)))
        kept = sorted(kept, key=lambda t: t[2], reverse=True)[:12]

        BP4 = s_unpack_BP4HW(bbox_preds, P, H, W) if torch.is_tensor(bbox_preds) else None
        XYXY = s_decode_dxdy(BP4, H, W, int(IMG_SIZE)) if BP4 is not None else None

        parts_boxes = []
        parts_scores = []
        for (p_idx, p_name, sc) in kept:
            heat = heat_map[p_idx]
            flat = torch.softmax(heat.reshape(-1), dim=0)
            arg  = int(torch.argmax(flat).item())
            gi, gj = arg // W, arg % W
            if XYXY is not None:
                x1 = float(XYXY[0, p_idx, 0, gi, gj].item())
                y1 = float(XYXY[0, p_idx, 1, gi, gj].item())
                x2 = float(XYXY[0, p_idx, 2, gi, gj].item())
                y2 = float(XYXY[0, p_idx, 3, gi, gj].item())
            else:
                x1,y1,x2,y2 = s_box_from_heatmap(heat, int(IMG_SIZE), gamma=3.2, temperature=0.8)

            minw = 0.04 * IMG_SIZE; minh = 0.04 * IMG_SIZE
            if (x2 - x1) < minw:
                cx = 0.5 * (x1 + x2); x1, x2 = cx - 0.5 * minw, cx + 0.5 * minw
            if (y2 - y1) < minh:
                cy = 0.5 * (y1 + y2); y1, y2 = cy - 0.5 * minh, cy + 0.5 * minh
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(float(IMG_SIZE - 1), x2); y2 = min(float(IMG_SIZE - 1), y2)

            parts_debug_sq.append({"name": p_name, "conf": sc, "box_sq": [x1, y1, x2, y2]})
            parts_boxes.append([x1,y1,x2,y2])
            parts_scores.append(sc)

        vp = _veh_from_parts_quantile(parts_boxes, parts_scores, int(IMG_SIZE)) if parts_boxes else None
        any_part_heat = heat_map.max(dim=0).values if torch.is_tensor(part_logits) else None
        va = _veh_from_any_part_heat(any_part_heat, int(IMG_SIZE))

        candidates = []
        if vh is not None: candidates.append(("veh-head", vh))
        if vp is not None: candidates.append(("parts-quantile", vp))
        if va is not None: candidates.append(("any-part-heat", va))
        veh_src, veh_box_sq = ("none", None)
        if candidates:
            pref = {"veh-head":0, "parts-quantile":1, "any-part-heat":2}
            candidates.sort(key=lambda t: pref.get(t[0], 99))
            veh_src, veh_box_sq = candidates[0][0], candidates[0][1]
    else:
        veh_box_pred = out.get("veh_box_preds")
        veh_src = "none"
        if veh_box_pred is not None:
            v = s_veh_box_from_any(veh_box_pred, int(IMG_SIZE))
            if v is not None:
                veh_box_sq = tuple(float(v[0,i].item()) for i in range(4))
                veh_src = "veh-head"

    plate_box_sq = None
    if isinstance(parts_debug_sq, list) and parts_debug_sq:
        cand = [p for p in parts_debug_sq if str(p.get("name","")).lower().endswith("plate")]
        if cand:
            best = max(cand, key=lambda r: float(r.get("conf",0.0)))
            bb = best.get("box_sq")
            if isinstance(bb, (list,tuple)) and len(bb)==4:
                plate_box_sq = tuple(float(v) for v in bb)

    veh_box   = s_invert_letterbox_box(veh_box_sq,  ow, oh, pad, scale)  if veh_box_sq  is not None else None
    plate_box = s_invert_letterbox_box(plate_box_sq, ow, oh, pad, scale) if plate_box_sq is not None else None
    timer.tick("decode")

    # ---- assemble dets ----
    dets = {
        "type": type_name,   "type_conf":  float(confs[0]),
        "make": make_name,   "make_conf":  float(confs[1]),
        "model": model_name, "model_conf": float(confs[2]),

        # [PH9] strict thresholds naming (+ part_presence_min)
        "thresholds": {
            "type_min": float(TAU_TYPE),
            "make_min": float(TAU_MAKE),
            "model_min": float(TAU_MODEL),
            "part_presence_min": float(PART_TAU),
        },
        "below_threshold": {
            "type":  bool(confs[0] < float(TAU_TYPE)),
            "make":  bool(confs[1] < float(TAU_MAKE)),
            "model": bool(confs[2] < float(TAU_MODEL)),
        },

        "parts": parts_list,
        "colors": [],  # FBL array
        "plate_text": "",
        "plate_conf": 0.0,
        "veh_box": veh_box,
        "veh_box_src": veh_src if 'veh_src' in locals() else ("veh-head" if veh_box_sq is not None else "none"),
        "plate_box": plate_box,
        "_debug_parts_sq": _filter_parts_debug_list(parts_debug_sq),
        "_debug_pad_scale": {"pad":[float(pad[0]),float(pad[1])], "scale": float(scale)},
    }
    timer.tick("postproc")

    # ---- color stage → FBL only ----
    if ENABLE_COLOR:
        try:
            veh_crop_box = dets.get("veh_box")
            if _COLOR is not None and hasattr(_COLOR, "detect_vehicle_color"):
                colors_fbl = _COLOR.detect_vehicle_color(img, veh_box=veh_crop_box, allow_multitone=True, timeout_s=20)
            else:
                crop = img if veh_crop_box is None else img.crop(tuple(map(int, veh_crop_box)))
                coarse = detect_vehicle_color(crop)
                labels = [c.get("name","") for c in (coarse or [])]
                confsC = [float(c.get("conf",0.0)) for c in (coarse or [])]
                FINISH = {"Metallic","Matte","Glossy"}
                LIGHT  = {"Light","Medium","Dark"}
                BASE   = {"Red","Orange","Yellow","Green","Blue","Purple","Pink","White","Gray","Black","Silver","Gold","Brown","Beige","Maroon","Cyan"}
                def _parse(label):
                    finish = base = lightness = None
                    toks = [t.strip().capitalize() for t in str(label or "").split() if t.strip()]
                    for t in toks:
                        if t in FINISH and finish is None: finish = t
                    for t in toks:
                        if t in LIGHT and lightness is None: lightness = t
                    for t in toks:
                        if t in BASE and base is None: base = t; break
                    if base is None and toks: base = toks[-1]
                    return {"finish": finish, "base": base, "lightness": lightness}
                keep = [0] + ([1] if len(labels)>1 and len(confsC)>1 and confsC[1]>=0.70 else []) + ([2] if len(labels)>2 and len(confsC)>2 and confsC[2]>=0.70 else [])
                colors_fbl = []
                for i in keep[:3]:
                    fbl=_parse(labels[i]); fbl["conf"]=float(confsC[i] if i<len(confsC) else confsC[0] if confsC else 0.0)
                    colors_fbl.append(fbl)
                if not colors_fbl:
                    colors_fbl=[{"finish":None,"base":None,"lightness":None,"conf":0.0}]
            dets["colors"] = colors_fbl
        except Exception as _e:
            LOG.warn("color_error", error=str(_e))
    else:
        LOG.warn("color_disabled", note="ENABLE_COLOR=0")
    timer.tick("color")

    # ---- plate OCR ----
    if ENABLE_PLATE and not dets.get("plate_text"):
        try:
            if _PLATE is not None and hasattr(_PLATE, "recognize_plate"):
                boxes = []
                if dets.get("plate_box"): boxes.append(tuple(map(float, dets["plate_box"])))
                elif dets.get("veh_box"):  boxes.append(tuple(map(float, dets["veh_box"])))
                got = _PLATE.recognize_plate(img, plate_boxes_orig=(boxes or None), topk=1, timeout_s=20)
                dets["plate_text"] = (got.get("text") or "").upper()
                dets["plate_conf"] = float(got.get("confidence") or 0.0)
                if not dets.get("plate_box") and got.get("bbox"):
                    dets["plate_box"] = tuple(float(v) for v in got["bbox"])
                if got.get("candidates"): dets["plate_candidates"] = got["candidates"]
            else:
                crop_src = None
                if dets.get("plate_box"): crop_src = img.crop(tuple(int(v) for v in dets["plate_box"]))
                elif dets.get("veh_box"): crop_src = img.crop(tuple(int(v) for v in dets["veh_box"]))
                if crop_src is not None:
                    resp = read_plate_text(crop_src)
                    dets["plate_text"] = (resp.get("text","") or "").upper()
                    dets["plate_conf"] = float(resp.get("conf", 0.0))
        except Exception as ee:
            LOG.warn("plate_error", error=str(ee))
    else:
        LOG.warn("plate_disabled", note="ENABLE_PLATE=0")
    timer.tick("plate")

    # (placeholders for S3 / DB)
    timer.tick("s3")
    timer.tick("db")

    timings = timer.done()

    # ---- metrics
    gflops_val = estimate_gflops(model, int(IMG_SIZE)) if estimate_gflops else _estimate_gflops(model, int(IMG_SIZE))
    mem    = per_infer_memory_metrics(dev) if per_infer_memory_metrics else {"mem_gb": None, "memory_usage": None}
    lat_ms = float(timings.get("total", 0)) - float(warm_ms or 0)
    if lat_ms < 0: lat_ms = 0.0

    metrics = {
        "latency_ms": lat_ms,
        "gflops": gflops_val,
        "memory_gb": mem.get("mem_gb"),
        "mem_gb": mem.get("mem_gb"),             # back-compat
        "memory_usage": mem.get("memory_usage"),
        "device": dev,
        "trained": bool(rep.get("trained", False)),
    }

    try:
        LOG.info("decode_summary",
                 kept_parts=int(len((dets.get("_debug_parts_sq") or []))) if isinstance(dets, dict) else None,
                 grid=list(part_logits.shape[-2:]) if torch.is_tensor(part_logits) else None,
                 veh_src=dets.get("veh_box_src"))
    except Exception as _e:
        LOG.warn("decode_summary_failed", error=str(_e))

    try:
        _dbg = (dets.get("_debug_parts_sq") or []) if isinstance(dets, dict) else []
        if isinstance(_dbg, list):
            if not DRAW_ALL:
                _dbg = _dbg[:int(TOPK)]
            dets["_debug_parts_sq"] = _dbg
        LOG.info("parts_debug_toggles", topk=int(TOPK), draw_all=bool(DRAW_ALL), kept=len(dets.get("_debug_parts_sq") or []))
    except Exception as _e:
        LOG.warn("parts_debug_toggles_failed", error=str(_e))

    return dets, timings, metrics

if __name__ == "__main__":
    LOG.info("selftest_begin")
    m, rep = load_model(os.getenv("VARIANT","baseline"))
    LOG.info("selftest_model", head_sig=_model_vocab_signature(m), aligned=rep.get("aligned"), trained=rep.get("trained"))
    LOG.info("selftest_ok")

# ======================== END OF EC2 CODE ENGINE (REWRITTEN — Phase-A Stable, Phase-9 Keys, SSOT-Aligned + Alias Hooks) =======================

