# Auto-generated CMT wrapper to avoid 'import *' issues.
import uuid, re
UUID_RE = re.compile(r'^[0-9a-fA-F-]{36}$')
def _is_uuid(s:str)->bool:
    try:
        return bool(UUID_RE.match(s)) and str(uuid.UUID(s)) == s.lower()
    except Exception:
        return False
import os
from api.workers import image_worker_baseline as base

# Force CMT invariants
base.VARIANT   = "cmt"
base.QUEUE_ENV = "SQS_ANALYSIS_CMT_URL"
base.S3_SUBDIR = "cmt"
base.QUEUE_URL = os.getenv(base.QUEUE_ENV)
base.SQS_URL = base.QUEUE_URL

if __name__ == "__main__":
    base.main()

# --- Color (FBL) injection (Finish / Base / Lightness + single conf) ---
try:
    from api.analysis.utils import detect_vehicle_color, fbl_overall_conf
    _veh_box = (detection.get("veh_box") if "detection" in locals() else None)
    _colors_fbl = detect_vehicle_color(image_pil, veh_box=_veh_box)
    _overall = fbl_overall_conf(_colors_fbl)
    results.setdefault("metadata", {})
    results["metadata"]["colors_fbl"] = _colors_fbl               # [{finish,base,lightness,conf}]
    results["metadata"]["colors_overall_conf"] = _overall         # single 0..1
except Exception:
    pass

