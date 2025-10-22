import os, time, json, math
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image
from .utils import JsonLogger

_LOG = JsonLogger("cvitx-engine", json_logs=os.getenv("JSON_LOGS","1"))

# -------- StageTimer with Colab-like stage timings (logged, not stored) --------
class StageTimer:
    def __init__(self, logger=_LOG):
        self.logger = logger
        self._t0 = time.perf_counter()
        self._marks = []  # list of (name, ms)
        self.logger.info("stages_begin")

    def stage(self, name: str):
        class _Ctx:
            def __init__(self, outer, nm): self.o=outer; self.n=nm
            def __enter__(self):
                self.t = time.perf_counter()
                self.o.logger.info("stage_start", name=self.n)
                return self
            def __exit__(self, exc_type, exc, tb):
                ms = int((time.perf_counter()-self.t)*1000)
                self.o._marks.append((self.n, ms))
                self.o.logger.info("stage_end", name=self.n, ms=ms, ok=exc is None)
        return _Ctx(self, name)

    def summary(self):
        total = int((time.perf_counter()-self._t0)*1000)
        per = [{"name":n,"ms":ms} for (n,ms) in self._marks]
        self.logger.info("stages_summary", total_ms=total, per_stage=per)
        return {"total_ms": total, "per_stage": per}

# -------- Minimal model cache / loader (stub for Phase 1) --------
_MODEL_CACHE: Dict[str, Any] = {}

class _DummyModel:
    def __init__(self, variant: str):
        self.variant = variant
    def infer(self, img: Image.Image) -> Dict[str, Any]:
        w,h = img.size
        box = (max(0,int(0.1*w)), max(0,int(0.1*h)), min(w,int(0.9*w)), min(h,int(0.6*h)))
        return {
            "type": "CAR", "type_conf": 0.88 if self.variant=="baseline" else 0.92,
            "make": "TOYOTA", "make_conf": 0.83 if self.variant=="baseline" else 0.90,
            "model": "FORTUNER", "model_conf": 0.78 if self.variant=="baseline" else 0.87,
            "parts": [{"name":"headlamp","conf":0.95},{"name":"grille","conf":0.93}],
            "colors": [],
            "plate_text": "ABC1234",
            "boxes": [box],
            "veh_box": box,
        }
    def report_gflops(self) -> float:
        return 2.1 if self.variant=="baseline" else 2.3

def load_model(variant: str):
    if variant not in ("baseline","cmt"):
        raise ValueError(f"Unknown variant: {variant}")
    m = _MODEL_CACHE.get(variant)
    if m is None:
        # later: read BASELINE_BUNDLE_PATH / CMT_BUNDLE_PATH here
        m = _DummyModel(variant)
        _MODEL_CACHE[variant] = m
        _LOG.info("model_loaded", variant=variant, cached=False)
    else:
        _LOG.info("model_loaded", variant=variant, cached=True)
    return m

# -------- Inference runner with stage timings (pre/forward/post/color/plate) ----
def run_inference(image_bytes: bytes, variant: str,
                  enable_color: bool=False, enable_plate: bool=False) -> Dict[str, Any]:
    st = StageTimer(_LOG)
    with st.stage("preprocess"):
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        _ = (img.width, img.height)  # noop; placeholder for letterbox/resize

    with st.stage("model_forward"):
        model = load_model(variant)
        dets = model.infer(img)

    with st.stage("postprocess"):
        # placeholder for cascade / bbox decode etc.
        pass

    if enable_color:
        with st.stage("color"):
            # placeholder: just mirror a simple color result; real logic in Phase 3+
            dets["colors"] = [{"finish":"Metallic","base":"White","lightness":"Light","conf":0.90}]

    if enable_plate:
        with st.stage("plate"):
            # placeholder for OCR call
            dets["plate_conf"] = 0.85

    meta = st.summary()
    dets["_timing"] = meta  # kept in memory/logs; callers may drop before DB
    return dets

# --------- CLI self-test (no S3, no DB) ---------------------------------------
if __name__ == "__main__":
    # create a dummy image (640x480 gray)
    img = Image.new("RGB", (640,480), (200,200,200))
    buf = BytesIO(); img.save(buf, format="JPEG", quality=90)
    out = run_inference(buf.getvalue(), os.getenv("VARIANT","baseline"),
                        enable_color=os.getenv("ENABLE_COLOR","0")=="1",
                        enable_plate=os.getenv("ENABLE_PLATE","0")=="1")
    _LOG.info("selftest_result", variant=os.getenv("VARIANT","baseline"),
              type=out.get("type"), make=out.get("make"), model=out.get("model"),
              latency_ms=out["_timing"]["total_ms"])
