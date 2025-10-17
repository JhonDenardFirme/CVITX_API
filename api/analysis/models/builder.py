import os, time
from typing import Any, Dict

# If you later add torch/onnx, switch here based on ENV paths:
BASELINE_CKPT = os.getenv("BASELINE_CHECKPOINT_PATH", "/models/baseline/mobilevit_partaware_baseline.ckpt")
CMT_CKPT      = os.getenv("CMT_CHECKPOINT_PATH",      "/models/cmt/mobilevit_partaware_cmt.ckpt")

class DummyModel:
    def __init__(self, variant: str):
        self.variant = variant
    def infer(self, image_bytes: bytes) -> Dict[str, Any]:
        # Produce deterministic dummy output
        return {
            "type": "CAR", "type_conf": 0.88 if self.variant=="baseline" else 0.92,
            "make": "TOYOTA", "make_conf": 0.83 if self.variant=="baseline" else 0.90,
            "model": "FORTUNER", "model_conf": 0.78 if self.variant=="baseline" else 0.87,
            "parts": [{"name":"headlamp","conf":0.95},{"name":"grille","conf":0.93}],
            "colors": ["white","black"],
            "plate_text": "ABC1234",
            "boxes": [(20,20,180,120)],  # x1,y1,x2,y2
        }
    def report_gflops(self) -> float:
        return 2.1 if self.variant=="baseline" else 2.3

def get_model(variant: str):
    # If you later detect existing ckpt files, return a real model instead.
    # For now always return DummyModel to keep Phase 3 shippable.
    return DummyModel(variant)
