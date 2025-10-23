import os
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO

# 17 PH vehicle classes (index = class id)
CLASS_NAMES = [
    "Car", "SUV", "Pickup", "Van", "Utility Vehicle", "Motorcycle",
    "Bicycle", "E-Bike", "Pedicab", "Tricycle", "Jeepney",
    "E-Jeepney", "Bus", "Carousel Bus", "Light Truck",
    "Container Truck", "Special Vehicle"
]

def _parse_int_list(s: str | None) -> List[int]:
    if not s: return []
    out = []
    for tok in s.replace(" ","").split(","):
        if tok == "": continue
        try: out.append(int(tok))
        except: pass
    return out

class YoloDetector:
    """
    Thin adapter around Ultralytics YOLOv8 that returns detections in:
        dets = [([x, y, w, h], conf, cls_id), ...]
    Adds hygiene filters:
      - optional allowed class list (YOLO_ALLOWED_CLS="0,1,...")
      - crowd bump: if too many dets, raise conf floor
      - small-box extra conf floor
    """
    def __init__(
        self,
        weights_path: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> None:
        self.weights_path = weights_path or os.getenv("YOLO_WEIGHTS") or "yolov8n.pt"
        self.conf  = float(os.getenv("YOLO_CONF",  str(conf  if conf  is not None else 0.35)))
        self.iou   = float(os.getenv("YOLO_IOU",   str(iou   if iou   is not None else 0.45)))
        self.imgsz = int(os.getenv("YOLO_IMGSZ",   str(imgsz if imgsz is not None else 640)))
        self.device = device or os.getenv("ULTRALYTICS_DEVICE") or "cpu"

        self.allowed_cls = _parse_int_list(os.getenv("YOLO_ALLOWED_CLS", ""))
        self.crowd_thresh = int(os.getenv("YOLO_CROWD_DET_THRESHOLD","40"))
        self.crowd_bump   = float(os.getenv("YOLO_CROWD_CONF_BUMP","0.05"))
        self.small_min_side = int(os.getenv("YOLO_SMALL_MIN_SIDE","12"))
        self.small_conf_min = float(os.getenv("YOLO_SMALL_CONF_MIN","0.35"))

        self._model: YOLO | None = None

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.weights_path)
        return self._model

    def infer(self, frame: np.ndarray) -> List[Tuple[list, float, int]]:
        model = self._ensure_model()
        res = model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device
        )[0]

        dets: List[Tuple[list, float, int]] = []
        H, W = frame.shape[:2]
        if res.boxes is None:
            return dets

        raw: List[Tuple[list, float, int]] = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else -1

            # in-frame validity
            if x2 <= x1 or y2 <= y1:         continue
            if x2 <= 0 or y2 <= 0:           continue
            if x1 >= W or y1 >= H:           continue
            w, h = x2 - x1, y2 - y1
            if w < 10 or h < 10:             continue
            ar = w / max(h, 1)
            if ar > 6 or ar < 0.15:          continue

            # optional allowlist
            if self.allowed_cls and (cls_id not in self.allowed_cls):
                continue

            raw.append(([x1, y1, w, h], conf, cls_id))

        # Crowd bump: if too many, raise conf floor slightly
        conf_floor = self.conf
        if len(raw) >= self.crowd_thresh:
            conf_floor = max(conf_floor, min(0.99, self.conf + self.crowd_bump))

        for (tlwh, conf, cls_id) in raw:
            x, y, w, h = tlwh
            # small-box extra conf floor
            if min(w, h) <= self.small_min_side and conf < self.small_conf_min:
                continue
            if conf < conf_floor:
                continue
            dets.append((tlwh, conf, cls_id))

        return dets
