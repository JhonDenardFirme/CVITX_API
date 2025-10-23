import os, logging
from typing import List, Tuple
import numpy as np

# deep_sort_realtime interface
from deep_sort_realtime.deepsort_tracker import DeepSort

def _env_int(k, d): 
    try: return int(os.getenv(k, str(d)))
    except: return d

def _env_float(k, d): 
    try: return float(os.getenv(k, str(d)))
    except: return d

class DeepSortTracker:
    """
    Thin wrapper over deep_sort_realtime that ACCEPTS YOLO-style TLWH detections:
        detections = [([x,y,w,h], conf, cls_id), ...]
    and internally converts them to XYXY for DeepSort.

    Returns a list of dicts for CONFIRMED tracks on the current frame:
        [{ "id": int, "tlbr": (x1,y1,x2,y2), "conf": float, "cls": int }, ...]
    """
    def __init__(self):
        self.trk = DeepSort(
            max_age=_env_int("DEEPSORT_MAX_AGE", 50),
            n_init=_env_int("DEEPSORT_N_INIT", 3),
            max_iou_distance=_env_float("DEEPSORT_MAX_IOU_DISTANCE", 0.7),
            nn_budget=_env_int("DEEPSORT_NN_BUDGET", 200),
            embedder=os.getenv("DEEPSORT_EMBEDDER", "mobilenet"),
            half=True,     # use half precision on GPU embedder if available
            bgr=True       # frames are BGR (OpenCV)
        )
        logging.info("DeepSort Tracker initialised")
        logging.info("- max age: %s", str(_env_int("DEEPSORT_MAX_AGE", 50)))
        logging.info("- n_init: %s", str(_env_int("DEEPSORT_N_INIT", 3)))
        logging.info("- max IoU distance: %s", str(_env_float("DEEPSORT_MAX_IOU_DISTANCE", 0.7)))
        logging.info("- nn_budget: %s", str(_env_int("DEEPSORT_NN_BUDGET", 200)))
        logging.info("- embedder: %s", os.getenv("DEEPSORT_EMBEDDER", "mobilenet"))

    @staticmethod
    def _tlwh_to_xyxy(tlwh):
        x, y, w, h = tlwh
        return [int(x), int(y), int(x + w), int(y + h)]

    def update(self, detections: List[Tuple[list, float, int]], frame) -> list:
        """
        detections: [([x,y,w,h], conf, cls), ...]  # TLWH
        frame: np.ndarray (BGR)
        """
        ds_xyxy = []
        for box, conf, cls_id in detections:
            # normalize TLWH -> XYXY for deep_sort_realtime
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            if x2 <= x1 or y2 <= y1:
                continue
            ds_xyxy.append(([x1, y1, x2, y2], float(conf), int(cls_id)))

        tracks = self.trk.update_tracks(ds_xyxy, frame=frame)
        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            # prefer current bbox, skip stale
            tlbr = t.to_tlbr()
            if tlbr is None:
                continue
            x1, y1, x2, y2 = map(int, tlbr)
            out.append({
                "id": int(t.track_id),
                "tlbr": (x1, y1, x2, y2),
                "conf": float(getattr(t, "det_confidence", 1.0)),
                "cls": int(getattr(t, "det_class", -1)),
            })
        return out
