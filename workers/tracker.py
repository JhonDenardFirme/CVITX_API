import os
from typing import List, Dict
from deep_sort_realtime.deepsort_tracker import DeepSort

MAX_AGE      = int(os.getenv("DEEPSORT_MAX_AGE", "50"))
N_INIT       = int(os.getenv("DEEPSORT_N_INIT", "2"))
MAX_IOU_DIST = float(os.getenv("DEEPSORT_MAX_IOU_DISTANCE", "0.8"))
NN_BUDGET    = int(os.getenv("DEEPSORT_NN_BUDGET", "200"))
EMBEDDER     = os.getenv("DEEPSORT_EMBEDDER", "mobilenet")

class DeepSortTracker:
    """
    Consume dets [[x1,y1,x2,y2,conf,cls], ...] → return [{id, tlbr, conf, cls}, ...]
    """
    def __init__(self):
        self.tracker = DeepSort(
            max_age=MAX_AGE, n_init=N_INIT, max_iou_distance=MAX_IOU_DIST,
            nn_budget=NN_BUDGET, embedder=EMBEDDER, half=True, bgr=True,
        )

    def update(self, dets: List[List[float]], frame) -> List[Dict]:
        ds = [((d[0],d[1],d[2],d[3]), float(d[4]), int(d[5])) for d in dets]
        tracks = self.tracker.update_tracks(ds, frame=frame)
        out = []
        # we’ll select the best conf/cls among overlapping dets for each track
        def iou(a,b):
            ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
            x1,y1=max(ax1,bx1),max(ay1,by1)
            x2,y2=min(ax2,bx2),min(ay2,by2)
            iw,ih=max(0,x2-x1),max(0,y2-y1)
            inter=iw*ih
            if inter==0: return 0.0
            return inter/float((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-9)
        for t in tracks:
            if not t.is_confirmed(): continue
            l,t0,r,b = map(int, t.to_ltrb())
            best_conf, best_cls = 0.0, -1
            for (bb, conf, cls) in ds:
                if iou((l,t0,r,b), bb) > 0.0 and conf > best_conf:
                    best_conf, best_cls = conf, cls
            out.append({"id": int(t.track_id), "tlbr": [l,t0,r,b], "conf": best_conf, "cls": best_cls})
        return out
