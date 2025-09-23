import os, json
from ultralytics import YOLO

YOLO_WEIGHTS     = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")  # auto-downloads
YOLO_DEVICE      = os.getenv("YOLO_DEVICE", "cpu")
YOLO_CONF_THRES  = float(os.getenv("YOLO_CONF_THRES", "0.15"))
YOLO_IOU_THRES   = float(os.getenv("YOLO_IOU_THRES", "0.45"))
CLASS_NAMES_PATH = os.getenv("YOLO_CLASS_NAMES_JSON")  # path OR raw JSON

# COCO vehicle ids: 2 car, 3 motorbike, 5 bus, 7 truck
VEHICLE_CLS = {2, 3, 5, 7}

DEFAULT_COCO = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
    "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
]

def _load_names():
    if CLASS_NAMES_PATH:
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, "r") as f:
                data = json.load(f)
        else:
            try:
                data = json.loads(CLASS_NAMES_PATH)
            except Exception:
                data = None
        if isinstance(data, dict):
            return [data[str(i)] for i in range(len(data))]
        if isinstance(data, list):
            return data
    return DEFAULT_COCO

CLASS_NAMES = _load_names()

class YoloDetector:
    def __init__(self):
        self.model = YOLO(YOLO_WEIGHTS)

    def infer(self, frame_bgr):
        """
        Return [[x1,y1,x2,y2,conf,cls], ...] on ORIGINAL frame.
        If using COCO, keep only vehicle classes for the prototype.
        """
        res = self.model.predict(
            source=frame_bgr,
            conf=YOLO_CONF_THRES,
            iou=YOLO_IOU_THRES,
            device=YOLO_DEVICE,
            imgsz=640,
            verbose=False,
        )
        out = []
        for r in res:
            bs = getattr(r, "boxes", None)
            if bs is None:
                continue
            for b in bs:
                cls = int(b.cls[0].item())
                if CLASS_NAMES is DEFAULT_COCO and cls not in VEHICLE_CLS:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0].item())
                if x2 > x1 and y2 > y1:
                    out.append([x1, y1, x2, y2, conf, cls])
        return out

    def class_name(self, cls_id: int) -> str:
        if 0 <= cls_id < len(CLASS_NAMES):
            return CLASS_NAMES[cls_id]
        return f"cls_{cls_id}"
