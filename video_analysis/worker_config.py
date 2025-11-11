from typing import Dict, Tuple

CONFIG: Dict[str, str] = {
    "AWS_REGION": "ap-southeast-2",
    "S3_BUCKET": "cvitx-uploads-dev-jdfirme",
    "SQS_VIDEO_QUEUE_URL": "https://sqs.example/video-tasks",
    "SQS_SNAPSHOT_QUEUE_URL": "https://sqs.example/snapshot-tasks",
    "SQS_VIS_TIMEOUT": "300",
    "SQS_HEARTBEAT_SEC": "60",
    "RECEIVE_WAIT_TIME_SEC": "10",
    "FRAME_STRIDE_DEFAULT": "3",
    "YOLO_IMGSZ": "640",
    "YOLO_CONF": "0.35",
    "YOLO_IOU": "0.45",
    "SNAPSHOT_MARGIN": "0.15",
    "SNAPSHOT_NEIGHBOR_IOU": "0.25",
    "SNAPSHOT_SIZE": "640",
    "JPG_QUALITY": "95",
}
YOLO_VEHICLE_TYPES: Tuple[str, ...] = (
    "Car","SUV","Van","LightTruck","Utility","Motorcycle","CarouselBus","E-Jeepney"
)
def s3_uri(key: str) -> str:
    return f"s3://{CONFIG['S3_BUCKET']}/{key}"
BUNDLES = {
    "YOLO_WEIGHTS": "video_analysis/yolo_worker/bundle/yolov8m.pt",
}
