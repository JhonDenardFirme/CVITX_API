from typing import Any, Dict, Optional
class log:
    @staticmethod
    def info(msg: str): pass
    @staticmethod
    def error(msg: str, exc_info: bool=False): pass
def validate_process_video(p: Dict[str,Any]) -> Dict[str,Any]: return p
def validate_process_video_db(p: Dict[str,Any]) -> Dict[str,Any]: return p
def validate_snapshot_ready(p: Dict[str,Any]) -> Dict[str,Any]: return p
def get_s3(): 
    class _S3: 
        def download_file(self, b,k,p): pass
        def put_object(self, Bucket, Key, Body, ContentType): pass
    return _S3()
def get_sqs():
    class _SQS:
        def send_message(self, QueueUrl, MessageBody): pass
        def receive_message(self, **kw): return {"Messages": []}
        def delete_message(self, **kw): pass
        def change_message_visibility(self, **kw): pass
    return _SQS()
def get_video_by_id(video_id: str) -> Dict[str,Any]:
    return {"workspace_code":"CTX1001","camera_code":"CAM1","s3_key_raw":"demo_user/w/v/raw.mp4","frame_stride":3,"recorded_at":None}
def build_snapshot_key(workspace_id: str, video_id: str, workspace_code: str, camera_code: str, track_id: int, offset_ms: int) -> str:
    return f"demo_user/{workspace_id}/{video_id}/snapshots/{workspace_code}_{camera_code}_{track_id:06d}_{offset_ms:06d}.jpg"
def crop_with_margin(frame, box, margin: float, neighbor_iou: float):
    return frame  # stub
def letterbox_to_square(img, size: int):
    return img  # stub
def encode_jpeg(img, quality: int) -> bytes:
    return b"JPEG"  # stub
def ms_from_frame(idx: int, fps: float) -> int:
    return int(round(1000.0 * (idx / max(1.0, fps))))
def detected_at(recorded_at_iso: Optional[str], offset_ms: int) -> Optional[str]:
    return recorded_at_iso
