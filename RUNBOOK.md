# CVITX GPU Instance – Ops Notes

## Instance
Type: g4dn.xlarge (T4)  
Region: ap-southeast-2  
Public IP: <PUT CURRENT PUBLIC IP HERE>  
Private IP: <PUT PRIVATE IP HERE>  
User: ubuntu  
Key pair: <your-key.pem>  
IAM role: cvitx-dev-ec2-role  
Security group: cvitx-dev-ec2-sg

## Paths
Repo root: /home/ubuntu/cvitx  
API: /home/ubuntu/cvitx/api  
Worker: /home/ubuntu/cvitx/workers/yolo_worker.py  
Venv: /home/ubuntu/cvitx/api/.venv

## AWS
Bucket: cvitx-uploads-dev-jdfirme  
Video queue: https://sqs.ap-southeast-2.amazonaws.com/118730128890/cvitx-video-tasks  
Snapshot queue: https://sqs.ap-southeast-2.amazonaws.com/118730128890/cvitx-snapshot-tasks

## YOLO settings
YOLO_DEVICE=cuda:0
YOLO_WEIGHTS=<path or yolov8n.pt>
YOLO_CONF_THRES=0.20
YOLO_IMG_SIZE=640

## Notes
- Use /mnt/nvme/tmp for TMP_DIR if available
- Worker can run with `--poll` or single payload file
