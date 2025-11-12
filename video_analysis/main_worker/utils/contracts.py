# contracts.py — maps SNAPSHOT_READY → engine I/O (variant='main')
# Fill in mappings if not already present in your project.
from typing import Any, Dict

def snapshot_ready_to_engine_inputs(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a SNAPSHOT_READY SQS message into engine.run() kwargs.
    Expected keys (example): input_image_s3_uri, workspace_id, analysis_id, etc.
    """
    return {
        "variant": "main",
        "input_image_s3_uri": msg.get("input_image_s3_uri"),
        "workspace_id": msg.get("workspace_id"),
        "analysis_id": msg.get("analysis_id"),
        # Add other passthroughs if your engine.run supports them:
        # "presign": msg.get("presign", False),
        # "ttl": msg.get("ttl", 900),
    }
