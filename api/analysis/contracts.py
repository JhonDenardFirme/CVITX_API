import json
from typing import Any, Dict

class ContractError(Exception):
    pass

def parse_analyze_image_message(body: str) -> Dict[str, Any]:
    try:
        d = json.loads(body)
    except Exception as e:
        raise ContractError(f"Invalid JSON: {e}")
    # Accept either 'input_image_s3_uri' or legacy 's3_uri'
    if "input_image_s3_uri" not in d and "s3_uri" in d:
        d["input_image_s3_uri"] = d["s3_uri"]
    required = ["event","analysis_id","workspace_id","input_image_s3_uri","model_variant"]
    missing = [k for k in required if k not in d]
    if missing:
        raise ContractError(f"Missing fields: {missing}")
    if d["event"] != "ANALYZE_IMAGE":
        raise ContractError(f"Unexpected event: {d['event']}")
    return d
