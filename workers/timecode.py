from datetime import timedelta
from dateutil import parser as dateparser

def ms_from_frame(frame_idx: int, fps: float) -> int:
    return int((frame_idx / max(1e-6, fps)) * 1000)

def iso_add_ms(recorded_iso: str | None, ms: int) -> str | None:
    if not recorded_iso: return None
    dt = dateparser.isoparse(recorded_iso)
    return (dt + timedelta(milliseconds=ms)).isoformat()
