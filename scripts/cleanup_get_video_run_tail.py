#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
target = ROOT / "video_analysis" / "worker_utils" / "common.py"

text = target.read_text(encoding="utf-8")

# We know the stray block starts at this line:
marker_start = "\n(workspace_id: str, video_id: str, variant: str) -> Optional[Dict[str, Any]]:"
# And ends right before this next function:
marker_end = "\ndef set_video_expected("

start = text.find(marker_start)
end = text.find(marker_end)

if start == -1 or end == -1 or end <= start:
    print("[cleanup] ERROR: markers not found; not modifying file.", file=sys.stderr)
    sys.exit(1)

new_text = text[:start] + text[end:]

target.write_text(new_text, encoding="utf-8")
print("[cleanup] Removed duplicate get_video_run tail from", target)
