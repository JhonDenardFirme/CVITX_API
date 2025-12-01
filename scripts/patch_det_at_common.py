from pathlib import Path

path = Path("video_analysis/worker_utils/common.py")
text = path.read_text()

# -------------------------------------------------------------------
# Patch 1: Remove dead detected_at normalization in upsert_results(...)
# -------------------------------------------------------------------
block = (
"    # Normalize detected_at (accept ISO strings or datetime/None)\n"
"    det_at: Optional[datetime]\n"
"    if isinstance(detected_at, str):\n"
"        det_at = _parse_iso(detected_at)\n"
"    else:\n"
"        det_at = detected_at\n"
"\n"
)

if block in text:
    text = text.replace(block, "", 1)
    print("[patch] Removed dead detected_at block in upsert_results().")
else:
    print("[patch] WARN: normalize block not found in upsert_results(); maybe already patched.")

# -------------------------------------------------------------------
# Patch 2: Insert detected_at normalization in upsert_video_detection_and_progress(...)
# -------------------------------------------------------------------
marker_func = "def upsert_video_detection_and_progress(\n"
idx = text.find(marker_func)
if idx == -1:
    raise SystemExit("[patch] ERROR: upsert_video_detection_and_progress() not found.")

sub = text[idx:]

marker_inner = '    type_obj = result.get("type") or {}\n'
inner_idx = sub.find(marker_inner)
if inner_idx == -1:
    raise SystemExit("[patch] ERROR: inner marker in upsert_video_detection_and_progress() not found.")

norm_block = (
"    # Normalize detected_at (accept ISO strings or datetime/None)\n"
"    det_at: Optional[datetime]\n"
"    if isinstance(detected_at, str):\n"
"        det_at = _parse_iso(detected_at)\n"
"    else:\n"
"        det_at = detected_at\n"
"\n"
)

sub = sub.replace(marker_inner, norm_block + marker_inner, 1)
text = text[:idx] + sub

path.write_text(text)
print("[patch] Inserted detected_at normalization in upsert_video_detection_and_progress().")
