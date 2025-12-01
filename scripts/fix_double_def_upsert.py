#!/usr/bin/env python
import pathlib

target = pathlib.Path("video_analysis/worker_utils/common.py")

text = target.read_text(encoding="utf-8")

old = "def upsert_video_detection_and_progress(\ndef upsert_video_detection_and_progress(\n"
new = "def upsert_video_detection_and_progress(\n"

if old not in text:
    print("Pattern not found; no changes made.")
else:
    text_new = text.replace(old, new, 1)
    target.write_text(text_new, encoding="utf-8")
    print("Fixed duplicate def in:", target)
