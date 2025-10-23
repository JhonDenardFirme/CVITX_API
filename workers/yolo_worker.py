# DEPRECATED: dev-only wrapper that delegates to the modular pipeline.
import json, os, sys
print("[yolo_worker.py] DEPRECATED / DEV-ONLY. Delegating to workers.video_runner...", file=sys.stderr)
try:
    from workers import video_runner as vr
except Exception as e:
    print("ERR: unable to import workers.video_runner:", e, file=sys.stderr)
    sys.exit(2)

def main():
    # Accept a single JSON payload via stdin OR env PAYLOAD_JSON
    payload_json = os.environ.get("PAYLOAD_JSON")
    if not payload_json:
        payload_json = sys.stdin.read() or ""
    if not payload_json.strip():
        print("Usage: echo '{\"event\":\"PROCESS_VIDEO\",...}' | python workers/yolo_worker.py", file=sys.stderr)
        sys.exit(1)
    try:
        payload = json.loads(payload_json)
    except Exception as e:
        print("ERR: invalid JSON:", e, file=sys.stderr); sys.exit(1)
    vr.handle_process_video(payload)

if __name__ == "__main__":
    main()
