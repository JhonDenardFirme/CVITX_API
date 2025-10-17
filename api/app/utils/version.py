import subprocess

def git_sha_short(default="unknown"):
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, timeout=1)
        return out.decode().strip()
    except Exception:
        return default
