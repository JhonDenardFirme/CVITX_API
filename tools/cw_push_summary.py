#!/usr/bin/env python3
import argparse, sys, re, time, json, boto3, os

# Extract key=value pairs from a [summary] line
def parse_summary_line(line):
    if "[summary]" not in line: return None
    # tolerate JSON mode too
    if line.strip().startswith("{"):
        try:
            obj = json.loads(line)
            if obj.get("type") == "summary":
                return obj
        except Exception:
            return None
    # key=value pairs
    m = {}
    for kv in re.findall(r"(\b[a-zA-Z0-9_]+)=([^\s]+)", line):
        k,v = kv
        # normalize numerics if possible
        try:
            if "." in v: m[k]=float(v)
            else: m[k]=int(v)
        except:
            m[k]=v
    if m: return m
    return None

def put_metrics(cw, ns, dims, metrics):
    # metrics is dict name->float
    data=[]
    for name,val in metrics.items():
        if not isinstance(val,(int,float)): continue
        data.append({
            "MetricName": name,
            "Dimensions": [{"Name":k,"Value":v} for k,v in dims.items() if v],
            "Unit": "None",
            "Value": float(val)
        })
    # CloudWatch allows max 20 per call
    for i in range(0, len(data), 20):
        cw.put_metric_data(Namespace=ns, MetricData=data[i:i+20])

def main():
    ap = argparse.ArgumentParser(description="Push CVITX worker summary metrics to CloudWatch")
    ap.add_argument("--file", help="Path to log file (reads last summary line). Omit to read from stdin.")
    ap.add_argument("--namespace", default=os.getenv("CW_NAMESPACE","cvitx"))
    ap.add_argument("--dim-app", default=os.getenv("CW_DIM_APP","worker_video"))
    ap.add_argument("--dim-env", default=os.getenv("CW_DIM_ENV","dev"))
    ap.add_argument("--dim-workspace", default=os.getenv("WORKSPACE_ID",""))
    ap.add_argument("--dim-video", default=os.getenv("VIDEO_ID",""))
    ap.add_argument("--region", default=os.getenv("AWS_REGION","ap-southeast-2"))
    args = ap.parse_args()

    lines=[]
    if args.file:
        with open(args.file,"r",encoding="utf-8",errors="ignore") as f:
            lines=f.readlines()
    else:
        lines=sys.stdin.readlines()

    summary=None
    for ln in reversed(lines):
        s = parse_summary_line(ln)
        if s: summary=s; break

    if not summary:
        print("ERR: no summary line found", file=sys.stderr)
        sys.exit(2)

    # Map likely keys → metric names (keep names simple)
    metrics={}
    # Accept both total_sec or total_seconds
    metrics["total_seconds"]  = float(summary.get("total_sec") or summary.get("total_seconds") or 0.0)
    metrics["pass1_seconds"]  = float(summary.get("pass1_sec") or 0.0)
    metrics["pass2_seconds"]  = float(summary.get("pass2_sec") or 0.0)
    metrics["frames"]         = float(summary.get("frames") or 0.0)
    metrics["iters"]          = float(summary.get("iters") or summary.get("iters_total") or 0.0)
    metrics["dets_total"]     = float(summary.get("dets_total") or summary.get("dets") or 0.0)
    metrics["tracks"]         = float(summary.get("tracks") or 0.0)
    metrics["ready"]          = float(summary.get("ready") or 0.0)
    metrics["emitted"]        = float(summary.get("emitted") or 0.0)

    dims = {
        "App": args.dim_app,
        "Env": args.dim_env,
        "WorkspaceId": args.dim_workspace,
        "VideoId": args.dim_video
    }

    cw = boto3.client("cloudwatch", region_name=args.region)
    put_metrics(cw, args.namespace, dims, metrics)

    print("OK: pushed metrics:", json.dumps({"namespace":args.namespace,"dimensions":dims,"metrics":metrics}))
    sys.exit(0)

if __name__ == "__main__":
    main()
