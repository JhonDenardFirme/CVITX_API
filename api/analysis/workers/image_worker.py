import os, sys, time, json, argparse, traceback
import boto3
from sqlalchemy import create_engine, text

from analysis.contracts import parse_analyze_image_message, ContractError
from analysis.models.builder import get_model
from analysis.models.postprocess import annotate
from analysis.workers.utils import s3_get_bytes, s3_put_bytes, S3_BUCKET, S3_PREFIX

DB_URL = os.environ["DB_URL"].replace("postgresql+psycopg2","postgresql")
AWS_REGION = os.getenv("AWS_REGION","ap-southeast-2")

def upsert_result(engine, payload, variant: str, annotated_key: str, dets, latency_ms: int, gflops: float):
    with engine.begin() as conn:
        conn.execute(text("""
          INSERT INTO image_analysis_results (
            id, analysis_id, workspace_id, model_variant,
            type, type_conf, make, make_conf, model, model_conf,
            parts, colors, plate_text, annotated_image_s3_key,
            latency_ms, gflops, status, error_msg
          )
          VALUES (gen_random_uuid(), :aid, :wid, :variant,
            :type,:type_conf,:make,:make_conf,:model,:model_conf,
            NULL::jsonb, NULL::jsonb, :plate_text, :ann_key,
            :latency_ms, :gflops, 'ready', NULL
          )
          ON CONFLICT (analysis_id, model_variant) DO UPDATE SET
            type=EXCLUDED.type,
            type_conf=EXCLUDED.type_conf,
            make=EXCLUDED.make,
            make_conf=EXCLUDED.make_conf,
            model=EXCLUDED.model,
            model_conf=EXCLUDED.model_conf,
            parts=EXCLUDED.parts,
            colors=EXCLUDED.colors,
            plate_text=EXCLUDED.plate_text,
            annotated_image_s3_key=EXCLUDED.annotated_image_s3_key,
            latency_ms=EXCLUDED.latency_ms,
            gflops=EXCLUDED.gflops,
            status=EXCLUDED.status,
            error_msg=NULL;
        """), dict(
            aid=payload["analysis_id"], wid=payload["workspace_id"], variant=variant,
            type=dets.get("type"), type_conf=dets.get("type_conf"),
            make=dets.get("make"), make_conf=dets.get("make_conf"),
            model=dets.get("model"), model_conf=dets.get("model_conf"),
            parts=json.dumps(dets.get("parts") or []),
            colors=json.dumps(dets.get("colors") or []),
            plate_text=dets.get("plate_text"),
            ann_key=annotated_key, latency_ms=latency_ms, gflops=gflops,
        ))
        # Update parent to 'done' only when both variants exist (simple check)
        r = conn.execute(text("""
          SELECT COUNT(*) FROM image_analysis_results
          WHERE analysis_id=:aid
        """), {"aid": payload["analysis_id"]}).scalar_one()
        new_status = 'done' if r >= 2 else 'processing'
        conn.execute(text("UPDATE image_analyses SET status=:st WHERE id=:aid"),
                     {"st": new_status, "aid": payload["analysis_id"]})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["baseline","cmt"])
    args = ap.parse_args()
    variant = args.variant

    queue_url = os.getenv("SQS_ANALYSIS_BASELINE_URL") if variant=="baseline" \
                else os.getenv("SQS_ANALYSIS_CMT_URL")
    if not queue_url:
        print(f"[FATAL] Queue URL for {variant} missing in env.", file=sys.stderr)
        sys.exit(2)

    engine = create_engine(DB_URL, pool_pre_ping=True)
    sqs = boto3.client("sqs", region_name=AWS_REGION)

    model = get_model(variant)
    print(f"[startup] worker={variant} queue={queue_url}")

    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=60
            )
            msgs = resp.get("Messages", [])
            if not msgs:
                continue
            for m in msgs:
                receipt = m["ReceiptHandle"]
                body = m.get("Body","")
                try:
                    payload = parse_analyze_image_message(body)
                    t0 = time.perf_counter()
                    img_bytes = s3_get_bytes(payload["input_image_s3_uri"])
                    dets = model.infer(img_bytes)
                    latency_ms = int((time.perf_counter() - t0) * 1000)

                    ann_bytes = annotate(img_bytes, dets)
                    ann_key = f"{S3_PREFIX}/{payload['workspace_id']}/{payload['analysis_no']}/{variant}/annotated.jpg"
                    s3_put_bytes(S3_BUCKET, ann_key, ann_bytes, "image/jpeg")

                    upsert_result(engine, payload, variant, ann_key, dets, latency_ms, model.report_gflops())

                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    print(f"[ok] {variant} analysis_id={payload['analysis_id']} latency_ms={latency_ms}")
                except ContractError as e:
                    print(f"[drop] bad contract: {e}")
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                except Exception as e:
                    print(f"[err] {variant} exception: {e}\n{traceback.format_exc()}", file=sys.stderr)
                    # leave message for retry (visibility timeout)
        except Exception as outer:
            print(f"[loop-err] {outer}", file=sys.stderr)
            time.sleep(2)

if __name__ == "__main__":
    main()
