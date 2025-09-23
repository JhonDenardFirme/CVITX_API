import os, json, time
import boto3

QUEUE_URL = os.getenv("SQS_SNAPSHOT_QUEUE_URL")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
assert QUEUE_URL, "SQS_SNAPSHOT_QUEUE_URL not set"

sqs = boto3.client("sqs", region_name=AWS_REGION)

def _safe_load_json(s: str):
    try:
        return json.loads(s), None
    except Exception as e:
        return None, e

def main():
    print(f"[snapshot-worker] polling {QUEUE_URL}")
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=10,
            VisibilityTimeout=60,
        )

        for m in resp.get("Messages", []):
            body, err = _safe_load_json(m.get("Body", ""))
            if err:
                print("[snapshot-worker] non-JSON message; deleting:", err)
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=m["ReceiptHandle"])
                continue

            if body.get("event") != "SNAPSHOT_READY":
                print("[snapshot-worker] ignoring message with event:", body.get("event"))
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=m["ReceiptHandle"])
                continue

            print("[snapshot-worker] got message:", json.dumps(body, indent=2))
            # TODO: real snapshot work goes here
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=m["ReceiptHandle"])
            print("[snapshot-worker] deleted message", m.get("MessageId"))

        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("bye")

