# /home/ubuntu/cvitx/workers/snapshot_worker.py
import json, os
import boto3

REGION = os.getenv("AWS_REGION", "ap-southeast-2")
QUEUE_URL = os.getenv("SQS_SNAPSHOT_QUEUE_URL")
sqs = boto3.client("sqs", region_name=REGION)

def main():
    assert QUEUE_URL, "SQS_SNAPSHOT_QUEUE_URL not set"
    print(f"[snapshot-worker] polling {QUEUE_URL}")
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,   # long poll
            VisibilityTimeout=60
        )
        for m in resp.get("Messages", []):
            body = json.loads(m["Body"])
            print("[snapshot-worker] got message:", json.dumps(body, indent=2))
            # TODO: do real snapshot work here
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=m["ReceiptHandle"])
            print("[snapshot-worker] deleted message", m["MessageId"])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("bye")
