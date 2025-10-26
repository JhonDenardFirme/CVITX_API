#!/usr/bin/env bash
set -euo pipefail
: "${SQS_ANALYSIS_BASELINE_URL:?}"
BL_DLQ_URL="$(aws sqs list-queues --queue-name-prefix cvitx-analysis-baseline-dlq --query 'QueueUrls[0]' --output text 2>/dev/null || true)"
if [ -z "$BL_DLQ_URL" ] || [ "$BL_DLQ_URL" = "None" ]; then
  echo "DLQ not available (IAM or provisioning). Skipping poison test."
  exit 0
fi
aws sqs send-message --queue-url "$SQS_ANALYSIS_BASELINE_URL" --message-body '{"event":"BAD","foo":1}' >/dev/null
echo "Poisoned 1 message. Forcing receives..."
for i in 1 2 3; do
  aws sqs receive-message --queue-url "$SQS_ANALYSIS_BASELINE_URL" --wait-time-seconds 1 --visibility-timeout 1 >/dev/null || true
  sleep 2
done
echo "Baseline queue:"; aws sqs get-queue-attributes --queue-url "$SQS_ANALYSIS_BASELINE_URL" --attribute-names All | jq '.Attributes|{Visible:.ApproximateNumberOfMessages,NotVisible:.ApproximateNumberOfMessagesNotVisible}'
echo "DLQ visible:"; aws sqs get-queue-attributes --queue-url "$BL_DLQ_URL" --attribute-names ApproximateNumberOfMessages | jq
