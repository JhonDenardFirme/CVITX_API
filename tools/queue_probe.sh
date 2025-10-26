#!/usr/bin/env bash
set -euo pipefail
: "${SQS_ANALYSIS_BASELINE_URL:?}"; : "${SQS_ANALYSIS_CMT_URL:?}"
aws sqs get-queue-attributes --queue-url "$SQS_ANALYSIS_BASELINE_URL" --attribute-names All | jq
aws sqs get-queue-attributes --queue-url "$SQS_ANALYSIS_CMT_URL"      --attribute-names All | jq
