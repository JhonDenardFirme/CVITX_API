#!/usr/bin/env bash
set -euo pipefail

echo "============ CVITX VIDEO/RUN PROBE ============"

# ─────────────────────────────────────────────
# 0. Load env and set core variables
# ─────────────────────────────────────────────

cd /home/ubuntu/cvitx/api

if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo "✖ .env not found in /home/ubuntu/cvitx/api" >&2
  exit 1
fi

# Prefer DB_URL, fallback to DATABASE_URL
DB_URL_EFFECTIVE="${DB_URL:-${DATABASE_URL:-}}"

if [ -z "$DB_URL_EFFECTIVE" ]; then
  echo "✖ DB_URL / DATABASE_URL not set in env" >&2
  exit 1
fi

# Convert SQLAlchemy URL -> psql-compatible URL (strip +psycopg2)
DB_PSQL_URL="${DB_URL_EFFECTIVE/postgresql+psycopg2/postgresql}"

API_BASE="${API_BASE:-http://13.55.23.204}"
API_KEY="${API_KEY:-supersecret}"

WORKSPACE_ID="d3371b46-03c3-4c54-ab78-f96f66eb10da"  # CTX1008
WORKSPACE_CODE="CTX1008"

VIDEO_ID="da5c16f2-67e2-43cc-82fc-e0c42a0c46b2"
VARIANT="cmt"

AWS_REGION="${AWS_REGION:-ap-southeast-2}"

echo "API_BASE      = $API_BASE"
echo "WORKSPACE_ID  = $WORKSPACE_ID"
echo "VIDEO_ID      = $VIDEO_ID"
echo "AWS_REGION    = $AWS_REGION"
echo "DB_URL        = $DB_URL_EFFECTIVE"
echo "DB_PSQL_URL   = $DB_PSQL_URL"
echo ""

# ─────────────────────────────────────────────
# 1. DB PROBE — videos & video_analyses
# ─────────────────────────────────────────────

echo "============ DB PROBE: videos & video_analyses for VIDEO_ID = $VIDEO_ID ============"

psql "$DB_PSQL_URL" <<SQL
-- 1) Video row status
SELECT
  id,
  workspace_id,
  status,
  error_msg,
  s3_key_raw,
  recorded_at,
  frame_stride,
  processing_started_at,
  processing_finished_at,
  updated_at
FROM videos
WHERE id = '$VIDEO_ID';

-- 2) Any analysis/run rows for this video
SELECT
  id,
  video_id,
  variant,
  status,
  created_at,
  updated_at,
  error_msg
FROM video_analyses
WHERE video_id = '$VIDEO_ID'
ORDER BY created_at DESC
LIMIT 20;

-- 3) Latest analyses overall (sanity check)
SELECT
  id,
  video_id,
  variant,
  status,
  created_at,
  updated_at
FROM video_analyses
ORDER BY created_at DESC
LIMIT 20;
SQL

echo ""

# ─────────────────────────────────────────────
# 2. API PROBE — video detail, analyses, progress
# ─────────────────────────────────────────────

echo "============ API PROBE: /videos detail & analyses & progress ============"

echo "[API] GET /workspaces/{wid}/videos/{vid}"
curl -sS -H "X-API-Key: $API_KEY" \
  "$API_BASE/workspaces/$WORKSPACE_ID/videos/$VIDEO_ID" \
  | python -m json.tool
echo ""

echo "[API] GET /workspaces/{wid}/videos/{vid}/analyses"
curl -sS -H "X-API-Key: $API_KEY" \
  "$API_BASE/workspaces/$WORKSPACE_ID/videos/$VIDEO_ID/analyses" \
  | python -m json.tool
echo ""

echo "[API] GET /workspaces/{wid}/videos/{vid}/progress?variant=$VARIANT"
curl -sS -H "X-API-Key: $API_KEY" \
  "$API_BASE/workspaces/$WORKSPACE_ID/videos/$VIDEO_ID/progress?variant=$VARIANT" \
  | python -m json.tool || echo "⚠ progress returned non-2xx (expected if no run yet)"
echo ""

# ─────────────────────────────────────────────
# 3. AWS / SQS PROBE — identity & queues
# ─────────────────────────────────────────────

echo "============ AWS PROBE: identity & SQS queues ============"

aws sts get-caller-identity --region "$AWS_REGION" || echo "⚠ aws sts get-caller-identity failed"
echo ""

echo "Queues matching 'cvitx' (for reference):"
aws sqs list-queues --region "$AWS_REGION" | tr '"' '\n' | grep cvitx || echo "No cvitx queues listed"
echo ""

# TODO: set these to real URLs after inspecting list-queues or .env
VIDEO_QUEUE_URL="${VIDEO_QUEUE_URL:-}"
VIDEO_DLQ_URL="${VIDEO_DLQ_URL:-}"

echo "VIDEO_QUEUE_URL = ${VIDEO_QUEUE_URL:-<not set>}"
echo "VIDEO_DLQ_URL   = ${VIDEO_DLQ_URL:-<not set>}"
echo ""

if [ -n "$VIDEO_QUEUE_URL" ]; then
  echo "Queue attributes for VIDEO_QUEUE_URL:"
  aws sqs get-queue-attributes \
    --queue-url "$VIDEO_QUEUE_URL" \
    --attribute-names All \
    --region "$AWS_REGION" \
    | python -m json.tool || echo "⚠ get-queue-attributes failed"
  echo ""

  echo "Sample messages (non-destructive peek):"
  aws sqs receive-message \
    --queue-url "$VIDEO_QUEUE_URL" \
    --max-number-of-messages 5 \
    --visibility-timeout 0 \
    --wait-time-seconds 2 \
    --region "$AWS_REGION" \
    | python -m json.tool || echo "No visible messages at this time"
  echo ""
fi

if [ -n "$VIDEO_DLQ_URL" ]; then
  echo "DLQ attributes for VIDEO_DLQ_URL:"
  aws sqs get-queue-attributes \
    --queue-url "$VIDEO_DLQ_URL" \
    --attribute-names All \
    --region "$AWS_REGION" \
    | python -m json.tool || echo "⚠ get-queue-attributes (DLQ) failed"
  echo ""
fi

# ─────────────────────────────────────────────
# 4. SYSTEMD & LOGS — API + workers
# ─────────────────────────────────────────────

echo "============ SYSTEMD PROBE: list cvitx services ============"

systemctl list-units --type=service | grep -i cvitx || echo "No cvitx services found"
echo ""

echo "API service status:"
sudo systemctl status cvitx-api.service --no-pager || echo "⚠ cvitx-api.service status failed"
echo ""

# TODO: set this to the actual video worker unit, e.g. cvitx-video-worker.service
VIDEO_WORKER_UNIT="${VIDEO_WORKER_UNIT:-cvitx-video-worker.service}"

echo "Video worker service status: $VIDEO_WORKER_UNIT"
sudo systemctl status "$VIDEO_WORKER_UNIT" --no-pager || echo "⚠ Check worker unit name"
echo ""

echo "Recent API logs (last 30 minutes):"
sudo journalctl -u cvitx-api.service --since "30 minutes ago" --no-pager | tail -n 200 || echo "⚠ journalctl for api failed"
echo ""

echo "Recent video worker logs (last 30 minutes):"
sudo journalctl -u "$VIDEO_WORKER_UNIT" --since "30 minutes ago" --no-pager | tail -n 200 || echo "⚠ journalctl for worker failed"
echo ""

echo "============ PROBE COMPLETE ============"
