#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DBURL=$(sed -n 's/^DB_URL=//p' .env); PGURL=$(echo "$DBURL" | sed 's#postgresql+psycopg2:#postgresql:#')
APIKEY=$(sed -n 's/^API_KEY=//p' .env)
VID=$(psql "$PGURL" -At -c "SELECT id FROM videos ORDER BY updated_at DESC NULLS LAST LIMIT 1;")
WS=$(psql "$PGURL" -At -c "SELECT workspace_id FROM detections ORDER BY detected_at DESC LIMIT 1;")
echo "VID=$VID"; echo "WS=$WS"
echo "== status =="; curl -s -H "X-API-Key: $APIKEY" "http://127.0.0.1/api/videos/$VID/status" | jq .
echo "== export (WS) =="; curl -s -H "X-API-Key: $APIKEY" \
 "http://127.0.0.1/api/exports/detections.csv?workspace_id=$WS" -o /tmp/detections.csv; head -n 2 /tmp/detections.csv
echo "== export (WS+VID) =="; curl -s -H "X-API-Key: $APIKEY" \
 "http://127.0.0.1/api/exports/detections.csv?workspace_id=$WS&video_id=$VID" -o /tmp/det_by_vid.csv; head -n 3 /tmp/det_by_vid.csv
