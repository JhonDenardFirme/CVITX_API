#!/usr/bin/env bash
set -euo pipefail
: "${API_BASE:?}"; : "${API_TOKEN:?}"
curl -sS -H "Authorization: Bearer $API_TOKEN" "$API_BASE/healthz" | jq
