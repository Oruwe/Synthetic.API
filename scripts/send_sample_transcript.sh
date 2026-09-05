#!/usr/bin/env bash
# Triggers the orchestrator with a sample transcript, as if it came from
# Omi. Use this for local dev/demo instead of a real Omi webhook payload.
set -euo pipefail

ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://localhost:8000}"
TRANSCRIPT="${1:-Check the shipping portal for delayed orders and update the team}"

echo "POST ${ORCHESTRATOR_URL}/trigger"
echo "transcript: ${TRANSCRIPT}"
echo

response=$(curl -sS -X POST "${ORCHESTRATOR_URL}/trigger" \
  -H "Content-Type: application/json" \
  -d "{\"transcript\": $(printf '%s' "$TRANSCRIPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}")

echo "$response"
run_id=$(echo "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')

echo
echo "Run ID: ${run_id}"
echo "Poll status with:"
echo "  curl -s ${ORCHESTRATOR_URL}/runs/${run_id} | python3 -m json.tool"
