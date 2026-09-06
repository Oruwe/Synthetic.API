#!/usr/bin/env bash
# Triggers the orchestrator with a sample transcript, as if it came from
# Omi. Use this for local dev/demo instead of a real Omi webhook payload.
#
# Deliberately has NO dependency on python/python3/jq being installed and
# on PATH -- this needs to run in plain bash across Linux, macOS, and
# Windows Git Bash alike. On Windows in particular, `python3` (and often
# bare `python`) resolves to a Microsoft Store app-execution-alias stub
# that prints a message to stderr and produces no stdout, which used to
# make this script silently POST a malformed JSON body.
set -euo pipefail

ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://localhost:8000}"
TRANSCRIPT="${1:-Check the shipping portal for delayed orders and update the team}"

echo "POST ${ORCHESTRATOR_URL}/trigger"
echo "transcript: ${TRANSCRIPT}"
echo

# Minimal JSON string escaping in pure sed: backslashes first (so we don't
# double-escape the quote-escaping backslashes we add next), then double
# quotes. Covers the realistic case (a plain sentence); doesn't attempt to
# handle embedded control characters, which a CLI-arg transcript won't have.
escaped_transcript=$(printf '%s' "$TRANSCRIPT" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')

response=$(curl -sS -X POST "${ORCHESTRATOR_URL}/trigger" \
  -H "Content-Type: application/json" \
  -d "{\"transcript\": \"${escaped_transcript}\"}")

echo "$response"

# Pull run_id out of {"run_id": "...", "status": "..."} without needing a
# JSON parser on PATH -- the response shape is fixed and simple enough
# that a regex extraction is reliable here.
run_id=$(printf '%s' "$response" | sed -n 's/.*"run_id" *: *"\([^"]*\)".*/\1/p')

if [ -z "$run_id" ]; then
  echo
  echo "Could not find a run_id in the response above -- the trigger likely failed (see the raw response)."
  exit 1
fi

echo
echo "Run ID: ${run_id}"
echo "Poll status with:"
echo "  curl -s ${ORCHESTRATOR_URL}/runs/${run_id}"
