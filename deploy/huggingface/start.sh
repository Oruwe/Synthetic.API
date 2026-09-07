#!/usr/bin/env bash
# Supervises this single container's processes, in place of docker-compose's
# separate services -- see this directory's Dockerfile/SETUP.md for
# why. Runs demo_target and agents-synthesizer in the background, then execs
# the merged FastAPI+Gradio app (deploy/huggingface/merged_app.py) in the
# foreground, since Docker treats whichever process is PID 1 as the
# container's lifecycle -- that has to be the app actually serving traffic,
# not a background fixture.
set -euo pipefail
cd /app

# HF Spaces sets $PORT itself; 7860 is both its and this Space's own
# app_port default (see SETUP.md's frontmatter) if it's ever unset.
export PORT="${PORT:-7860}"

# Self-referential -- ui/app.py (mounted by merged_app.py onto this same
# process) calls the Orchestrator's own /trigger, /runs/{id} etc. over a
# real HTTP request back into this same container, on this same port. See
# merged_app.py's module docstring for why this stays a real HTTP call
# rather than an in-process shortcut.
export ORCHESTRATOR_URL="http://localhost:${PORT}"

# Ephemeral on HF Spaces' free tier (no persistent disk) -- fine here, see
# SETUP.md's "what's ephemeral, and why that's OK" section.
export RUN_STORE_DIR="${RUN_STORE_DIR:-/app/data/runs}"
export SCREENSHOT_DIR="${SCREENSHOT_DIR:-/app/data/screenshots}"
mkdir -p "$RUN_STORE_DIR" "$SCREENSHOT_DIR"

# Both explicitly disabled for this deployment -- see Dockerfile's own
# comment on why (both already fail open / degrade gracefully by design;
# neither is worth running in a single, non-scaled container).
export LANGFUSE_ENABLED="false"
export REDIS_URL=""

echo "[start.sh] launching demo_target (fixture pages for the ambient RPA / human-in-the-loop demo) on :5050"
uv run python demo_target/app.py &

echo "[start.sh] launching agents-synthesizer (background poll loop)"
uv run python -m agents.synthesizer.main &

echo "[start.sh] launching the merged Orchestrator+UI app on :${PORT}"
exec uv run uvicorn deploy.huggingface.merged_app:app --host 0.0.0.0 --port "${PORT}"
