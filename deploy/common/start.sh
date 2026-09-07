#!/usr/bin/env bash
# Supervises this single container's processes, in place of docker-compose's
# separate services -- shared by every single-container deployment target
# (deploy/huggingface/, deploy/render/ -- see each target's own SETUP.md for
# the platform-specific parts). Runs demo_target and agents-synthesizer in
# the background, then execs the merged FastAPI+Gradio app
# (deploy/common/merged_app.py) in the foreground, since Docker treats
# whichever process is PID 1 as the container's lifecycle -- that has to be
# the app actually serving traffic, not a background fixture.
set -euo pipefail
cd /app

# Both HF Spaces and Render set $PORT themselves; 7860 is the fallback
# default (and HF's own app_port frontmatter default) if it's ever unset.
export PORT="${PORT:-7860}"

# Self-referential -- ui/app.py (mounted by merged_app.py onto this same
# process) calls the Orchestrator's own /trigger, /runs/{id} etc. over a
# real HTTP request back into this same container, on this same port. See
# merged_app.py's module docstring for why this stays a real HTTP call
# rather than an in-process shortcut.
export ORCHESTRATOR_URL="http://localhost:${PORT}"

# Ephemeral on every free-tier target this is used with (no persistent
# disk) -- fine here, see each target's SETUP.md "what's ephemeral, and
# why that's OK" section.
export RUN_STORE_DIR="${RUN_STORE_DIR:-/app/data/runs}"
export SCREENSHOT_DIR="${SCREENSHOT_DIR:-/app/data/screenshots}"
mkdir -p "$RUN_STORE_DIR" "$SCREENSHOT_DIR"

# Both explicitly disabled for this deployment -- see each target's
# Dockerfile for why (both already fail open / degrade gracefully by
# design; neither is worth running in a single, non-scaled container).
export LANGFUSE_ENABLED="false"
export REDIS_URL=""

# Loopback-only, deliberately -- demo_target only ever needs to be reached
# from WITHIN this container (the Playwright browser the ambient-RPA/
# human-in-the-loop demo launches, running in this same process/container).
# Binding 0.0.0.0 here (demo_target/app.py's own default, needed instead
# for docker-compose, where it's a separate container reached over the
# docker network) was a real, live bug on Render specifically: Render's
# port-autodetection scans for the first open listening socket in the
# container and treats it as the PUBLIC one, and this tiny Flask app opens
# its port almost immediately -- long before the merged Orchestrator+UI
# app below finishes importing everything and binds its own -- so Render
# was routing all public traffic to this fixture instead of the real
# product. See demo_target/app.py's own comment on this.
export DEMO_TARGET_HOST="127.0.0.1"

echo "[start.sh] launching demo_target (fixture pages for the ambient RPA / human-in-the-loop demo) on :5050"
uv run python demo_target/app.py &

echo "[start.sh] launching agents-synthesizer (background poll loop)"
uv run python -m agents.synthesizer.main &

echo "[start.sh] launching the merged Orchestrator+UI app on :${PORT}"
exec uv run uvicorn deploy.common.merged_app:app --host 0.0.0.0 --port "${PORT}"
