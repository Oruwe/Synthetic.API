# Hugging Face Spaces (Docker SDK) deployment.
#
# STATUS (flagged honestly, see SETUP.md's top note): as of ~July 2026,
# Hugging Face marked the Docker SDK as a PAID feature when creating a new
# Space (confirmed on HF's own community forums, described there as
# possibly temporary, tied to an infrastructure shortage). This file is
# kept working and ready -- deploy/render/ is the currently-free path built
# on the exact same deploy/common/ pieces, use that instead unless HF's
# Docker SDK is free again by the time you read this (worth checking).
#
# See deploy/huggingface/SETUP.md for the full setup story and
# deploy/common/merged_app.py's docstring for why this exists at all: HF
# Spaces routes external traffic to exactly one port, unlike
# docker-compose's separate agents-orchestrator (8000) / demo-ui (7860)
# services -- so this bundles both, plus demo_target and
# agents-synthesizer, into one container, supervised by start.sh.
#
# Builds by git-cloning the real source from GitHub at build time (see
# SOURCE_REPO/SOURCE_REF below) rather than requiring the HF Space's own
# repo to mirror this repo's whole file tree -- the Space's own repo only
# ever needs to contain a copy of THIS file plus deploy/huggingface/
# README.md (HF's own frontmatter file, copied as-is -- see SETUP.md for
# the full walkthrough). A rebuild always pulls whatever SOURCE_REF
# currently points to.
#
# Deliberately does NOT bundle a Qdrant server -- this deployment points
# QDRANT_URL at a free Qdrant Cloud cluster (qdrant_api_key added to
# config.py specifically for this) instead of self-hosting one in the
# container. Simpler and lower-risk than copying a Qdrant binary out of
# another image and hoping it's glibc-compatible with this base image, and
# arguably a better hackathon story: it's Qdrant's own official managed
# offering, not a workaround.
#
# Also deliberately does NOT run Langfuse/Postgres/Redis -- both are
# already designed to fail open (see langfuse_tracer.py's module docstring
# and REDIS_URL's comment in .env.example): the pipeline runs identically
# without them, just without a trace UI and with an in-process (not
# cross-process-distributed) workflow-memory lock, which is correct anyway
# for a single container with no horizontal scaling.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers

# git: to clone the source below. curl: uv's own installer needs it.
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG SOURCE_REPO=https://github.com/Oruwe/Synthetic.API.git
ARG SOURCE_REF=main
RUN git clone --branch "${SOURCE_REF}" --depth 1 "${SOURCE_REPO}" /app

RUN pip install --no-cache-dir uv

# --extra hf-space pulls gradio into this SAME uv-managed venv alongside
# every other agents/ dependency (see pyproject.toml's
# [project.optional-dependencies]) -- the docker-compose deployment never
# installs this extra, so agents-orchestrator/agents-synthesizer's own
# images stay exactly as lean as before this file existed.
RUN uv sync --frozen --no-dev --extra hf-space || uv sync --no-dev --extra hf-space

# Playwright's Chromium, with its headless-mode system lib deps.
RUN uv run playwright install --with-deps chromium

# Pre-fetch the FastEmbed ONNX model at build time -- same reasoning as
# agents/Dockerfile: the first DAG run shouldn't pay a slow (and, on a
# free-tier host, possibly network-restricted) download on the demo's
# critical path.
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# demo_target's own (separate, lightweight) Flask dependency -- folded
# into this same venv since this is one container/process group, unlike
# docker-compose's separate demo_target service+image.
RUN uv run python -m pip install --no-cache-dir -r demo_target/requirements.txt

RUN chmod +x /app/deploy/common/start.sh

# HF Spaces' Docker SDK routes external traffic to exactly this port --
# must match SETUP.md's `app_port` frontmatter and start.sh's own
# $PORT default.
EXPOSE 7860

CMD ["/app/deploy/common/start.sh"]
