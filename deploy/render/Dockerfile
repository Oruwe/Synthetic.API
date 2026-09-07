# Render.com free-tier Docker Web Service deployment.
#
# See deploy/render/SETUP.md for the full setup story and
# deploy/common/merged_app.py's docstring for why this exists at all: like
# Hugging Face Spaces, Render routes external traffic to exactly one port
# per service, unlike docker-compose's separate agents-orchestrator (8000)
# / demo-ui (7860) services -- so this bundles both, plus demo_target and
# agents-synthesizer, into one container, supervised by
# deploy/common/start.sh. That file and deploy/common/merged_app.py are
# shared verbatim with deploy/huggingface/ -- only how the image gets
# built and where Qdrant lives differ per target, and Qdrant lives in the
# same place either way (a free Qdrant Cloud cluster, not self-hosted).
#
# Unlike deploy/huggingface/Dockerfile, this does NOT git-clone the source
# at build time -- Render builds directly from this repo (see SETUP.md for
# the exact "Dockerfile Path" / "Docker Build Context Directory" fields to
# set in Render's dashboard), so the build context already IS the whole
# repo.
#
# Real, unverified risk worth knowing before you build this: Render's free
# tier is 512MB RAM -- much tighter than Hugging Face Spaces' 16GB. This
# has not been confirmed to fit (build-time or run-time) from either
# sandbox available while writing this; SETUP.md's verification step is
# the actual test, not a formality.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers

WORKDIR /app

# Render's build context is this repo's root (see SETUP.md) -- the whole
# tree is available here already, unlike deploy/huggingface/Dockerfile's
# git-clone-at-build-time approach.
COPY . /app

RUN pip install --no-cache-dir uv

# --extra hf-space pulls gradio into this SAME uv-managed venv alongside
# every other agents/ dependency (see pyproject.toml's
# [project.optional-dependencies] -- named for the first target it was
# built for, reused verbatim here since the dependency need is identical).
# --group demo-target adds demo_target's own Flask dependency into that
# SAME venv too (see pyproject.toml's [dependency-groups] comment for why
# this is a group, not its own requirements.txt or another extra) -- one
# container/process group here, unlike docker-compose's separate
# demo_target service+image, so both belong in the one environment.
# agents-orchestrator/agents-synthesizer's own docker-compose images never
# install either.
RUN uv sync --frozen --no-dev --extra hf-space --group demo-target \
    || uv sync --no-dev --extra hf-space --group demo-target

# Playwright's Chromium, with its headless-mode system lib deps.
RUN uv run playwright install --with-deps chromium

# Pre-fetch the FastEmbed ONNX model at build time -- same reasoning as
# agents/Dockerfile: the first DAG run shouldn't pay a slow (and, on a
# free-tier host, possibly network-restricted) download on the demo's
# critical path.
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

RUN chmod +x /app/deploy/common/start.sh

# Render sets $PORT itself at runtime and expects the service to bind to
# it -- deploy/common/start.sh already reads $PORT (falling back to 7860
# if unset), so nothing else needs to change here for that.
EXPOSE 7860

CMD ["/app/deploy/common/start.sh"]
