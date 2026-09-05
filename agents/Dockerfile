# Shared image for both agent services (orchestrator + synthesizer);
# docker-compose.yml selects which process to run via `command:`.
FROM python:3.11-slim

# Playwright's Chromium needs these system libs even in headless mode.
ENV PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

RUN uv run playwright install --with-deps chromium

# Pre-fetch the FastEmbed ONNX model at build time so the first DAG run
# doesn't pay a slow (and, in a network-restricted deployment, possibly
# unavailable) download on the demo's critical path.
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY agents ./agents
COPY .env.example .env.example

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "agents.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
