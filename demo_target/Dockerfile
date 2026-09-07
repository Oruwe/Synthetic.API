# Build context is the REPO ROOT (see docker-compose.yml's context: .,
# dockerfile: demo_target/Dockerfile) -- needed to reach the root
# pyproject.toml/uv.lock below. See ui/Dockerfile's comment for the full
# reasoning on why this is a `uv sync --only-group` dependency group, not
# its own separate requirements.txt or the hf-space optional-dependencies
# extra.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group demo-target --no-install-project

COPY demo_target/app.py .
COPY demo_target/templates ./templates
COPY demo_target/static ./static

EXPOSE 5050

CMD [".venv/bin/python", "app.py"]
