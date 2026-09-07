# Build context is the REPO ROOT (see docker-compose.yml's context: .,
# dockerfile: mock_portal/Dockerfile) -- needed to reach the root
# pyproject.toml/uv.lock below. See ui/Dockerfile's comment for the full
# reasoning on why this is a `uv sync --only-group` dependency group, not
# its own separate requirements.txt or the hf-space optional-dependencies
# extra.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group mock-portal --no-install-project

COPY mock_portal/app.py .
COPY mock_portal/templates ./templates
COPY mock_portal/static ./static
COPY mock_portal/data ./data

EXPOSE 5000

CMD [".venv/bin/python", "app.py"]
