# Build context is the REPO ROOT (see docker-compose.yml's context: .,
# dockerfile: ui/Dockerfile), not this directory -- needed to reach the
# root pyproject.toml/uv.lock below. This service's dependencies are
# pinned there too now, in the `ui` dependency group -- not its own
# separate requirements.txt -- while staying just as lean as one: `uv
# sync --only-group ui --no-install-project` installs ONLY that group
# (gradio, requests, and their real transitive deps), never the main
# agents/ dependency list (fastapi, playwright, qdrant-client, ...) or
# the agents package itself. See pyproject.toml's own comment on why this
# is a dependency GROUP, not the hf-space optional-dependencies EXTRA
# (which deliberately does pull the full list in, for a different
# deployment that actually wants both together).
FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group ui --no-install-project

COPY ui/app.py .

EXPOSE 7860

# .venv/bin/python directly, not `uv run` -- `uv sync` above already
# created and populated /app/.venv; invoking it directly is simpler and
# more predictable than relying on `uv run`'s own project-detection
# behavior in a directory that deliberately has no [project] this group
# installs as (--no-install-project above).
CMD [".venv/bin/python", "app.py"]
