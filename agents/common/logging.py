"""Structured JSON logging shared by all three agents.

Every log line is a JSON object correlated by `run_id` and, where
applicable, `node_id` — this is what makes a DAG run inspectable in
`docker compose logs` independent of whether Langfuse is up, since
Langfuse tracing is best-effort (see langfuse_tracer.py) and logging
must never depend on it.
"""

import logging
import sys

import structlog


def configure_logging(service_name: str) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(**initial_context):
    return structlog.get_logger(**initial_context)


def bind_run_context(*, run_id: str, node_id: str | None = None) -> None:
    """Bind run_id/node_id so every subsequent log line in this async/thread
    context carries them without having to pass them explicitly."""
    ctx = {"run_id": run_id}
    if node_id is not None:
        ctx["node_id"] = node_id
    structlog.contextvars.bind_contextvars(**ctx)


def clear_run_context() -> None:
    structlog.contextvars.unbind_contextvars("run_id", "node_id")
