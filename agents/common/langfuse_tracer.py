"""Langfuse tracing, wrapped so it can NEVER break the pipeline.

Structured logging (agents/common/logging.py) is the source of truth for
"is this run healthy" and works with Langfuse entirely down. Langfuse adds
a trace UI (spans, latency, tokens) on top, grouped by run_id — genuinely
useful for the demo video, but strictly best-effort: every call here is
wrapped in try/except, and a Langfuse outage/misconfiguration only ever
produces a warning log line, never an exception that propagates to the
caller.
"""

import functools
import time
from typing import Callable

from agents.common.config import settings
from agents.common.logging import get_logger

logger = get_logger(component="langfuse_tracer")

_langfuse_client = None
_langfuse_init_attempted = False


def _get_client():
    global _langfuse_client, _langfuse_init_attempted
    if _langfuse_init_attempted:
        return _langfuse_client
    _langfuse_init_attempted = True
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key or None,
            secret_key=settings.langfuse_secret_key or None,
        )
    except Exception as exc:  # noqa: BLE001 - tracing must never block the pipeline
        logger.warning("langfuse_init_failed", error=str(exc))
        _langfuse_client = None
    return _langfuse_client


def traced_llm_call(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, system_prompt: str, user_input: str, *, run_id: str, node_id: str):
            client = _get_client()
            started = time.monotonic()
            trace = None
            if client is not None:
                try:
                    trace = client.trace(name=name, session_id=run_id, metadata={"node_id": node_id})
                except Exception as exc:  # noqa: BLE001
                    logger.warning("langfuse_trace_start_failed", error=str(exc), run_id=run_id)

            try:
                result = fn(self, system_prompt, user_input, run_id=run_id, node_id=node_id)
            except Exception as exc:
                if trace is not None:
                    _safe(lambda: trace.update(output={"error": str(exc)}, level="ERROR"))
                raise
            else:
                if trace is not None:
                    latency_ms = (time.monotonic() - started) * 1000
                    _safe(
                        lambda: trace.generation(
                            name=name,
                            input=user_input,
                            output=result,
                            metadata={"latency_ms": latency_ms},
                        )
                    )
                return result

        return wrapper

    return decorator


def _safe(fn: Callable) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - tracing must never block the pipeline
        logger.warning("langfuse_call_failed", error=str(exc))
