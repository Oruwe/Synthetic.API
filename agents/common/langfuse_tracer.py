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
from datetime import datetime, timezone
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


def _model_usage(usage: dict | None):
    """Builds a langfuse.model.ModelUsage from the {"input","output","total"}
    dict backends attach to LLMResult (see lyzr_wrapper.py). Returns None if
    there's nothing to report, rather than sending a bogus all-zero usage
    that would show up on the trace looking like a real (empty) call."""
    if not usage:
        return None
    from langfuse.model import ModelUsage

    return ModelUsage(unit="TOKENS", **usage)


def traced_llm_call(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, system_prompt: str, user_input: str, *, run_id: str, node_id: str):
            client = _get_client()
            started_at = datetime.now(timezone.utc)
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
                    ended_at = datetime.now(timezone.utc)
                    # Set by LyzrAgentWrapper.run() (lyzr_wrapper.py) during
                    # the call above -- without these, every generation
                    # showed 0 tokens / $0.00 regardless of the real call,
                    # since start_time==end_time and no model/usage was
                    # ever passed to Langfuse.
                    model = getattr(self, "last_model", None)
                    usage = _model_usage(getattr(self, "last_usage", None))
                    _safe(
                        lambda: trace.generation(
                            name=name,
                            input=user_input,
                            output=result,
                            model=model,
                            usage=usage,
                            start_time=started_at,
                            end_time=ended_at,
                            metadata={"latency_ms": (ended_at - started_at).total_seconds() * 1000},
                        )
                    )
                return result

        return wrapper

    return decorator


def traced_vision_call(name: str) -> Callable:
    """Like `traced_llm_call`, but for image+prompt -> text calls
    (agents/common/vision_wrapper.py). The traced "input" is `image_ref` (a
    file path or URL) rather than the actual image bytes -- tracing must
    never balloon a trace with a multi-MB base64 payload."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, image_ref: str, prompt: str, *, run_id: str, node_id: str):
            client = _get_client()
            started_at = datetime.now(timezone.utc)
            trace = None
            if client is not None:
                try:
                    trace = client.trace(name=name, session_id=run_id, metadata={"node_id": node_id})
                except Exception as exc:  # noqa: BLE001
                    logger.warning("langfuse_trace_start_failed", error=str(exc), run_id=run_id)

            try:
                result = fn(self, image_ref, prompt, run_id=run_id, node_id=node_id)
            except Exception as exc:
                if trace is not None:
                    _safe(lambda: trace.update(output={"error": str(exc)}, level="ERROR"))
                raise
            else:
                if trace is not None:
                    ended_at = datetime.now(timezone.utc)
                    model = getattr(self, "last_model", None)
                    usage = _model_usage(getattr(self, "last_usage", None))
                    _safe(
                        lambda: trace.generation(
                            name=name,
                            input={"image_ref": image_ref, "prompt": prompt},
                            output=result,
                            model=model,
                            usage=usage,
                            start_time=started_at,
                            end_time=ended_at,
                            metadata={"latency_ms": (ended_at - started_at).total_seconds() * 1000},
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
