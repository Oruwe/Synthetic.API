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
import logging as _stdlib_logging
from collections.abc import Callable
from datetime import UTC, datetime

from agents.common.config import settings
from agents.common.logging import get_logger

logger = get_logger(component="langfuse_tracer")

_langfuse_client = None
_langfuse_init_attempted = False


class _SDKLogForwardingHandler(_stdlib_logging.Handler):
    """Catches the Langfuse SDK's OWN internal `logging.getLogger("langfuse")`
    calls (see langfuse/parse_error.py's handle_exception/handle_fern_exception)
    and routes them through this project's structlog logger instead.

    Why this exists: the SDK batches trace/generation events and flushes them
    on a background thread we don't control -- every try/except in this file
    only covers the *synchronous* enqueue call (trace()/generation() themselves
    essentially never raise; they just append to a local queue), so a real send
    failure (bad/missing API keys, self-hosted instance unreachable, a 4xx from
    a malformed event) surfaces later, on that background thread, via the SDK's
    own stdlib logger -- confirmed live: a run with no LANGFUSE_PUBLIC_KEY/
    LANGFUSE_SECRET_KEY set printed exactly this ("Unexpected error occurred.
    Please check your request and contact support: https://langfuse.com/
    support.") straight to stdout mid-run, indistinguishable from a real
    pipeline crash next to this project's structured JSON log lines, even
    though the run itself completed successfully (fail-open working exactly
    as designed -- just not observable as such). Tagging it and routing it
    through the same logger as everything else fixes that without touching
    agents/common/logging.py's deliberate bare `%(message)s` root format,
    which other third-party loggers may still rely on."""

    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        try:
            log_fn = logger.error if record.levelno >= _stdlib_logging.ERROR else logger.warning
            log_fn("langfuse_sdk_background_failure", detail=record.getMessage())
        except Exception:  # noqa: BLE001 - a logging handler must never itself raise
            pass


def _install_sdk_log_forwarding() -> None:
    """Idempotent -- safe to call more than once (e.g. module re-imported
    under test) without stacking duplicate handlers."""
    sdk_logger = _stdlib_logging.getLogger("langfuse")
    if any(isinstance(h, _SDKLogForwardingHandler) for h in sdk_logger.handlers):
        return
    sdk_logger.addHandler(_SDKLogForwardingHandler())
    # Don't ALSO let it fall through to the root logger's bare-message
    # handler (agents/common/logging.py's configure_logging) -- that would
    # print the same failure twice, once tagged and once raw.
    sdk_logger.propagate = False


_install_sdk_log_forwarding()


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
    that would show up on the trace looking like a real (empty) call.

    Called OUTSIDE _safe() at both call sites below (building the kwargs
    for trace.generation(), before entering the _safe(lambda: ...) call),
    so it must never raise on its own -- confirmed live: LyzrBackend's
    usage dict is opportunistic and unverified (see lyzr_wrapper.py's
    module docstring), and if it ever happened to include a "unit" key,
    `ModelUsage(unit="TOKENS", **usage)` raises TypeError (duplicate
    keyword) with nothing here to catch it, discarding an otherwise-
    successful LLM answer up in drafter.py's own try/except and silently
    downgrading to the template fallback.
    """
    if not usage:
        return None
    try:
        from langfuse.model import ModelUsage

        return ModelUsage(unit="TOKENS", **usage)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the pipeline (see module docstring)
        logger.warning("langfuse_model_usage_build_failed", error=str(exc))
        return None


def traced_llm_call(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, system_prompt: str, user_input: str, *, run_id: str, node_id: str):
            client = _get_client()
            started_at = datetime.now(UTC)
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
                    # Capture the message into a plain local BEFORE the
                    # lambda, not `str(exc)` inside it -- Python deletes the
                    # `except ... as exc` binding when this block ends, so a
                    # closure over `exc` itself only works today because
                    # _safe() happens to invoke it synchronously; capturing
                    # the string now removes that fragile assumption.
                    error_str = str(exc)
                    _safe(lambda: trace.update(output={"error": error_str}, level="ERROR"))
                raise
            else:
                if trace is not None:
                    ended_at = datetime.now(UTC)
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
            started_at = datetime.now(UTC)
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
                    # Capture the message into a plain local BEFORE the
                    # lambda, not `str(exc)` inside it -- Python deletes the
                    # `except ... as exc` binding when this block ends, so a
                    # closure over `exc` itself only works today because
                    # _safe() happens to invoke it synchronously; capturing
                    # the string now removes that fragile assumption.
                    error_str = str(exc)
                    _safe(lambda: trace.update(output={"error": error_str}, level="ERROR"))
                raise
            else:
                if trace is not None:
                    ended_at = datetime.now(UTC)
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
