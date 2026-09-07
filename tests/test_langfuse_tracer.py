"""Tests for langfuse_tracer.py's fail-open contract: tracing must never
break the pipeline, no matter what a backend's usage dict happens to
contain.

Real bug this guards against: _model_usage() called `ModelUsage(unit=
"TOKENS", **usage)` with no try/except of its own, called OUTSIDE the
_safe() wrapper both decorators otherwise use for every other Langfuse
call. LyzrBackend's usage dict is explicitly documented as opportunistic/
unverified (see lyzr_wrapper.py) -- if it ever included a "unit" key,
ModelUsage(**usage) raises TypeError (duplicate keyword), uncaught,
discarding an otherwise-successful LLM answer up in drafter.py's own
try/except and silently downgrading to the degraded template fallback.
"""

import logging

import pytest

from agents.common import langfuse_tracer
from agents.common.config import settings


class _FakeLogger:
    """Captures (event, kwargs) tuples instead of asserting on rendered log
    text -- see test_action_executor.py's own _FakeLogger for why: this
    codebase's structlog output format depends on global configure_logging()
    state set by whichever test module happens to import something that
    calls it first."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []  # (level, event, kwargs)

    def warning(self, event, **kwargs):
        self.calls.append(("warning", event, kwargs))

    def error(self, event, **kwargs):
        self.calls.append(("error", event, kwargs))


def test_model_usage_builds_normally_for_a_clean_usage_dict():
    # ModelUsage is a TypedDict (a plain dict at runtime, no validation) --
    # confirmed via the installed langfuse package rather than assumed.
    result = langfuse_tracer._model_usage({"input": 10, "output": 5, "total": 15})

    assert result is not None
    assert result["input"] == 10
    assert result["output"] == 5
    assert result["total"] == 15
    assert result["unit"] == "TOKENS"


def test_model_usage_returns_none_for_empty_or_missing_usage():
    assert langfuse_tracer._model_usage(None) is None
    assert langfuse_tracer._model_usage({}) is None


def test_model_usage_never_raises_on_a_colliding_unit_key():
    """The exact bug: a usage dict that happens to include "unit" collides
    with the hardcoded unit="TOKENS" kwarg. Must degrade to None, not raise."""
    result = langfuse_tracer._model_usage({"unit": "TOKENS", "input": 1, "output": 1, "total": 2})

    assert result is None


def test_model_usage_does_not_raise_on_unrelated_extra_keys():
    # A TypedDict provides no runtime key validation, so an unrelated extra
    # key is harmless (unlike the "unit" collision above, which is a
    # Python calling-convention conflict, not a langfuse-side check).
    result = langfuse_tracer._model_usage({"totally_unexpected_key": "value"})

    assert result == {"unit": "TOKENS", "totally_unexpected_key": "value"}


def test_traced_llm_call_succeeds_even_when_last_usage_has_a_colliding_key(monkeypatch):
    """End-to-end: a real Langfuse client whose trace.generation() would be
    called with a bad usage dict must not prevent the decorated function's
    real return value from getting back to the caller."""
    monkeypatch.setattr(settings, "langfuse_enabled", True)

    class _FakeTrace:
        def generation(self, **kwargs):
            # Exercise the real _model_usage call path end to end.
            langfuse_tracer._model_usage(kwargs.get("usage"))

        def update(self, **kwargs):
            pass

    class _FakeClient:
        def trace(self, **kwargs):
            return _FakeTrace()

    monkeypatch.setattr(langfuse_tracer, "_get_client", lambda: _FakeClient())

    class _Wrapped:
        last_model = "some-model"
        last_usage = {"unit": "TOKENS", "input": 1, "output": 1, "total": 2}  # the colliding case

        @langfuse_tracer.traced_llm_call(name="test_call")
        def run(self, system_prompt, user_input, *, run_id, node_id):
            return "the real answer"

    result = _Wrapped().run("sp", "ui", run_id="r1", node_id="n1")

    assert result == "the real answer"


def test_traced_llm_call_reports_the_real_exception_message_on_trace_update(monkeypatch):
    """The exact bug found by ruff (F821) while adding SDK log forwarding
    above: `except Exception as exc: ... lambda: trace.update(output=
    {"error": str(exc)})` closes over a name Python deletes at the end of
    the except block. This only worked because _safe() happens to call the
    lambda synchronously -- fragile, not proven. Assert the real error
    message end to end, through the real (non-monkeypatched) _safe(), so a
    regression back to closing over `exc` directly would show up as either
    a NameError inside _safe (swallowed, logged as `langfuse_call_failed`,
    and this assertion failing because update() never got a real message)
    or a wrong/missing error string reaching trace.update()."""
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    captured_updates = []

    class _FakeTrace:
        def generation(self, **kwargs):
            pass

        def update(self, **kwargs):
            captured_updates.append(kwargs)

    class _FakeClient:
        def trace(self, **kwargs):
            return _FakeTrace()

    monkeypatch.setattr(langfuse_tracer, "_get_client", lambda: _FakeClient())

    class _Wrapped:
        @langfuse_tracer.traced_llm_call(name="test_call")
        def run(self, system_prompt, user_input, *, run_id, node_id):
            raise ValueError("the specific real failure message")

    with pytest.raises(ValueError):
        _Wrapped().run("sp", "ui", run_id="r1", node_id="n1")

    assert len(captured_updates) == 1
    assert captured_updates[0]["output"]["error"] == "the specific real failure message"
    assert captured_updates[0]["level"] == "ERROR"


# --- SDK background-thread log forwarding ---------------------------------
#
# Real bug this guards against: the Langfuse SDK batches trace/generation
# events and flushes them on a background thread this module's own
# try/except blocks cannot reach (they only cover the synchronous enqueue
# call). A real send failure -- confirmed live with no LANGFUSE_PUBLIC_KEY/
# LANGFUSE_SECRET_KEY set -- surfaced as a bare, unlabeled
# "Unexpected error occurred..." string printed straight to stdout mid-run,
# via the SDK's own `logging.getLogger("langfuse")` calls (see
# langfuse/parse_error.py), indistinguishable from a real crash next to this
# project's structured JSON log lines even though the run itself completed
# successfully. _install_sdk_log_forwarding() must route that logger's
# output through the same structlog logger as everything else instead.


def test_install_sdk_log_forwarding_is_idempotent():
    sdk_logger = logging.getLogger("langfuse")
    before = len(sdk_logger.handlers)

    langfuse_tracer._install_sdk_log_forwarding()
    langfuse_tracer._install_sdk_log_forwarding()

    after = len(sdk_logger.handlers)
    # Module import already installed one handler; calling the installer
    # again (module already imported at collection time, same as a second
    # `import langfuse_tracer` elsewhere) must not stack a duplicate.
    assert after == before


def test_sdk_error_log_is_forwarded_as_a_tagged_structlog_error(monkeypatch):
    fake = _FakeLogger()
    monkeypatch.setattr(langfuse_tracer, "logger", fake)

    logging.getLogger("langfuse").error(
        "Unexpected error occurred. Please check your request and contact support: "
        "https://langfuse.com/support."
    )

    assert len(fake.calls) == 1
    level, event, kwargs = fake.calls[0]
    assert level == "error"
    assert event == "langfuse_sdk_background_failure"
    assert "Unexpected error occurred" in kwargs["detail"]


def test_sdk_warning_log_is_forwarded_at_warning_level_not_error(monkeypatch):
    fake = _FakeLogger()
    monkeypatch.setattr(langfuse_tracer, "logger", fake)

    logging.getLogger("langfuse").warning("some non-fatal SDK notice")

    assert len(fake.calls) == 1
    level, event, kwargs = fake.calls[0]
    assert level == "warning"
    assert event == "langfuse_sdk_background_failure"
    assert kwargs["detail"] == "some non-fatal SDK notice"


def test_sdk_logger_does_not_propagate_to_root_and_double_print(capsys):
    """Must not ALSO fall through to the root logger's bare-message handler
    (agents/common/logging.py's configure_logging) -- that would print the
    same failure twice, once tagged and once raw."""
    assert logging.getLogger("langfuse").propagate is False


def test_a_handler_emit_failure_never_raises(monkeypatch):
    """A logging handler must never itself raise -- confirmed here by making
    the forwarding call blow up and checking .emit() swallows it rather than
    propagating into whatever code path triggered the original log call."""
    handler = langfuse_tracer._SDKLogForwardingHandler()

    class _ExplodingLogger:
        def error(self, *args, **kwargs):
            raise RuntimeError("logging itself is broken")

        def warning(self, *args, **kwargs):
            raise RuntimeError("logging itself is broken")

    monkeypatch.setattr(langfuse_tracer, "logger", _ExplodingLogger())
    record = logging.LogRecord("langfuse", logging.ERROR, __file__, 1, "boom", None, None)

    handler.emit(record)  # must not raise
