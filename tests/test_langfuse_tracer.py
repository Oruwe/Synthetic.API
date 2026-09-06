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

from agents.common import langfuse_tracer
from agents.common.config import settings


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
