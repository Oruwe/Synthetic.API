"""Tests for the real Lyzr Agent SDK integration in lyzr_wrapper.py.

The SDK's real network call is mocked (patching lyzr_python_sdk.LyzrAgentAPI,
which LyzrBackend.complete() imports lazily) -- offline and deterministic,
consistent with the rest of the suite. What's actually exercised: the
agent_id-required guard, the defensive response-shape parsing in
_extract_chat_text() (the one piece of the real SDK that isn't verifiable
from outside a real account -- see the module docstring), that run_id is
threaded through as Lyzr's session_id, and that LyzrAgentWrapper still
falls back to OpenRouter cleanly when the real Lyzr call fails.
"""

import lyzr_python_sdk
import pytest

import agents.common.lyzr_wrapper as lyzr_wrapper
from agents.common.config import settings
from agents.common.lyzr_wrapper import LLMResult, LyzrAgentWrapper, LyzrBackend, _extract_chat_text


class _FakeInference:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def chat(self, payload):
        self.calls.append(payload)
        return self._response


class _FakeLyzrAgentAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.inference = _FakeInference(_FakeLyzrAgentAPI.next_response)

    # set by each test before constructing LyzrBackend's call
    next_response = {"response": "default"}


def _patch_lyzr_client(monkeypatch, response):
    _FakeLyzrAgentAPI.next_response = response
    monkeypatch.setattr(lyzr_python_sdk, "LyzrAgentAPI", _FakeLyzrAgentAPI)


# --- _extract_chat_text: the one genuinely unverifiable piece -----------


def test_extract_chat_text_from_dict_response_key():
    assert _extract_chat_text({"response": "the answer"}) == "the answer"


def test_extract_chat_text_falls_back_to_other_dict_keys():
    assert _extract_chat_text({"message": "the answer"}) == "the answer"
    assert _extract_chat_text({"content": "the answer"}) == "the answer"
    assert _extract_chat_text({"text": "the answer"}) == "the answer"


def test_extract_chat_text_from_object_attribute():
    class _Resp:
        response = "the answer"

    assert _extract_chat_text(_Resp()) == "the answer"


def test_extract_chat_text_unknown_shape_falls_back_to_str_and_warns(caplog):
    result = _extract_chat_text({"totally_unexpected_key": "value"})
    assert result == "{'totally_unexpected_key': 'value'}"


# --- LyzrBackend.complete() ------------------------------------------


def test_complete_requires_agent_id():
    backend = LyzrBackend(api_key="key", agent_role="synthesizer", agent_id="")

    with pytest.raises(RuntimeError, match="LYZR_AGENT_ID"):
        backend.complete("system prompt", "user input")


def test_complete_calls_chat_with_expected_payload_and_run_id_as_session(monkeypatch):
    _patch_lyzr_client(monkeypatch, {"response": "the drafted answer"})
    monkeypatch.setattr(settings, "lyzr_user_id", "synthetic-api")
    backend = LyzrBackend(api_key="key", agent_role="synthesizer", agent_id="agent-123")

    result = backend.complete("ignored system prompt", "what is the status?", run_id="run-42")

    assert result.text == "the drafted answer"
    assert result.model == "lyzr:agent-123"


def test_complete_uses_run_id_as_session_id(monkeypatch):
    _patch_lyzr_client(monkeypatch, {"response": "answer"})
    backend = LyzrBackend(api_key="key", agent_role="synthesizer", agent_id="agent-123")
    # Capture the actual fake client instance's recorded call.
    captured = {}

    class _CapturingInference(_FakeInference):
        def chat(self, payload):
            captured.update(payload)
            return super().chat(payload)

    class _CapturingLyzrAgentAPI(_FakeLyzrAgentAPI):
        def __init__(self, api_key):
            super().__init__(api_key)
            self.inference = _CapturingInference(_FakeLyzrAgentAPI.next_response)

    monkeypatch.setattr(lyzr_python_sdk, "LyzrAgentAPI", _CapturingLyzrAgentAPI)

    backend.complete("system prompt", "the question", run_id="run-42")

    assert captured["agent_id"] == "agent-123"
    assert captured["message"] == "the question"
    assert captured["session_id"] == "run-42"


def test_complete_falls_back_to_a_default_session_id_when_run_id_missing(monkeypatch):
    captured = {}

    class _CapturingInference(_FakeInference):
        def chat(self, payload):
            captured.update(payload)
            return super().chat(payload)

    class _CapturingLyzrAgentAPI(_FakeLyzrAgentAPI):
        def __init__(self, api_key):
            super().__init__(api_key)
            self.inference = _CapturingInference({"response": "answer"})

    monkeypatch.setattr(lyzr_python_sdk, "LyzrAgentAPI", _CapturingLyzrAgentAPI)
    backend = LyzrBackend(api_key="key", agent_role="synthesizer", agent_id="agent-123")

    backend.complete("system prompt", "the question")  # no run_id

    assert captured["session_id"] == "synthetic-api-default-session"


def test_complete_extracts_usage_when_present(monkeypatch):
    _patch_lyzr_client(monkeypatch, {"response": "answer", "usage": {"input": 10, "output": 5, "total": 15}})
    backend = LyzrBackend(api_key="key", agent_role="synthesizer", agent_id="agent-123")

    result = backend.complete("system prompt", "question")

    assert result.usage == {"input": 10, "output": 5, "total": 15}


def test_complete_leaves_usage_none_when_absent(monkeypatch):
    _patch_lyzr_client(monkeypatch, {"response": "answer"})
    backend = LyzrBackend(api_key="key", agent_role="synthesizer", agent_id="agent-123")

    result = backend.complete("system prompt", "question")

    assert result.usage is None


# --- LyzrAgentWrapper: falls back to OpenRouter on any real failure ------


def test_wrapper_falls_back_to_openrouter_when_lyzr_call_fails(monkeypatch):
    monkeypatch.setattr(settings, "lyzr_enabled", True)
    monkeypatch.setattr(settings, "lyzr_api_key", "key")
    monkeypatch.setattr(settings, "lyzr_agent_id", "")  # forces LyzrBackend to raise

    fallback_result = LLMResult(text="fallback answer", model="deepseek/deepseek-chat-v3.1", usage=None)
    monkeypatch.setattr(
        lyzr_wrapper.OpenRouterBackend, "complete", lambda self, sp, ui, *, run_id=None: fallback_result
    )

    wrapper = LyzrAgentWrapper(agent_role="synthesizer")
    text = wrapper.run("system prompt", "question", run_id="run-1", node_id="draft_answer")

    assert text == "fallback answer"
    assert wrapper.last_model == "deepseek/deepseek-chat-v3.1"
