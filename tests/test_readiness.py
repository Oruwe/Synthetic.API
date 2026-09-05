"""Tests for the liveness/readiness split: Qdrant unreachable must flip
readiness to False; missing API keys are reported but degrade gracefully
rather than failing readiness (the system still answers, just with fewer
sources / template drafts -- see agents/common/readiness.py)."""

import agents.common.readiness as readiness
from agents.common.config import settings


def test_ready_when_qdrant_reachable_and_keys_configured(monkeypatch):
    monkeypatch.setattr(readiness.qdrant_store, "get_client", lambda: _FakeClient(raises=False))
    monkeypatch.setattr(settings, "tavily_api_key", "key")
    monkeypatch.setattr(settings, "openrouter_api_key", "key")

    is_ready, checks = readiness.run_readiness_checks()

    assert is_ready is True
    assert all(c.ok for c in checks)


def test_not_ready_when_qdrant_unreachable(monkeypatch):
    monkeypatch.setattr(readiness.qdrant_store, "get_client", lambda: _FakeClient(raises=True))
    monkeypatch.setattr(settings, "tavily_api_key", "key")
    monkeypatch.setattr(settings, "openrouter_api_key", "key")

    is_ready, checks = readiness.run_readiness_checks()

    assert is_ready is False
    qdrant_check = next(c for c in checks if c.name == "qdrant")
    assert qdrant_check.ok is False
    assert qdrant_check.hard_required is True


def test_missing_api_keys_reported_but_do_not_fail_readiness(monkeypatch):
    monkeypatch.setattr(readiness.qdrant_store, "get_client", lambda: _FakeClient(raises=False))
    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    monkeypatch.setattr(settings, "lyzr_api_key", "")

    is_ready, checks = readiness.run_readiness_checks()

    assert is_ready is True  # soft checks don't block readiness
    tavily_check = next(c for c in checks if c.name == "tavily_api_key")
    assert tavily_check.ok is False
    assert tavily_check.hard_required is False


class _FakeClient:
    def __init__(self, raises: bool):
        self._raises = raises

    def get_collections(self):
        if self._raises:
            raise RuntimeError("connection refused")
        return None
