"""Tests for the Orchestrator API's shared-secret auth (auth.py,
ORCHESTRATOR_API_KEY) -- added for real deployment, where /trigger and
/runs/* would otherwise be completely open to anyone who can reach the
port. Covers both the dependency function directly and the three routes
it's wired onto, plus the two routes that must stay open regardless
(/health, /ready -- a load balancer/uptime check needs to reach those
without a credential).
"""

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agents.common import run_store
from agents.common.config import settings
from agents.common.models.dag import DAGPlan, PendingInputRequest, RunState
from agents.orchestrator import main
from agents.orchestrator.auth import require_api_key


def test_require_api_key_no_ops_when_unconfigured(monkeypatch):
    """The default (local dev, e.g. `docker compose up` with no
    ORCHESTRATOR_API_KEY set) must leave every route reachable, exactly
    as it worked before this existed."""
    monkeypatch.setattr(settings, "orchestrator_api_key", "")
    require_api_key(x_api_key=None)  # must not raise


def test_require_api_key_rejects_a_missing_header_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key=None)
    assert exc_info.value.status_code == 401


def test_require_api_key_rejects_a_wrong_key(monkeypatch):
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key="wrong-key")
    assert exc_info.value.status_code == 401


def test_require_api_key_accepts_the_correct_key(monkeypatch):
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    require_api_key(x_api_key="secret-key")  # must not raise


# --- Wired onto the real routes ------------------------------------------


def test_trigger_returns_401_without_a_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    client = TestClient(main.app)

    resp = client.post("/trigger", json={"transcript": "what's new"})

    assert resp.status_code == 401


def test_trigger_succeeds_with_the_correct_key(monkeypatch):
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    monkeypatch.setattr(main, "execute_plan", lambda plan: None)
    client = TestClient(main.app)

    resp = client.post(
        "/trigger", json={"transcript": "what's new"}, headers={"X-API-Key": "secret-key"}
    )

    assert resp.status_code == 200


def test_get_run_returns_401_without_a_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    plan = DAGPlan(run_id="auth-test-run", transcript="t", created_at=datetime.now(timezone.utc), nodes=[], edges=[])
    run_store.save_run(RunState(run_id="auth-test-run", plan=plan, node_states={}))
    client = TestClient(main.app)

    resp = client.get("/runs/auth-test-run")

    assert resp.status_code == 401


def test_get_run_succeeds_with_the_correct_key(monkeypatch):
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    plan = DAGPlan(run_id="auth-test-run-2", transcript="t", created_at=datetime.now(timezone.utc), nodes=[], edges=[])
    run_store.save_run(RunState(run_id="auth-test-run-2", plan=plan, node_states={}))
    client = TestClient(main.app)

    resp = client.get("/runs/auth-test-run-2", headers={"X-API-Key": "secret-key"})

    assert resp.status_code == 200


def test_resume_returns_401_without_a_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    plan = DAGPlan(run_id="auth-resume-run", transcript="t", created_at=datetime.now(timezone.utc), nodes=[], edges=[])
    run = RunState(
        run_id="auth-resume-run", plan=plan, node_states={}, overall_status="awaiting_human_input",
        pending_input=PendingInputRequest(fields=["email"], prompt="need info", url="https://gated.test", node_id="fetch"),
    )
    run_store.save_run(run)
    client = TestClient(main.app)

    resp = client.post("/runs/auth-resume-run/resume", json={"email": "judge@example.com"})

    assert resp.status_code == 401


def test_health_and_ready_stay_open_even_when_a_key_is_configured(monkeypatch):
    """A load balancer / uptime check has no way to supply a credential
    -- these two routes must never gate on ORCHESTRATOR_API_KEY."""
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    client = TestClient(main.app)

    assert client.get("/health").status_code == 200
    # /ready may itself return 503 if a real dependency (Qdrant) isn't
    # reachable in this test environment -- the property under test is
    # that it's never 401, not that it's always 200.
    assert client.get("/ready").status_code != 401


def test_webhook_omi_is_unaffected_by_orchestrator_api_key(monkeypatch):
    """/webhook/omi carries its own separate secret (verify_webhook_secret,
    X-Omi-Signature) -- ORCHESTRATOR_API_KEY must not additionally gate
    it (and, with OMI_WEBHOOK_SECRET unset here, no credential at all
    should be required)."""
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")
    monkeypatch.setattr(settings, "omi_webhook_secret", "")
    monkeypatch.setattr(main, "execute_plan", lambda plan: None)
    client = TestClient(main.app)

    resp = client.post("/webhook/omi", json={"transcript": "what's new"})

    assert resp.status_code == 200
