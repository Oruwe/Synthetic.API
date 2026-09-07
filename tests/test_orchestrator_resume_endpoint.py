"""Tests for POST /runs/{run_id}/resume -- the HTTP surface a human answers
a paused run's gated-content prompt through. The validation logic here
(404/409/400, which fields are actually required) isn't exercised by
test_executor_pause_resume.py (that file tests resume_plan() directly),
so it needs its own coverage. First FastAPI-endpoint-level test file in
this repo -- everything else tests the functions main.py wires together,
not the HTTP layer itself, but this endpoint's own request validation is
real logic worth proving, not just wiring.
"""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from agents.common import run_store
from agents.common.models.dag import DAGPlan, PendingInputRequest, RunState
from agents.orchestrator import main


def _paused_run(run_id: str, fields=("email",)) -> RunState:
    plan = DAGPlan(run_id=run_id, transcript="t", created_at=datetime.now(timezone.utc), nodes=[], edges=[])
    run = RunState(
        run_id=run_id,
        plan=plan,
        node_states={},
        overall_status="awaiting_human_input",
        pending_input=PendingInputRequest(
            fields=list(fields), prompt="need info", url="https://gated.test", node_id="fetch"
        ),
    )
    run_store.save_run(run)
    return run


def test_resume_returns_404_for_an_unknown_run():
    client = TestClient(main.app)
    resp = client.post("/runs/does-not-exist/resume", json={"email": "x@example.com"})
    assert resp.status_code == 404


def test_resume_returns_409_when_the_run_is_not_awaiting_input(monkeypatch):
    plan = DAGPlan(run_id="not-paused", transcript="t", created_at=datetime.now(timezone.utc), nodes=[], edges=[])
    run = RunState(run_id="not-paused", plan=plan, node_states={}, overall_status="completed")
    run_store.save_run(run)

    client = TestClient(main.app)
    resp = client.post("/runs/not-paused/resume", json={"email": "x@example.com"})

    assert resp.status_code == 409


def test_resume_returns_400_when_a_required_field_is_missing(monkeypatch):
    _paused_run("missing-field-run", fields=["email", "password"])

    client = TestClient(main.app)
    resp = client.post("/runs/missing-field-run/resume", json={"email": "x@example.com"})  # no password

    assert resp.status_code == 400
    assert "password" in resp.json()["detail"]


def test_resume_accepts_a_valid_request_and_queues_the_background_task(monkeypatch):
    _paused_run("valid-run")
    captured = {}
    monkeypatch.setattr(main, "resume_plan", lambda run_id, provided: captured.update(run_id=run_id, provided=provided))

    client = TestClient(main.app)
    resp = client.post("/runs/valid-run/resume", json={"email": "judge@example.com"})

    assert resp.status_code == 200
    assert resp.json()["run_id"] == "valid-run"
    assert captured == {"run_id": "valid-run", "provided": {"email": "judge@example.com"}}


def test_resume_never_logs_the_password_value(monkeypatch, caplog):
    _paused_run("login-run", fields=["email", "password"])
    monkeypatch.setattr(main, "resume_plan", lambda run_id, provided: None)

    client = TestClient(main.app)
    resp = client.post("/runs/login-run/resume", json={"email": "judge@example.com", "password": "hunter2"})

    assert resp.status_code == 200
    assert "hunter2" not in caplog.text
