"""Tests for the Synthesizer's run-store-based watcher: it must detect
newly-completed runs (terminal overall_status) exactly once each, and
leave in-progress runs alone."""

from datetime import datetime, timezone

from agents.common.models.dag import DAGPlan, RunState
from agents.synthesizer import watcher


def _run(run_id: str, overall_status: str) -> RunState:
    plan = DAGPlan(run_id=run_id, transcript="t", created_at=datetime.now(timezone.utc), nodes=[], edges=[])
    return RunState(run_id=run_id, plan=plan, node_states={}, overall_status=overall_status)


def test_poll_once_detects_newly_completed_runs(monkeypatch):
    runs = [_run("r1", "completed"), _run("r2", "running")]
    monkeypatch.setattr("agents.common.run_store.list_runs", lambda: runs)

    seen = set()
    found = watcher.poll_once(seen)

    assert [r.run_id for r in found] == ["r1"]
    assert seen == {"r1"}


def test_poll_once_treats_failed_and_circuit_broken_as_terminal(monkeypatch):
    runs = [_run("r1", "failed"), _run("r2", "circuit_broken")]
    monkeypatch.setattr("agents.common.run_store.list_runs", lambda: runs)

    found = watcher.poll_once(set())

    assert {r.run_id for r in found} == {"r1", "r2"}


def test_poll_once_does_not_re_notify_already_seen_runs(monkeypatch):
    monkeypatch.setattr("agents.common.run_store.list_runs", lambda: [_run("r1", "completed")])

    seen = {"r1"}
    found = watcher.poll_once(seen)

    assert found == []


def test_load_and_save_seen_round_trip(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    watcher._save_seen({"run-a", "run-b"})

    reloaded = watcher._load_seen()
    assert reloaded == {"run-a", "run-b"}
