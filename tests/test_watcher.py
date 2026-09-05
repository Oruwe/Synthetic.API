"""Tests for the Synthesizer's run-store-based watcher: it must detect
newly-completed runs (terminal overall_status) exactly once each via the
lightweight index (list_run_summaries), leave in-progress runs alone, and
periodically sweep for retention without doing so on every single poll."""

from datetime import datetime, timezone

from agents.common.models.dag import DAGPlan, RunState
from agents.synthesizer import watcher


def _run(run_id: str, overall_status: str) -> RunState:
    plan = DAGPlan(run_id=run_id, transcript="t", created_at=datetime.now(timezone.utc), nodes=[], edges=[])
    return RunState(run_id=run_id, plan=plan, node_states={}, overall_status=overall_status)


def _mock_index(monkeypatch, runs: dict[str, RunState]):
    summaries = {run_id: {"overall_status": r.overall_status} for run_id, r in runs.items()}
    monkeypatch.setattr("agents.common.run_store.list_run_summaries", lambda: summaries)
    monkeypatch.setattr("agents.common.run_store.load_run", lambda run_id: runs.get(run_id))


def test_poll_once_detects_newly_completed_runs(monkeypatch):
    _mock_index(monkeypatch, {"r1": _run("r1", "completed"), "r2": _run("r2", "running")})

    seen = set()
    found = watcher.poll_once(seen)

    assert [r.run_id for r in found] == ["r1"]
    assert seen == {"r1"}


def test_poll_once_treats_failed_and_circuit_broken_as_terminal(monkeypatch):
    _mock_index(monkeypatch, {"r1": _run("r1", "failed"), "r2": _run("r2", "circuit_broken")})

    found = watcher.poll_once(set())

    assert {r.run_id for r in found} == {"r1", "r2"}


def test_poll_once_does_not_re_notify_already_seen_runs(monkeypatch):
    _mock_index(monkeypatch, {"r1": _run("r1", "completed")})

    seen = {"r1"}
    found = watcher.poll_once(seen)

    assert found == []


def test_poll_once_skips_run_missing_between_index_and_load(monkeypatch):
    """The index says r1 is terminal, but its full file is gone by the time
    we go to load it (e.g. pruned concurrently) -- must not crash."""
    summaries = {"r1": {"overall_status": "completed"}}
    monkeypatch.setattr("agents.common.run_store.list_run_summaries", lambda: summaries)
    monkeypatch.setattr("agents.common.run_store.load_run", lambda run_id: None)

    found = watcher.poll_once(set())
    assert found == []


def test_load_and_save_seen_round_trip(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    watcher._save_seen({"run-a", "run-b"})

    reloaded = watcher._load_seen()
    assert reloaded == {"run-a", "run-b"}


def test_heartbeat_is_written_every_iteration(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    _mock_index(monkeypatch, {})

    watcher.poll_loop(lambda runs: None, interval_s=0, max_iterations=1)

    heartbeat = watcher._heartbeat_file()
    assert heartbeat.exists()
    import json

    data = json.loads(heartbeat.read_text())
    assert "last_poll_at" in data


def test_heartbeat_write_failure_does_not_crash_the_loop(monkeypatch):
    _mock_index(monkeypatch, {})
    monkeypatch.setattr(watcher, "_heartbeat_file", lambda: (_ for _ in ()).throw(OSError("disk full")))

    # Must not raise.
    watcher.poll_loop(lambda runs: None, interval_s=0, max_iterations=1)


def test_prune_runs_only_every_n_polls(monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "synthesizer_prune_every_n_polls", 3)
    _mock_index(monkeypatch, {})

    prune_calls = {"count": 0}
    monkeypatch.setattr(watcher, "_prune_old_data", lambda: prune_calls.__setitem__("count", prune_calls["count"] + 1))

    watcher.poll_loop(lambda runs: None, interval_s=0, max_iterations=6)

    assert prune_calls["count"] == 2  # fires on iteration 3 and 6
