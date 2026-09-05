"""Tests for run_store.list_runs() -- the Synthesizer watcher's trigger
source. Corrupt files and the watcher's own seen-file must be skipped,
not raised."""

from datetime import datetime, timezone

from agents.common import run_store
from agents.common.config import settings
from agents.common.models.dag import DAGPlan, RunState


def _plan(run_id: str) -> DAGPlan:
    return DAGPlan(run_id=run_id, transcript="t", created_at=datetime.now(timezone.utc), nodes=[], edges=[])


def test_list_runs_returns_all_saved_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    run_store.create_run(_plan("r1"))
    run_store.create_run(_plan("r2"))

    runs = run_store.list_runs()
    assert {r.run_id for r in runs} == {"r1", "r2"}


def test_list_runs_skips_underscore_prefixed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    run_store.create_run(_plan("r1"))
    (tmp_path / "_synthesizer_seen_runs.json").write_text('["r1"]')

    runs = run_store.list_runs()
    assert [r.run_id for r in runs] == ["r1"]


def test_list_runs_skips_corrupt_files_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    run_store.create_run(_plan("r1"))
    (tmp_path / "corrupt.json").write_text("{not valid json")

    runs = run_store.list_runs()
    assert [r.run_id for r in runs] == ["r1"]


def test_list_runs_on_missing_directory_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path / "does_not_exist"))
    assert run_store.list_runs() == []


# --- Index (list_run_summaries) -------------------------------------------


def test_create_run_populates_the_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    run_store.create_run(_plan("r1"))

    summaries = run_store.list_run_summaries()
    assert summaries["r1"]["overall_status"] == "running"


def test_save_run_updates_the_index_status(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    run = run_store.create_run(_plan("r1"))
    run.overall_status = "completed"
    run_store.save_run(run)

    summaries = run_store.list_run_summaries()
    assert summaries["r1"]["overall_status"] == "completed"


def test_index_is_cheap_and_does_not_require_reading_run_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    for i in range(5):
        run_store.create_run(_plan(f"r{i}"))

    summaries = run_store.list_run_summaries()
    assert set(summaries.keys()) == {f"r{i}" for i in range(5)}


def test_index_survives_corrupt_index_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    (tmp_path / "_runs_index.json").write_text("{not valid json")

    # Must not raise -- degrades to an empty index rather than crashing a save.
    run_store.create_run(_plan("r1"))
    summaries = run_store.list_run_summaries()
    assert "r1" in summaries


def test_concurrent_index_updates_are_not_lost(tmp_path, monkeypatch):
    """Multiple runs' nodes can complete concurrently (the DAG executor's
    thread pool) -- the index's read-modify-write must be locked, or a
    race between two concurrent updates could silently drop one."""
    import threading

    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))

    def _create(i):
        run_store.create_run(_plan(f"r{i}"))

    threads = [threading.Thread(target=_create, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summaries = run_store.list_run_summaries()
    assert set(summaries.keys()) == {f"r{i}" for i in range(20)}


# --- Retention / pruning ----------------------------------------------------


def test_prune_old_runs_deletes_files_past_the_cutoff(tmp_path, monkeypatch):
    import os
    import time

    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    run_store.create_run(_plan("old"))
    run_store.create_run(_plan("new"))

    old_path = tmp_path / "old.json"
    old_time = time.time() - (48 * 3600)
    os.utime(old_path, (old_time, old_time))

    pruned = run_store.prune_old_runs(max_age_hours=24)

    assert pruned == 1
    assert run_store.load_run("old") is None
    assert run_store.load_run("new") is not None
    assert "old" not in run_store.list_run_summaries()
    assert "new" in run_store.list_run_summaries()


def test_prune_old_runs_returns_zero_when_nothing_is_old(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    run_store.create_run(_plan("fresh"))

    assert run_store.prune_old_runs(max_age_hours=24) == 0


def test_prune_old_runs_on_missing_directory_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path / "does_not_exist"))
    assert run_store.prune_old_runs(max_age_hours=24) == 0
