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
