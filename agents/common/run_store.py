"""Run-state persistence: one JSON file per run_id under RUN_STORE_DIR,
plus a lightweight index (_runs_index.json) for cheap status lookups.

Chosen over SQLite for zero-schema simplicity given the hackathon
timeline. The interface (create_run / update_node_state / load_run) is
small enough to swap to SQLite later without touching callers. Writes are
atomic (write to a temp file, then os.replace) so a crash mid-write never
leaves a corrupt/partial run file — this is what makes
`cat data/runs/<run_id>.json` a reliable way to inspect a run live.

The index exists because the Synthesizer's watcher (agents/synthesizer/
watcher.py) polls this directory every few seconds forever: without it,
`list_runs()` re-reads and re-parses every run file this process has ever
written, on every single poll -- fine at demo scale, a real scaling
problem after a day of real traffic. `list_run_summaries()` reads one
small index file instead; `load_run()` (a full file read) only happens
for run_ids the index says are newly terminal.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.dag import DAGPlan, NodeExecutionState, RunState

logger = get_logger(component="run_store")

_index_lock = threading.Lock()


def _run_path(run_id: str) -> Path:
    return Path(settings.run_store_dir) / f"{run_id}.json"


def _index_path() -> Path:
    return Path(settings.run_store_dir) / "_runs_index.json"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(content)
    os.replace(tmp_path, path)


def _read_index() -> dict[str, dict]:
    path = _index_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - a corrupt index must degrade, not crash every save
        logger.warning("runs_index_unreadable", error=str(exc))
        return {}


def _update_index(run: RunState) -> None:
    """Read-modify-write of the shared index, guarded by a lock: multiple
    runs' nodes can complete concurrently (the DAG executor runs handlers
    in a thread pool -- see agents/orchestrator/executor.py), and without
    this lock two concurrent updates could race and silently drop one
    run's status change from the index."""
    with _index_lock:
        index = _read_index()
        index[run.run_id] = {
            "overall_status": run.overall_status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_atomic(_index_path(), json.dumps(index, indent=2))


def _remove_from_index(run_id: str) -> None:
    with _index_lock:
        index = _read_index()
        if run_id in index:
            del index[run_id]
            _write_atomic(_index_path(), json.dumps(index, indent=2))


def create_run(plan: DAGPlan) -> RunState:
    node_states = {node.id: NodeExecutionState(node_id=node.id) for node in plan.nodes}
    overall_status = "no_capability" if plan.status == "no_capability" else "running"
    run = RunState(run_id=plan.run_id, plan=plan, node_states=node_states, overall_status=overall_status)
    _write_atomic(_run_path(plan.run_id), run.model_dump_json(indent=2))
    _update_index(run)
    return run


def save_run(run: RunState) -> None:
    # Every caller funnels through here (update_node_state, the answer
    # persist step in synthesizer/main.py, executor.py's terminal-status
    # writes), so this is the one place that can keep updated_at actually
    # meaning "last modified" -- it was previously stamped once by
    # RunState's own default_factory at construction and never touched
    # again, so every subsequent rewrite of a run file re-serialized the
    # same stale creation timestamp despite the file changing many times.
    run.updated_at = datetime.now(timezone.utc)
    _write_atomic(_run_path(run.run_id), run.model_dump_json(indent=2))
    _update_index(run)


def update_node_state(run: RunState, node_id: str, state: NodeExecutionState) -> RunState:
    run.node_states[node_id] = state
    save_run(run)
    return run


def load_run(run_id: str) -> RunState | None:
    path = _run_path(run_id)
    if not path.exists():
        return None
    return RunState.model_validate_json(path.read_text())


def list_run_summaries() -> dict[str, dict]:
    """Cheap: one small file read, not N. This is what the watcher's
    poll_once should use to decide WHICH runs are newly terminal before
    paying for a full load_run() on just those."""
    return _read_index()


def list_runs() -> list[RunState]:
    """Full listing (every run file, fully parsed) -- kept for callers that
    genuinely need complete run state for everything (an admin/debug CLI,
    tests), but NOT used by the watcher's hot poll loop anymore; prefer
    list_run_summaries() + load_run() for that. See module docstring.

    Files starting with "_" (the index, the watcher's seen-run tracking
    file) are skipped, as are any that fail to parse -- a corrupt/partial
    run file is logged and skipped rather than raised, consistent with
    every other per-item isolation in this codebase.
    """
    directory = Path(settings.run_store_dir)
    if not directory.exists():
        return []

    runs: list[RunState] = []
    for path in directory.glob("*.json"):
        if path.name.startswith("_"):
            continue
        try:
            runs.append(RunState.model_validate_json(path.read_text()))
        except Exception as exc:  # noqa: BLE001 - one bad file must not break the whole listing
            logger.warning("run_file_unreadable", path=str(path), error=str(exc))
            continue
    return runs


def prune_old_runs(max_age_hours: float | None = None) -> int:
    """Deletes run files (and their index entries) older than
    `max_age_hours` (by file mtime). Bounds the otherwise-unbounded growth
    of data/runs/ -- called periodically from the Synthesizer's poll loop
    (see watcher.py), not on every poll, since it's an O(all files) sweep.
    Returns the number of runs pruned. Never raises: a single bad file is
    logged and skipped, same as list_runs().
    """
    max_age_hours = max_age_hours if max_age_hours is not None else settings.run_retention_hours
    directory = Path(settings.run_store_dir)
    if not directory.exists():
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    pruned = 0
    for path in directory.glob("*.json"):
        if path.name.startswith("_"):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                run_id = path.stem
                path.unlink(missing_ok=True)
                path.with_suffix(".tmp").unlink(missing_ok=True)
                _remove_from_index(run_id)
                pruned += 1
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the sweep
            logger.warning("run_prune_failed", path=str(path), error=str(exc))
            continue

    if pruned:
        logger.info("runs_pruned", count=pruned, max_age_hours=max_age_hours)
    return pruned
