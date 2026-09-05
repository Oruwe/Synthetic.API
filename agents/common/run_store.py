"""Run-state persistence: one JSON file per run_id under RUN_STORE_DIR.

Chosen over SQLite for zero-schema simplicity given the hackathon
timeline. The interface (create_run / update_node_state / load_run) is
small enough to swap to SQLite later without touching callers. Writes are
atomic (write to a temp file, then os.replace) so a crash mid-write never
leaves a corrupt/partial run file — this is what makes
`cat data/runs/<run_id>.json` a reliable way to inspect a run live.
"""

import json
import os
from pathlib import Path

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.dag import DAGPlan, NodeExecutionState, RunState

logger = get_logger(component="run_store")


def _run_path(run_id: str) -> Path:
    return Path(settings.run_store_dir) / f"{run_id}.json"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(content)
    os.replace(tmp_path, path)


def create_run(plan: DAGPlan) -> RunState:
    node_states = {node.id: NodeExecutionState(node_id=node.id) for node in plan.nodes}
    overall_status = "no_capability" if plan.status == "no_capability" else "running"
    run = RunState(run_id=plan.run_id, plan=plan, node_states=node_states, overall_status=overall_status)
    _write_atomic(_run_path(plan.run_id), run.model_dump_json(indent=2))
    return run


def save_run(run: RunState) -> None:
    _write_atomic(_run_path(run.run_id), run.model_dump_json(indent=2))


def update_node_state(run: RunState, node_id: str, state: NodeExecutionState) -> RunState:
    run.node_states[node_id] = state
    save_run(run)
    return run


def load_run(run_id: str) -> RunState | None:
    path = _run_path(run_id)
    if not path.exists():
        return None
    return RunState.model_validate_json(path.read_text())


def list_runs() -> list[RunState]:
    """Used by the Synthesizer's watcher (agents/synthesizer/watcher.py) to
    find newly-completed runs -- this directory is bind-mounted into both
    the orchestrator and synthesizer containers (see docker-compose.yml),
    so it's a reliable, already-shared place to read "is this run's data
    actually all there yet" from, rather than inferring it from Qdrant
    writes landing mid-batch.

    Files starting with "_" (e.g. the watcher's own seen-run tracking file)
    are skipped, as are any that fail to parse -- a corrupt/partial run
    file is logged and skipped rather than raised, consistent with every
    other per-item isolation in this codebase.
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
