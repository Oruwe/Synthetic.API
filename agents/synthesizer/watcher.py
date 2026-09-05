"""Detects newly-completed DAG runs and dispatches them to a callback.

This is the async coordination mechanism the hackathon brief asks for:
the Orchestrator never calls the Synthesizer directly -- it wakes up on
its own schedule and notices work is ready.

The trigger itself reads run_store's shared run-state directory (bind-
mounted into both the orchestrator and synthesizer containers, see
docker-compose.yml) rather than diffing a Qdrant scroll: a run's
`overall_status` only becomes terminal (completed/failed/circuit_broken)
once the executor has finished every node, so this is a strictly more
reliable "is this run's data actually all there" signal than "did I see a
new point," which could in principle fire between two upserts inside a
single embed_pages call and hand the Synthesizer a partial chunk set.
Qdrant remains the actual content/coordination layer -- semantic_search_pages()
still does the real read once a run is confirmed ready; this only decides
WHEN to read from it. `seen` run_ids are persisted to disk so a container
restart doesn't re-notify on old runs.
"""

import json
import time
from pathlib import Path
from typing import Callable

from agents.common import run_store
from agents.common.config import settings
from agents.common.logging import get_logger

logger = get_logger(component="watcher")

_TERMINAL_STATUSES = {"completed", "failed", "circuit_broken"}


def _seen_file() -> Path:
    return Path(settings.run_store_dir) / "_synthesizer_seen_runs.json"


def _load_seen() -> set[str]:
    path = _seen_file()
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_seen(seen: set[str]) -> None:
    path = _seen_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(seen)))


def poll_once(seen: set[str]) -> list[run_store.RunState]:
    new_runs: list[run_store.RunState] = []
    for run in run_store.list_runs():
        if run.run_id in seen or run.overall_status not in _TERMINAL_STATUSES:
            continue
        seen.add(run.run_id)
        new_runs.append(run)
    if new_runs:
        _save_seen(seen)
    return new_runs


def poll_loop(
    on_completed_runs: Callable[[list[run_store.RunState]], None],
    interval_s: float | None = None,
    max_iterations: int | None = None,
) -> None:
    """Runs forever (or `max_iterations` times, for tests). `on_completed_runs`
    is called once per poll with any newly-completed runs."""
    interval_s = interval_s if interval_s is not None else settings.synthesizer_poll_interval_seconds
    seen = _load_seen()
    logger.info("watcher_started", interval_s=interval_s, already_seen=len(seen))

    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        try:
            new_runs = poll_once(seen)
            if new_runs:
                logger.info("new_completed_runs_detected", count=len(new_runs))
                on_completed_runs(new_runs)
        except Exception as exc:  # noqa: BLE001 - one bad poll must not kill the loop
            logger.warning("watcher_poll_error", error=str(exc))

        iteration += 1
        if max_iterations is None or iteration < max_iterations:
            time.sleep(interval_s)
