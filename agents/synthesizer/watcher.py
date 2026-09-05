"""Detects newly-completed DAG runs and dispatches them to a callback.

This is the async coordination mechanism the hackathon brief asks for:
the Orchestrator never calls the Synthesizer directly -- it wakes up on
its own schedule and notices work is ready.

The trigger itself reads run_store's shared run-state index (bind-mounted
into both the orchestrator and synthesizer containers, see
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

Uses run_store.list_run_summaries() (one small index file) rather than
list_runs() (every run file, fully parsed) -- this loop runs forever on a
fixed interval, so an O(all runs ever) read on every poll is a real
scaling problem after enough history accumulates; the index makes each
poll O(1) plus O(newly-terminal runs), which is normally 0 or 1.
"""

import json
import time
from pathlib import Path
from typing import Callable

from agents.common import qdrant_store, run_store
from agents.common.config import settings
from agents.common.logging import get_logger

logger = get_logger(component="watcher")

_TERMINAL_STATUSES = {"completed", "failed", "circuit_broken"}


def _seen_file() -> Path:
    return Path(settings.run_store_dir) / "_synthesizer_seen_runs.json"


def _heartbeat_file() -> Path:
    return Path(settings.run_store_dir) / "_synthesizer_heartbeat.json"


def _write_heartbeat() -> None:
    """The Synthesizer is a bare polling loop, not an HTTP server, so
    unlike the Orchestrator's /health it has no built-in way for Docker
    (or a human) to tell "running" from "hung." Written every iteration to
    the same shared run_store directory everything else already uses;
    docker-compose's healthcheck for this service checks the file's
    recency (see docker-compose.yml)."""
    try:
        path = _heartbeat_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"last_poll_at": time.time()}))
    except Exception as exc:  # noqa: BLE001 - a heartbeat write failing must not kill the loop
        logger.warning("heartbeat_write_failed", error=str(exc))


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
    for run_id, summary in run_store.list_run_summaries().items():
        if run_id in seen or summary.get("overall_status") not in _TERMINAL_STATUSES:
            continue
        run = run_store.load_run(run_id)
        if run is None:  # file vanished between the index read and now -- skip, don't crash
            continue
        seen.add(run_id)
        new_runs.append(run)
    if new_runs:
        _save_seen(seen)
    return new_runs


def _prune_old_data() -> None:
    """Bounds otherwise-unbounded growth of data/runs/*.json and the
    web_pages Qdrant collection. Called every N polls (not every poll --
    it's an O(all data) sweep), see settings.synthesizer_prune_every_n_polls.
    """
    try:
        runs_pruned = run_store.prune_old_runs()
        chunks_pruned = qdrant_store.prune_old_page_chunks()
        if runs_pruned or chunks_pruned:
            logger.info("retention_sweep_completed", runs_pruned=runs_pruned, chunks_pruned=chunks_pruned)
    except Exception as exc:  # noqa: BLE001 - a failed sweep must not kill the poll loop
        logger.warning("retention_sweep_failed", error=str(exc))


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
        _write_heartbeat()
        try:
            new_runs = poll_once(seen)
            if new_runs:
                logger.info("new_completed_runs_detected", count=len(new_runs))
                on_completed_runs(new_runs)
        except Exception as exc:  # noqa: BLE001 - one bad poll must not kill the loop
            logger.warning("watcher_poll_error", error=str(exc))

        iteration += 1
        if iteration % settings.synthesizer_prune_every_n_polls == 0:
            _prune_old_data()

        if max_iterations is None or iteration < max_iterations:
            time.sleep(interval_s)
