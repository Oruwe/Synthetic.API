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
from agents.common.run_store import _write_atomic

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
        # _write_atomic (write-to-tmp + os.replace), not a plain
        # write_text(): the docker-compose healthcheck reads and
        # json.loads() this file every 10s from another process, so a
        # direct write_text() risks the same torn-read class of bug
        # _load_seen()'s docstring above describes for the seen-runs file
        # -- a healthcheck read landing mid-write would see a truncated
        # file and raise, flapping the container's health status.
        _write_atomic(_heartbeat_file(), json.dumps({"last_poll_at": time.time()}))
    except Exception as exc:  # noqa: BLE001 - a heartbeat write failing must not kill the loop
        logger.warning("heartbeat_write_failed", error=str(exc))


def _load_seen() -> set[str]:
    path = _seen_file()
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception as exc:  # noqa: BLE001 - a corrupt seen-file must degrade, not crash-loop the Synthesizer
        # A truncated write here (process killed mid-write) used to crash
        # the whole Synthesizer on every restart, since poll_loop() called
        # this with no try/except of its own -- unlike every other
        # persistence path in this codebase (run_store._write_atomic is
        # used precisely to avoid this class of bug; _save_seen below now
        # reuses it for the same reason).
        logger.warning("seen_runs_file_corrupt_starting_fresh", error=str(exc))
        return set()


def _save_seen(seen: set[str]) -> None:
    _write_atomic(_seen_file(), json.dumps(sorted(seen)))


def poll_once(seen: set[str]) -> list[run_store.RunState]:
    """Returns newly-terminal runs not yet in `seen` -- does NOT mark them
    seen itself. That's poll_loop()'s job, done only once on_completed_runs
    has actually processed them: marking a run "seen" before it's actually
    been handled means a crash (or any exception raised by the callback)
    between this call and the callback finishing permanently skips that
    run's answer -- the exact "silently never gets an answer" failure mode
    the seen-file is supposed to protect against re-triggering, not cause.
    """
    new_runs: list[run_store.RunState] = []
    for run_id, summary in run_store.list_run_summaries().items():
        if run_id in seen or summary.get("overall_status") not in _TERMINAL_STATUSES:
            continue
        run = run_store.load_run(run_id)
        if run is None:  # file vanished between the index read and now -- skip, don't crash
            continue
        new_runs.append(run)
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
                # Marked seen only AFTER the callback returns -- if it
                # raises partway through (a systemic failure; per-run
                # failures are already caught inside
                # synthesizer/main.py's _handle_completed_runs and never
                # reach here), none of this batch is marked seen and all
                # of it is retried next poll. Reprocessing an
                # already-successfully-handled run wastes one redraft;
                # the alternative (the old behavior) was silently losing
                # it forever, a strictly worse failure mode.
                seen.update(run.run_id for run in new_runs)
                _save_seen(seen)
        except Exception as exc:  # noqa: BLE001 - one bad poll must not kill the loop
            logger.warning("watcher_poll_error", error=str(exc))

        iteration += 1
        # settings.synthesizer_prune_every_n_polls == 0 previously raised
        # ZeroDivisionError here -- uncaught (this line sits after the
        # try/except above), killing the whole poll loop on a
        # misconfiguration that every other failure path in this file
        # degrades gracefully from instead. Treat 0 as "never prune".
        if settings.synthesizer_prune_every_n_polls > 0 and iteration % settings.synthesizer_prune_every_n_polls == 0:
            _prune_old_data()

        if max_iterations is None or iteration < max_iterations:
            time.sleep(interval_s)
