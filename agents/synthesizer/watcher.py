"""Polls Qdrant for new "delayed" points and dispatches them to a callback.

This is the async coordination mechanism the hackathon brief asks for:
the Synthesizer is never called directly by the Web-Navigator — it wakes
up because new vectors landed in shared memory. `seen` point keys are
persisted to disk so a container restart doesn't re-notify on old points.
"""

import json
import time
from pathlib import Path
from typing import Callable

from qdrant_client.http import models as qm

from agents.common import qdrant_store
from agents.common.config import settings
from agents.common.logging import get_logger

logger = get_logger(component="watcher")


def _seen_file() -> Path:
    return Path(settings.run_store_dir) / "_synthesizer_seen.json"


def _load_seen() -> set[str]:
    path = _seen_file()
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_seen(seen: set[str]) -> None:
    path = _seen_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(seen)))


def poll_once(seen: set[str]) -> list[qm.Record]:
    new_records = qdrant_store.scroll_new_delayed(seen)
    for record in new_records:
        seen.add(record.payload["point_key"])
    if new_records:
        _save_seen(seen)
    return new_records


def poll_loop(
    on_new_orders: Callable[[list[qm.Record]], None],
    interval_s: float | None = None,
    max_iterations: int | None = None,
) -> None:
    """Runs forever (or `max_iterations` times, for tests). `on_new_orders`
    is called once per poll with any newly-seen delayed-order points."""
    interval_s = interval_s if interval_s is not None else settings.synthesizer_poll_interval_seconds
    seen = _load_seen()
    logger.info("watcher_started", interval_s=interval_s, already_seen=len(seen))

    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        try:
            new_records = poll_once(seen)
            if new_records:
                logger.info("new_points_detected", count=len(new_records))
                on_new_orders(new_records)
        except Exception as exc:  # noqa: BLE001 - one bad poll must not kill the loop
            logger.warning("watcher_poll_error", error=str(exc))

        iteration += 1
        if max_iterations is None or iteration < max_iterations:
            time.sleep(interval_s)
