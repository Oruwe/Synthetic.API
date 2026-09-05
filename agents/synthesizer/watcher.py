"""Polls Qdrant for new points across BOTH collections this system
coordinates through, and dispatches them to a callback tagged by kind:
  - "delayed"  -> a shipping order (delayed_orders collection)
  - "research" -> a curated (status=permanent) web-research finding
                  (web_knowledge collection, after curate_knowledge has
                  already deleted the "majority junk" candidates)

This is the async coordination mechanism the hackathon brief asks for:
neither Web-Navigator pipeline calls the Synthesizer directly -- it wakes
up because new vectors landed in shared memory. `seen` point keys (per
collection) are persisted to disk so a container restart doesn't
re-notify on old points.
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

Kind = str  # "delayed" | "research"


def _seen_file() -> Path:
    return Path(settings.run_store_dir) / "_synthesizer_seen.json"


def _load_seen() -> dict[str, set[str]]:
    path = _seen_file()
    if path.exists():
        raw = json.loads(path.read_text())
        return {"delayed": set(raw.get("delayed", [])), "research": set(raw.get("research", []))}
    return {"delayed": set(), "research": set()}


def _save_seen(seen: dict[str, set[str]]) -> None:
    path = _seen_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({kind: sorted(ids) for kind, ids in seen.items()}))


def poll_once(seen: dict[str, set[str]]) -> list[tuple[Kind, qm.Record]]:
    found: list[tuple[Kind, qm.Record]] = []

    for record in qdrant_store.scroll_new_delayed(seen["delayed"]):
        seen["delayed"].add(record.payload["point_key"])
        found.append(("delayed", record))

    for record in qdrant_store.scroll_new_permanent_research(seen["research"]):
        seen["research"].add(record.payload["point_key"])
        found.append(("research", record))

    if found:
        _save_seen(seen)
    return found


def poll_loop(
    on_new_items: Callable[[list[tuple[Kind, qm.Record]]], None],
    interval_s: float | None = None,
    max_iterations: int | None = None,
) -> None:
    """Runs forever (or `max_iterations` times, for tests). `on_new_items`
    is called once per poll with any newly-seen (kind, record) pairs."""
    interval_s = interval_s if interval_s is not None else settings.synthesizer_poll_interval_seconds
    seen = _load_seen()
    logger.info(
        "watcher_started",
        interval_s=interval_s,
        already_seen_delayed=len(seen["delayed"]),
        already_seen_research=len(seen["research"]),
    )

    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        try:
            new_items = poll_once(seen)
            if new_items:
                logger.info("new_points_detected", count=len(new_items))
                on_new_items(new_items)
        except Exception as exc:  # noqa: BLE001 - one bad poll must not kill the loop
            logger.warning("watcher_poll_error", error=str(exc))

        iteration += 1
        if max_iterations is None or iteration < max_iterations:
            time.sleep(interval_s)
