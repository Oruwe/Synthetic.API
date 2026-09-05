"""Synthesizer entrypoint: a long-running process that polls Qdrant and
drafts + delivers a summary whenever new delayed-order points appear."""

from qdrant_client.http import models as qm

from agents.common.logging import configure_logging, get_logger
from agents.synthesizer import drafter, notifier, watcher

configure_logging("synthesizer")
logger = get_logger(component="synthesizer.main")


def _handle_new_orders(records: list[qm.Record]) -> None:
    by_run: dict[str, list[qm.Record]] = {}
    for record in records:
        by_run.setdefault(record.payload.get("run_id", "unknown"), []).append(record)

    for run_id, run_records in by_run.items():
        logger.info("drafting_summary", run_id=run_id, order_count=len(run_records))
        summary = drafter.draft_summary(run_records, run_id)
        notifier.notify(summary, run_id)


def main() -> None:
    watcher.poll_loop(_handle_new_orders)


if __name__ == "__main__":
    main()
