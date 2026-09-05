"""Synthesizer entrypoint: a long-running process that polls Qdrant across
both collections this system coordinates through and drafts + delivers a
response whenever new points appear -- a shipping-order summary for
"delayed" points, a cited research answer for "research" (curated
permanent) points.
"""

from qdrant_client.http import models as qm

from agents.common.logging import configure_logging, get_logger
from agents.synthesizer import drafter, notifier, watcher

configure_logging("synthesizer")
logger = get_logger(component="synthesizer.main")


def _handle_new_items(items: list[tuple[str, qm.Record]]) -> None:
    delayed_by_run: dict[str, list[qm.Record]] = {}
    research_by_run: dict[str, tuple[str, list[qm.Record]]] = {}

    for kind, record in items:
        run_id = record.payload.get("run_id", "unknown")
        if kind == "delayed":
            delayed_by_run.setdefault(run_id, []).append(record)
        elif kind == "research":
            query = record.payload.get("query", "")
            _, records = research_by_run.setdefault(run_id, (query, []))
            records.append(record)

    for run_id, run_records in delayed_by_run.items():
        logger.info("drafting_summary", run_id=run_id, order_count=len(run_records))
        summary = drafter.draft_summary(run_records, run_id)
        notifier.notify(summary, run_id)

    for run_id, (query, run_records) in research_by_run.items():
        logger.info("drafting_research_answer", run_id=run_id, finding_count=len(run_records), query=query)
        answer = drafter.draft_research_answer(run_records, run_id, query)
        notifier.notify(answer, run_id)


def main() -> None:
    watcher.poll_loop(_handle_new_items)


if __name__ == "__main__":
    main()
