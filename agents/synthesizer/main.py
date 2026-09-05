"""Synthesizer entrypoint: a long-running process that watches for newly-
completed DAG runs (see watcher.py for why the trigger reads run_store
rather than diffing Qdrant) and drafts + delivers a cited answer for each,
retrieved via semantic search over that run's chunks.
"""

from agents.common import notifier, qdrant_store
from agents.common.logging import configure_logging, get_logger
from agents.common.models.dag import RunState
from agents.synthesizer import drafter, watcher

configure_logging("synthesizer")
logger = get_logger(component="synthesizer.main")


def _handle_completed_runs(runs: list[RunState]) -> None:
    for run in runs:
        question = run.plan.transcript.strip()
        fetch_node = next((n for n in run.plan.nodes if n.id == "fetch"), None)
        sources_attempted = len(fetch_node.params.get("search_results", [])) if fetch_node else 0

        logger.info("drafting_answer", run_id=run.run_id, overall_status=run.overall_status, question=question)
        chunks = qdrant_store.semantic_search_pages(run.run_id, question)
        sources_succeeded = len({c.payload.get("url") for c in chunks if c.payload and c.payload.get("url")})

        answer = drafter.draft_answer(
            chunks,
            run.run_id,
            question,
            sources_attempted=sources_attempted,
            sources_succeeded=sources_succeeded,
        )
        notifier.notify(answer, run.run_id)


def main() -> None:
    watcher.poll_loop(_handle_completed_runs)


if __name__ == "__main__":
    main()
