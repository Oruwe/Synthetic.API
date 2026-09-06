"""Synthesizer entrypoint: a long-running process that watches for newly-
completed DAG runs (see watcher.py for why the trigger reads run_store
rather than diffing Qdrant) and drafts + delivers a cited answer for each,
retrieved via semantic search over that run's chunks.
"""

from agents.common import notifier, qdrant_store, run_store
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

        # Persist the answer onto the run itself so GET /runs/{run_id} can
        # hand it back directly -- previously the only place it appeared
        # was this process's stdout/logs (see notifier.notify above).
        # Best-effort: a save failure here must not re-raise and abort the
        # notification that already succeeded above, and re-reading the
        # run guards against overwriting a newer save from a concurrent
        # writer (there isn't one today, but load-before-save costs nothing).
        try:
            fresh = run_store.load_run(run.run_id) or run
            fresh.answer = answer
            run_store.save_run(fresh)
        except Exception as exc:  # noqa: BLE001 - the answer is already delivered via notify()
            logger.warning("answer_persist_failed", run_id=run.run_id, error=str(exc))


def main() -> None:
    watcher.poll_loop(_handle_completed_runs)


if __name__ == "__main__":
    main()
