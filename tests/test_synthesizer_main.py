"""Tests that _handle_completed_runs persists the drafted answer (and its
structured sources) onto the run itself, not just to notifier.notify()'s
stdout/logs -- without this, GET /runs/{run_id} could never return the
answer an API caller actually wants back.

draft_answer() returns a DraftedAnswer, not a bare string -- see
drafter.py's DraftedAnswer / tests/test_drafter_page_answers.py."""

from datetime import datetime, timezone

from agents.common import run_store
from agents.common.config import settings
from agents.common.models.dag import DAGNode, DAGPlan, NodeType
from agents.common.models.page import Source
from agents.synthesizer import drafter, main as synthesizer_main


def _plan_with_fetch_node(run_id: str) -> DAGPlan:
    node = DAGNode(
        id="fetch", type=NodeType.FETCH_PAGES, name="fetch", handler_key="fetch_pages",
        params={"search_results": [{"title": "t", "url": "https://example.test"}]},
    )
    return DAGPlan(run_id=run_id, transcript="What is X?", created_at=datetime.now(timezone.utc), nodes=[node], edges=[])


def _drafted(text="the drafted answer", sources=None, attempted=1, succeeded=1):
    return drafter.DraftedAnswer(
        text=text,
        full=f"{text}\n\nSources used: https://example.test",
        sources=sources if sources is not None else [Source(url="https://example.test", title="t")],
        sources_attempted=attempted,
        sources_succeeded=succeeded,
    )


def test_handle_completed_runs_persists_answer_onto_run_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    monkeypatch.setattr(synthesizer_main.qdrant_store, "semantic_search_pages", lambda run_id, question: [])
    monkeypatch.setattr(synthesizer_main.drafter, "draft_answer", lambda *a, **kw: _drafted())
    monkeypatch.setattr(synthesizer_main.notifier, "notify", lambda summary, run_id: None)

    run = run_store.create_run(_plan_with_fetch_node("r1"))
    run.overall_status = "completed"
    run_store.save_run(run)

    synthesizer_main._handle_completed_runs([run])

    persisted = run_store.load_run("r1")
    assert persisted.answer == "the drafted answer\n\nSources used: https://example.test"
    assert persisted.answer_text == "the drafted answer"
    assert persisted.sources == [Source(url="https://example.test", title="t")]
    assert persisted.sources_attempted == 1
    assert persisted.sources_succeeded == 1
    assert persisted.overall_status == "completed"  # untouched by the answer save


def test_handle_completed_runs_notifies_with_the_full_backward_compatible_string(tmp_path, monkeypatch):
    """notifier.notify() (stdout/logs/webhook) must keep receiving the full
    flattened string, unchanged from before this field split -- only the
    persisted RunState gained new structured fields."""
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    monkeypatch.setattr(synthesizer_main.qdrant_store, "semantic_search_pages", lambda run_id, question: [])
    monkeypatch.setattr(synthesizer_main.drafter, "draft_answer", lambda *a, **kw: _drafted())
    notified = []
    monkeypatch.setattr(synthesizer_main.notifier, "notify", lambda summary, run_id: notified.append(summary))

    run = run_store.create_run(_plan_with_fetch_node("r1b"))
    run.overall_status = "completed"
    run_store.save_run(run)

    synthesizer_main._handle_completed_runs([run])

    assert notified == ["the drafted answer\n\nSources used: https://example.test"]


def test_handle_completed_runs_survives_a_failed_persist(tmp_path, monkeypatch, caplog):
    """notify() already delivered the answer by the time the persist step
    runs -- a save failure there must be logged, not raised back into the
    watcher's poll loop."""
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    monkeypatch.setattr(synthesizer_main.qdrant_store, "semantic_search_pages", lambda run_id, question: [])
    monkeypatch.setattr(synthesizer_main.drafter, "draft_answer", lambda *a, **kw: _drafted())
    notified = []
    monkeypatch.setattr(synthesizer_main.notifier, "notify", lambda summary, run_id: notified.append(summary))

    def _broken_load_run(run_id):
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(synthesizer_main.run_store, "load_run", _broken_load_run)

    run = _plan_with_fetch_node("r2")
    run_state = run_store.create_run(run)

    synthesizer_main._handle_completed_runs([run_state])  # must not raise

    assert notified == ["the drafted answer\n\nSources used: https://example.test"]
