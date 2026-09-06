"""Tests for the execute_action DAG node handler: replay-vs-explore
routing, persisting the outcome to Qdrant either way, and stashing the
result on RunContext for executor.execute_plan to report synchronously.
Both qdrant_store and action_executor are mocked -- offline, deterministic.
"""

from datetime import datetime, timezone

from agents.common.models.action import ActionStep, ActionWorkflow
from agents.common.models.dag import DAGNode, NodeType
from agents.orchestrator.executor import RunContext
from agents.web_navigator import action_handlers


def _node(intent="book a table", start_url="https://example.test"):
    return DAGNode(
        id="act",
        type=NodeType.EXECUTE_ACTION,
        name="act",
        handler_key="execute_action",
        params={"intent": intent, "start_url": start_url},
    )


def _workflow(success=True):
    return ActionWorkflow(
        run_id="r1",
        intent="book a table",
        start_url="https://example.test",
        steps=[ActionStep(kind="done", reasoning="done")],
        success=success,
        refused_reason=None,
        created_at=datetime.now(timezone.utc),
    )


def test_explores_fresh_when_no_similar_workflow_exists(monkeypatch):
    monkeypatch.setattr(action_handlers.qdrant_store, "find_similar_workflow", lambda intent: None)
    explore_calls = []

    def fake_explore(intent, start_url, run_id):
        explore_calls.append((intent, start_url, run_id))
        return _workflow()

    monkeypatch.setattr(action_handlers.action_executor, "execute_action_loop", fake_explore)
    upsert_calls = []
    monkeypatch.setattr(action_handlers.qdrant_store, "upsert_action_workflow", lambda wf: upsert_calls.append(wf))

    ctx = RunContext(run_id="r1")
    result = action_handlers.handle_execute_action(_node(), ctx)

    assert result is True
    assert explore_calls == [("book a table", "https://example.test", "r1")]
    assert upsert_calls == [ctx.data["action_workflow"]]


def test_replays_when_a_similar_successful_workflow_exists(monkeypatch):
    prior = _workflow()
    monkeypatch.setattr(action_handlers.qdrant_store, "find_similar_workflow", lambda intent: prior)
    replay_calls = []

    def fake_replay(workflow, run_id):
        replay_calls.append((workflow, run_id))
        return _workflow()

    monkeypatch.setattr(action_handlers.action_executor, "replay_workflow", fake_replay)
    explore_calls = []
    monkeypatch.setattr(
        action_handlers.action_executor,
        "execute_action_loop",
        lambda *a, **k: explore_calls.append(1) or _workflow(),
    )
    monkeypatch.setattr(action_handlers.qdrant_store, "upsert_action_workflow", lambda wf: None)

    ctx = RunContext(run_id="r1")
    result = action_handlers.handle_execute_action(_node(), ctx)

    assert result is True
    assert replay_calls == [(prior, "r1")]
    assert explore_calls == []  # live exploration must not run when a replay was available


def test_stashes_the_workflow_on_ctx_data_even_on_failure(monkeypatch):
    monkeypatch.setattr(action_handlers.qdrant_store, "find_similar_workflow", lambda intent: None)
    monkeypatch.setattr(action_handlers.action_executor, "execute_action_loop", lambda *a, **k: _workflow(success=False))
    monkeypatch.setattr(action_handlers.qdrant_store, "upsert_action_workflow", lambda wf: None)

    ctx = RunContext(run_id="r1")
    result = action_handlers.handle_execute_action(_node(), ctx)

    assert result is False
    assert ctx.data["action_workflow"].success is False
