"""Tests for the execute_action DAG node handler: replay-vs-explore
routing against the production memory layer (find_workflow_memory /
record_workflow_outcome), and stashing the result on RunContext for
executor.execute_plan to report synchronously. Both qdrant_store and
action_executor are mocked -- offline, deterministic.
"""

from datetime import datetime, timezone

from agents.common.models.action import ActionStep, ActionWorkflow, WorkflowMemory
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


def _memory():
    return WorkflowMemory(
        canonical_key="example.test:book a table",
        domain="example.test",
        representative_intent="book a table",
        start_url="https://example.test",
        steps=[ActionStep(kind="done", reasoning="done")],
        success_count=3,
        failure_count=0,
        created_at=datetime.now(timezone.utc),
        last_used_at=datetime.now(timezone.utc),
        last_success_at=datetime.now(timezone.utc),
    )


def test_explores_fresh_when_no_trusted_memory_exists(monkeypatch):
    monkeypatch.setattr(action_handlers.qdrant_store, "find_workflow_memory", lambda intent, start_url=None: None)
    explore_calls = []

    def fake_explore(intent, start_url, run_id):
        explore_calls.append((intent, start_url, run_id))
        return _workflow()

    monkeypatch.setattr(action_handlers.action_executor, "execute_action_loop", fake_explore)
    record_calls = []
    monkeypatch.setattr(action_handlers.qdrant_store, "record_workflow_outcome", lambda wf: record_calls.append(wf))

    ctx = RunContext(run_id="r1")
    result = action_handlers.handle_execute_action(_node(), ctx)

    assert result is True
    assert explore_calls == [("book a table", "https://example.test", "r1")]
    assert record_calls == [ctx.data["action_workflow"]]


def test_replays_when_a_trusted_memory_exists(monkeypatch):
    memory = _memory()
    monkeypatch.setattr(action_handlers.qdrant_store, "find_workflow_memory", lambda intent, start_url=None: memory)
    replay_calls = []

    def fake_replay(mem, run_id):
        replay_calls.append((mem, run_id))
        return _workflow()

    monkeypatch.setattr(action_handlers.action_executor, "replay_workflow", fake_replay)
    explore_calls = []
    monkeypatch.setattr(
        action_handlers.action_executor,
        "execute_action_loop",
        lambda *a, **k: explore_calls.append(1) or _workflow(),
    )
    monkeypatch.setattr(action_handlers.qdrant_store, "record_workflow_outcome", lambda wf: None)

    ctx = RunContext(run_id="r1")
    result = action_handlers.handle_execute_action(_node(), ctx)

    assert result is True
    assert replay_calls == [(memory, "r1")]
    assert explore_calls == []  # live exploration must not run when a trusted replay was available


def test_stashes_the_workflow_on_ctx_data_even_on_failure(monkeypatch):
    monkeypatch.setattr(action_handlers.qdrant_store, "find_workflow_memory", lambda intent, start_url=None: None)
    monkeypatch.setattr(
        action_handlers.action_executor, "execute_action_loop", lambda *a, **k: _workflow(success=False)
    )
    monkeypatch.setattr(action_handlers.qdrant_store, "record_workflow_outcome", lambda wf: None)

    ctx = RunContext(run_id="r1")
    result = action_handlers.handle_execute_action(_node(), ctx)

    assert result is False
    assert ctx.data["action_workflow"].success is False


def test_passes_start_url_to_the_memory_lookup_for_domain_filtering(monkeypatch):
    """find_workflow_memory needs start_url to apply its domain filter --
    a regression here would silently widen every replay match to any
    domain, which is exactly the cross-site-replay risk it exists to
    prevent."""
    captured = {}

    def fake_find(intent, start_url=None):
        captured["intent"] = intent
        captured["start_url"] = start_url
        return None

    monkeypatch.setattr(action_handlers.qdrant_store, "find_workflow_memory", fake_find)
    monkeypatch.setattr(action_handlers.action_executor, "execute_action_loop", lambda *a, **k: _workflow())
    monkeypatch.setattr(action_handlers.qdrant_store, "record_workflow_outcome", lambda wf: None)

    action_handlers.handle_execute_action(_node(intent="book a table", start_url="https://x.test"), RunContext(run_id="r1"))

    assert captured == {"intent": "book a table", "start_url": "https://x.test"}
