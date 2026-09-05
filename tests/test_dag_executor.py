"""Tests for the DAG executor: happy path, retry, exhausted retries,
timeout, circuit breaker, run-state persistence, and cycle rejection.

Each test registers its own uniquely-named handlers so tests never
interfere with each other via the shared HANDLER_REGISTRY.
"""

import time
import uuid
from datetime import datetime, timezone

import pytest

from agents.common import run_store
from agents.common.models.dag import DAGEdge, DAGNode, DAGPlan, NodeStatus, NodeType, PlanValidationError
from agents.orchestrator import executor


def _plan(nodes, edges=None, **kwargs) -> DAGPlan:
    return DAGPlan(
        run_id=str(uuid.uuid4()),
        transcript="test transcript",
        created_at=datetime.now(timezone.utc),
        nodes=nodes,
        edges=edges or [],
        **kwargs,
    )


def _node(node_id, handler_key, **kwargs) -> DAGNode:
    return DAGNode(id=node_id, type=NodeType.SCRAPE_PORTAL, name=node_id, handler_key=handler_key, **kwargs)


def test_happy_path_all_succeed():
    key = f"ok_{uuid.uuid4().hex}"
    executor.register_handler(key)(lambda node, ctx: "done")

    plan = _plan([_node("a", key), _node("b", key, depends_on=["a"])], [DAGEdge(from_node="a", to_node="b")])
    run = executor.execute_plan(plan)

    assert run.overall_status == "completed"
    assert run.node_states["a"].status == NodeStatus.SUCCEEDED
    assert run.node_states["b"].status == NodeStatus.SUCCEEDED


def test_retry_then_succeed_counts_attempts():
    key = f"flaky_{uuid.uuid4().hex}"
    calls = {"n": 0}

    def handler(node, ctx):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient failure")
        return "ok"

    executor.register_handler(key)(handler)
    plan = _plan([_node("a", key, max_retries=3, retry_backoff_seconds=0)])
    run = executor.execute_plan(plan)

    assert run.overall_status == "completed"
    assert run.node_states["a"].attempts == 2


def test_max_retries_exhausted_marks_node_failed():
    key = f"always_fail_{uuid.uuid4().hex}"
    executor.register_handler(key)(lambda node, ctx: (_ for _ in ()).throw(RuntimeError("nope")))

    plan = _plan([_node("a", key, max_retries=2, retry_backoff_seconds=0)])
    run = executor.execute_plan(plan)

    assert run.node_states["a"].status == NodeStatus.FAILED
    assert run.node_states["a"].attempts == 2
    assert run.overall_status == "failed"


def test_timeout_triggers_failure():
    key = f"slow_{uuid.uuid4().hex}"
    executor.register_handler(key)(lambda node, ctx: time.sleep(2))

    plan = _plan([_node("a", key, timeout_seconds=1, max_retries=1, retry_backoff_seconds=0)])
    run = executor.execute_plan(plan)

    assert run.node_states["a"].status == NodeStatus.FAILED
    assert "timed out" in run.node_states["a"].last_error


def test_circuit_breaker_halts_remaining_nodes():
    key = f"boom_{uuid.uuid4().hex}"
    executor.register_handler(key)(lambda node, ctx: (_ for _ in ()).throw(RuntimeError("boom")))

    nodes = [_node(f"n{i}", key, max_retries=1, retry_backoff_seconds=0) for i in range(6)]
    plan = _plan(nodes, circuit_breaker_threshold=5)
    run = executor.execute_plan(plan)

    assert run.overall_status == "circuit_broken"
    assert sum(1 for s in run.node_states.values() if s.status == NodeStatus.FAILED) == 5
    assert run.node_states["n5"].status == NodeStatus.SKIPPED


def test_downstream_node_skipped_when_dependency_fails():
    fail_key = f"fail_{uuid.uuid4().hex}"
    ok_key = f"ok_{uuid.uuid4().hex}"
    executor.register_handler(fail_key)(lambda node, ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    executor.register_handler(ok_key)(lambda node, ctx: "should not run")

    plan = _plan(
        [_node("a", fail_key, max_retries=1, retry_backoff_seconds=0), _node("b", ok_key, depends_on=["a"])],
        [DAGEdge(from_node="a", to_node="b")],
        circuit_breaker_threshold=99,
    )
    run = executor.execute_plan(plan)

    assert run.node_states["a"].status == NodeStatus.FAILED
    assert run.node_states["b"].status == NodeStatus.SKIPPED


def test_run_state_persisted_and_reloadable():
    key = f"ok_{uuid.uuid4().hex}"
    executor.register_handler(key)(lambda node, ctx: "done")

    plan = _plan([_node("a", key)])
    run = executor.execute_plan(plan)

    reloaded = run_store.load_run(plan.run_id)
    assert reloaded is not None
    assert reloaded.overall_status == "completed"
    assert reloaded.node_states["a"].status == NodeStatus.SUCCEEDED


def test_invalid_plan_with_cycle_is_rejected():
    key = f"ok_{uuid.uuid4().hex}"
    executor.register_handler(key)(lambda node, ctx: "done")

    plan = _plan(
        [_node("x", key, depends_on=["y"]), _node("y", key, depends_on=["x"])],
    )
    with pytest.raises(PlanValidationError):
        executor.execute_plan(plan)


def test_plan_with_unknown_dependency_is_rejected():
    key = f"ok_{uuid.uuid4().hex}"
    executor.register_handler(key)(lambda node, ctx: "done")

    plan = _plan([_node("a", key, depends_on=["ghost"])])
    with pytest.raises(PlanValidationError):
        executor.execute_plan(plan)
