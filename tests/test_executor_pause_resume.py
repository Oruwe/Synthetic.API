"""Tests for the human-in-the-loop pause/resume machinery
(AwaitingHumanInputError, resume_plan) added to the DAG executor for the
gated-content path. A handler raises AwaitingHumanInputError to pause a
run; POST /runs/{run_id}/resume (main.py, tested separately) eventually
calls resume_plan() with what the human answered.

Security property under test, not just an implementation detail:
`password` must never land in the persisted RunState
(run.human_provided_inputs) -- only in the in-memory RunContext for the
one resume call that uses it. See PendingInputRequest's docstring
(agents/common/models/dag.py) for the full reasoning.
"""

import uuid
from datetime import datetime, timezone

from agents.common import run_store
from agents.common.models.dag import DAGEdge, DAGNode, DAGPlan, NodeStatus, NodeType
from agents.orchestrator import executor
from agents.orchestrator.executor import AwaitingHumanInputError, RunContext


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
    return DAGNode(id=node_id, type=NodeType.FETCH_PAGES, name=node_id, handler_key=handler_key, **kwargs)


def test_awaiting_human_input_pauses_the_run_without_retrying():
    key = f"gate_{uuid.uuid4().hex}"
    calls = {"count": 0}

    def handler(node, ctx):
        calls["count"] += 1
        raise AwaitingHumanInputError(fields=["email"], prompt="need an email", url="https://gated.test")

    executor.register_handler(key)(handler)
    plan = _plan([_node("fetch", key, max_retries=3)])

    run = executor.execute_plan(plan)

    assert run.overall_status == "awaiting_human_input"
    assert run.node_states["fetch"].status == NodeStatus.AWAITING_INPUT
    assert run.pending_input is not None
    assert run.pending_input.fields == ["email"]
    assert run.pending_input.url == "https://gated.test"
    assert run.pending_input.node_id == "fetch"
    assert calls["count"] == 1  # NOT retried across max_retries=3 attempts
    assert run.failure_count == 0  # never counts toward the circuit breaker


def test_resume_plan_re_attempts_the_paused_node_with_the_answer_available():
    key = f"gate_resume_{uuid.uuid4().hex}"

    def handler(node, ctx):
        email = ctx.data.get("human_provided_inputs", {}).get(node.id, {}).get("email")
        if not email:
            raise AwaitingHumanInputError(fields=["email"], prompt="need an email", url="https://gated.test")
        return f"got past the gate with {email}"

    executor.register_handler(key)(handler)
    plan = _plan([_node("fetch", key)])

    run = executor.execute_plan(plan)
    assert run.overall_status == "awaiting_human_input"

    resumed = executor.resume_plan(run.run_id, {"email": "judge@example.com"})

    assert resumed.overall_status == "completed"
    assert resumed.node_states["fetch"].status == NodeStatus.SUCCEEDED
    assert resumed.pending_input is None


def test_resume_plan_persists_email_but_never_password():
    """The core security property: password must never land in the
    persisted RunState, only email does."""
    key = f"gate_login_{uuid.uuid4().hex}"

    def handler(node, ctx):
        provided = ctx.data.get("human_provided_inputs", {}).get(node.id, {})
        if "password" not in provided:
            raise AwaitingHumanInputError(fields=["email", "password"], prompt="need login", url="https://gated.test")
        return "logged in"

    executor.register_handler(key)(handler)
    plan = _plan([_node("fetch", key)])
    run = executor.execute_plan(plan)

    resumed = executor.resume_plan(run.run_id, {"email": "judge@example.com", "password": "hunter2"})

    assert resumed.overall_status == "completed"
    assert resumed.human_provided_inputs == {"fetch": {"email": "judge@example.com"}}
    # the persisted RunState (what run_store actually wrote to disk) must
    # never contain the password string anywhere
    reloaded = run_store.load_run(run.run_id)
    assert "hunter2" not in reloaded.model_dump_json()


def test_resume_plan_raises_when_a_required_field_is_missing():
    key = f"gate_missing_{uuid.uuid4().hex}"
    executor.register_handler(key)(
        lambda node, ctx: (_ for _ in ()).throw(
            AwaitingHumanInputError(fields=["email"], prompt="need an email", url="https://gated.test")
        )
    )
    plan = _plan([_node("fetch", key)])
    run = executor.execute_plan(plan)

    try:
        executor.resume_plan(run.run_id, {})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "email" in str(exc)


def test_resume_plan_raises_on_a_run_that_is_not_awaiting_input():
    key = f"gate_notpaused_{uuid.uuid4().hex}"
    executor.register_handler(key)(lambda node, ctx: "done")
    plan = _plan([_node("fetch", key)])
    run = executor.execute_plan(plan)
    assert run.overall_status == "completed"

    try:
        executor.resume_plan(run.run_id, {"email": "x@example.com"})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resume_plan_raises_on_an_unknown_run_id():
    try:
        executor.resume_plan("does-not-exist", {"email": "x@example.com"})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_downstream_node_runs_normally_after_the_paused_node_resumes_successfully():
    gate_key = f"gate_chain_{uuid.uuid4().hex}"
    downstream_key = f"downstream_{uuid.uuid4().hex}"
    downstream_calls = []

    def gate_handler(node, ctx):
        email = ctx.data.get("human_provided_inputs", {}).get(node.id, {}).get("email")
        if not email:
            raise AwaitingHumanInputError(fields=["email"], prompt="need an email", url="https://gated.test")
        ctx.data["unlocked_content"] = f"real content, unlocked with {email}"
        return "unlocked"

    def downstream_handler(node, ctx):
        downstream_calls.append(ctx.data.get("unlocked_content"))
        return "processed"

    executor.register_handler(gate_key)(gate_handler)
    executor.register_handler(downstream_key)(downstream_handler)
    plan = _plan(
        [_node("fetch", gate_key), _node("embed", downstream_key, depends_on=["fetch"])],
        edges=[DAGEdge(from_node="fetch", to_node="embed")],
    )

    run = executor.execute_plan(plan)
    assert run.overall_status == "awaiting_human_input"
    assert run.node_states["embed"].status == NodeStatus.PENDING  # never started yet

    resumed = executor.resume_plan(run.run_id, {"email": "judge@example.com"})

    assert resumed.overall_status == "completed"
    assert resumed.node_states["embed"].status == NodeStatus.SUCCEEDED
    assert downstream_calls == ["real content, unlocked with judge@example.com"]
