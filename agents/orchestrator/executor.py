"""DAG executor: topological execution with per-node timeout, retry with
backoff, write-through run-state persistence, and a circuit breaker.

Deliberately NOT Airflow/Temporal — a hand-rolled executor over a small
in-memory graph is enough for a handful of nodes per run, and networkx is
used only for topological ordering + cycle detection, not as a runtime.
"""

import contextvars
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import networkx as nx

from agents.common import notifier, run_store
from agents.common.logging import bind_run_context, clear_run_context, get_logger
from agents.common.models.action import ActionWorkflow
from agents.common.models.dag import (
    DAGNode,
    DAGPlan,
    NodeExecutionState,
    NodeStatus,
    NodeType,
    PlanValidationError,
    RunState,
)

logger = get_logger(component="executor")


@dataclass
class RunContext:
    """Mutable scratch space handlers use to pass data to their dependents
    (e.g. the scrape handler's rows -> the extract handler's input)."""

    run_id: str
    data: dict[str, Any] = field(default_factory=dict)


HandlerFn = Callable[[DAGNode, RunContext], Any]
HANDLER_REGISTRY: dict[str, HandlerFn] = {}


def register_handler(handler_key: str) -> Callable[[HandlerFn], HandlerFn]:
    def decorator(fn: HandlerFn) -> HandlerFn:
        HANDLER_REGISTRY[handler_key] = fn
        return fn

    return decorator


def _build_graph(plan: DAGPlan) -> nx.DiGraph:
    graph = nx.DiGraph()
    node_ids = {n.id for n in plan.nodes}
    for node in plan.nodes:
        graph.add_node(node.id)
    for node in plan.nodes:
        for dep in node.depends_on:
            if dep not in node_ids:
                raise PlanValidationError(f"node '{node.id}' depends on unknown node '{dep}'")
            graph.add_edge(dep, node.id)
    if not nx.is_directed_acyclic_graph(graph):
        raise PlanValidationError(f"plan for run {plan.run_id} is not a DAG (cycle detected)")
    return graph


_thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dag-node")


def execute_plan(plan: DAGPlan) -> RunState:
    bind_run_context(run_id=plan.run_id)
    try:
        if plan.status == "no_capability":
            # Actually run the single clarify node (previously this branch
            # returned before ever executing it, leaving the handler dead
            # code) AND notify synchronously: a no_capability plan never
            # writes a Qdrant point, so the Synthesizer's poll loop would
            # never see it -- without this, the user gets silence instead
            # of an answer when they ask for something unsupported.
            run = run_store.create_run(plan)
            ctx = RunContext(run_id=plan.run_id)
            if plan.nodes:
                _run_node_with_retry(run, plan.nodes[0], ctx)
            run.overall_status = "no_capability"
            # Persist the answer onto the run itself, not just notifier.notify()'s
            # logs/webhook -- a no_capability run is never picked up by the
            # Synthesizer (it never writes a Qdrant point, and watcher.py's
            # _TERMINAL_STATUSES doesn't include "no_capability"), so without
            # this, GET /runs/{run_id} would return answer=None forever and a
            # polling client (e.g. ui/app.py) would wait the full timeout for
            # an answer that was actually ready immediately.
            message = (
                f'I couldn\'t find a supported action for: "{plan.transcript}". '
                "Supported right now: checking the shipping portal for delayed orders, "
                "and web research queries."
            )
            run.answer = message
            run.answer_text = message
            run_store.save_run(run)
            notifier.notify(message, plan.run_id)
            logger.info("plan_no_capability_notified", transcript=plan.transcript)
            return run

        graph = _build_graph(plan)
        run = run_store.create_run(plan)
        ctx = RunContext(run_id=plan.run_id)
        node_by_id = {n.id: n for n in plan.nodes}

        for node_id in nx.topological_sort(graph):
            node = node_by_id[node_id]

            deps_ok = all(
                run.node_states[dep].status == NodeStatus.SUCCEEDED for dep in graph.predecessors(node_id)
            )
            if not deps_ok:
                _transition(run, node_id, NodeStatus.SKIPPED, last_error="upstream dependency did not succeed")
                continue

            if run.failure_count >= plan.circuit_breaker_threshold:
                _skip_remaining(run, graph, node_id)
                run.overall_status = "circuit_broken"
                run_store.save_run(run)
                logger.error("circuit_breaker_tripped", failure_count=run.failure_count, node_id=node_id)
                return run

            _run_node_with_retry(run, node, ctx)

            if run.failure_count >= plan.circuit_breaker_threshold:
                remaining = list(nx.topological_sort(graph))
                idx = remaining.index(node_id)
                _skip_remaining_from(run, remaining[idx + 1 :])
                run.overall_status = "circuit_broken"
                run_store.save_run(run)
                logger.error("circuit_breaker_tripped", failure_count=run.failure_count, node_id=node_id)
                return run

        run.overall_status = "failed" if any(
            s.status == NodeStatus.FAILED for s in run.node_states.values()
        ) else "completed"

        # Ambient RPA action path reports synchronously here, unlike the
        # live research path (fetch_pages/embed_pages), which only writes
        # Qdrant chunks and leaves drafting the final answer to the
        # Synthesizer's separate async poll loop. An action run has no LLM
        # drafting step -- the outcome is a deterministic step sequence,
        # not text to summarize -- so there's nothing for the Synthesizer
        # to do, and waiting on its poll interval would just add latency
        # for no benefit.
        if any(n.type == NodeType.EXECUTE_ACTION for n in plan.nodes):
            workflow = ctx.data.get("action_workflow")
            run.action_workflow = workflow
            message = _compose_action_answer(plan.transcript, workflow, run.overall_status)
            run.answer = message
            run.answer_text = message
            notifier.notify(message, plan.run_id)

        run_store.save_run(run)
        logger.info("run_finished", overall_status=run.overall_status)
        return run
    finally:
        clear_run_context()


def _compose_action_answer(transcript: str, workflow: ActionWorkflow | None, overall_status: str) -> str:
    """Builds a human-readable summary of an executed/replayed
    ActionWorkflow. No LLM call -- the outcome is a short, deterministic
    fact (done / refused / stuck / never ran), not something that benefits
    from drafting, and this keeps the action path independent of Lyzr/
    OpenRouter being configured at all."""
    if workflow is None:
        return f'Could not complete "{transcript}": the action node did not run (status: {overall_status}).'

    step_summary = "; ".join(f"{s.kind}" + (f" \"{s.reasoning}\"" if s.reasoning else "") for s in workflow.steps)
    if workflow.success:
        return f'Done: "{transcript}". Steps taken: {step_summary or "none"}.'
    if workflow.refused_reason:
        return f'Refused "{transcript}" for safety: {workflow.refused_reason}. Steps taken before refusing: {step_summary or "none"}.'
    return f'Could not complete "{transcript}" (got stuck). Steps taken: {step_summary or "none"}.'


def _run_node_with_retry(run: RunState, node: DAGNode, ctx: RunContext) -> None:
    bind_run_context(run_id=run.run_id, node_id=node.id)
    handler = HANDLER_REGISTRY.get(node.handler_key)
    if handler is None:
        _transition(run, node.id, NodeStatus.FAILED, last_error=f"no handler registered for '{node.handler_key}'")
        run.failure_count += 1
        return

    state = run.node_states[node.id]
    error: str | None = None
    for attempt in range(1, node.max_retries + 1):
        state.attempts = attempt
        state.status = NodeStatus.RUNNING if attempt == 1 else NodeStatus.RETRYING
        state.started_at = datetime.now(timezone.utc)
        run_store.update_node_state(run, node.id, state)
        logger.info("node_attempt_started", attempt=attempt, max_retries=node.max_retries)

        # contextvars (which structlog's bind_run_context/bind_contextvars
        # use, see logging.py) do NOT cross a ThreadPoolExecutor thread
        # boundary on their own -- submitting `handler` directly meant
        # every log line a handler emitted (page_fetcher.py, qdrant_store.py,
        # etc., all called from inside handlers) silently lost its
        # run_id/node_id, defeating the whole point of correlated JSON
        # logs independent of Langfuse. copy_context().run(...) explicitly
        # carries the calling thread's bound context (already set via
        # bind_run_context two lines above) into the worker thread.
        ctx_snapshot = contextvars.copy_context()
        future = _thread_pool.submit(ctx_snapshot.run, handler, node, ctx)
        try:
            result = future.result(timeout=node.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()  # no-op if the thread already started -- see note below
            error = f"timed out after {node.timeout_seconds}s"
            logger.warning("node_attempt_timeout", attempt=attempt, error=error)
            # HONEST LIMITATION, not a fix: concurrent.futures cannot force-
            # stop a thread that's already running (`cancel()` only works
            # before it starts). If the handler is truly hung (e.g. Playwright
            # stuck on a network call) rather than merely slow, the original
            # thread keeps running in the background after we've moved on and
            # marked this attempt failed -- it can eventually succeed/fail on
            # its own and leak a thread-pool slot (and possibly a live
            # browser process) until it does. Mitigated in portal_client.py
            # via tighter Playwright-level timeouts so most real hangs raise
            # from *inside* the handler well before this outer timeout fires;
            # this counter makes the residual risk observable rather than silent.
            logger.error(
                "node_thread_possibly_orphaned",
                attempt=attempt,
                detail="outer timeout fired; the underlying thread may still be running",
            )
        except Exception as exc:  # noqa: BLE001 - a handler failure must not crash the executor
            error = str(exc)
            logger.warning("node_attempt_failed", attempt=attempt, error=error)
        else:
            state.status = NodeStatus.SUCCEEDED
            state.finished_at = datetime.now(timezone.utc)
            state.result_summary = str(result)[:200] if result is not None else None
            run_store.update_node_state(run, node.id, state)
            logger.info("node_succeeded", attempt=attempt)
            return

        if attempt < node.max_retries:
            time.sleep(node.retry_backoff_seconds * attempt)

    state.status = NodeStatus.FAILED
    state.finished_at = datetime.now(timezone.utc)
    state.last_error = f"{error} (exhausted {node.max_retries} retries)"
    run_store.update_node_state(run, node.id, state)
    run.failure_count += 1
    logger.error("node_failed_exhausted_retries", max_retries=node.max_retries)


def _transition(run: RunState, node_id: str, status: NodeStatus, *, last_error: str | None = None) -> None:
    state = run.node_states[node_id]
    state.status = status
    state.last_error = last_error
    state.finished_at = datetime.now(timezone.utc)
    run_store.update_node_state(run, node_id, state)


def _skip_remaining(run: RunState, graph: nx.DiGraph, from_node_id: str) -> None:
    order = list(nx.topological_sort(graph))
    idx = order.index(from_node_id)
    _skip_remaining_from(run, order[idx:])


def _skip_remaining_from(run: RunState, node_ids: list[str]) -> None:
    for node_id in node_ids:
        state = run.node_states[node_id]
        if state.status in (NodeStatus.PENDING,):
            _transition(run, node_id, NodeStatus.SKIPPED, last_error="circuit breaker tripped")
