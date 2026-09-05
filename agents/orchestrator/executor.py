"""DAG executor: topological execution with per-node timeout, retry with
backoff, write-through run-state persistence, and a circuit breaker.

Deliberately NOT Airflow/Temporal — a hand-rolled executor over a small
in-memory graph is enough for a handful of nodes per run, and networkx is
used only for topological ordering + cycle detection, not as a runtime.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import networkx as nx

from agents.common import notifier, run_store
from agents.common.logging import bind_run_context, clear_run_context, get_logger
from agents.common.models.dag import (
    DAGNode,
    DAGPlan,
    NodeExecutionState,
    NodeStatus,
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
            _run_node_with_retry(run, plan.nodes[0], ctx)
            run.overall_status = "no_capability"
            run_store.save_run(run)
            notifier.notify(
                f'I couldn\'t find a supported action for: "{plan.transcript}". '
                "Supported right now: checking the shipping portal for delayed orders, "
                "and web research queries.",
                plan.run_id,
            )
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
        run_store.save_run(run)
        logger.info("run_finished", overall_status=run.overall_status)
        return run
    finally:
        clear_run_context()


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

        future = _thread_pool.submit(handler, node, ctx)
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
