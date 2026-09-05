"""Genuine concurrency test: multiple DAG runs executed from real threads
at the same time (not simulated) -- this is the actual shape of production
load (concurrent /trigger requests, each backed by FastAPI's own
BackgroundTasks thread), and it's exactly the scenario run_store's index
locking (agents/common/run_store.py) was built to survive: many runs'
node completions racing to update the same shared index file.
"""

import threading
import uuid
from datetime import datetime, timezone

from agents.common import run_store
from agents.common.config import settings
from agents.common.models.dag import DAGEdge, DAGNode, DAGPlan, NodeStatus, NodeType
from agents.orchestrator import executor

_CONCURRENT_RUN_COUNT = 20


def _plan(run_id: str, handler_key: str) -> DAGPlan:
    return DAGPlan(
        run_id=run_id,
        transcript=f"transcript for {run_id}",
        created_at=datetime.now(timezone.utc),
        nodes=[
            DAGNode(id="a", type=NodeType.FETCH_PAGES, name="a", handler_key=handler_key),
            DAGNode(id="b", type=NodeType.EMBED_PAGES, name="b", handler_key=handler_key, depends_on=["a"]),
        ],
        edges=[DAGEdge(from_node="a", to_node="b")],
    )


def test_many_concurrent_runs_do_not_corrupt_or_cross_contaminate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))

    key = f"concurrent_ok_{uuid.uuid4().hex}"
    executor.register_handler(key)(lambda node, ctx: f"handled {ctx.run_id}")

    run_ids = [str(uuid.uuid4()) for _ in range(_CONCURRENT_RUN_COUNT)]
    results: dict[str, object] = {}
    errors: list[Exception] = []

    def _run(run_id: str) -> None:
        try:
            plan = _plan(run_id, key)
            results[run_id] = executor.execute_plan(plan)
        except Exception as exc:  # noqa: BLE001 - captured for the assertion below, not swallowed silently
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(rid,)) for rid in run_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == _CONCURRENT_RUN_COUNT

    # Every run completed correctly, and -- the actual concurrency hazard
    # this test targets -- each run's own state is intact and uncorrupted:
    # no run's node_states leaked into another's file.
    for run_id, run in results.items():
        assert run.overall_status == "completed"
        assert run.run_id == run_id
        reloaded = run_store.load_run(run_id)
        assert reloaded is not None
        assert reloaded.run_id == run_id
        assert reloaded.node_states["a"].status == NodeStatus.SUCCEEDED
        assert reloaded.node_states["b"].status == NodeStatus.SUCCEEDED

    # The index (agents/common/run_store.py's _update_index, lock-protected)
    # must have EVERY run's final status -- a lost update here is exactly
    # the race the lock exists to prevent.
    summaries = run_store.list_run_summaries()
    assert set(summaries.keys()) == set(run_ids)
    assert all(s["overall_status"] == "completed" for s in summaries.values())


def test_concurrent_runs_where_some_fail_still_keep_accurate_independent_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))

    ok_key = f"concurrent_ok_{uuid.uuid4().hex}"
    fail_key = f"concurrent_fail_{uuid.uuid4().hex}"
    executor.register_handler(ok_key)(lambda node, ctx: "fine")
    executor.register_handler(fail_key)(lambda node, ctx: (_ for _ in ()).throw(RuntimeError("boom")))

    run_ids_ok = [str(uuid.uuid4()) for _ in range(8)]
    run_ids_fail = [str(uuid.uuid4()) for _ in range(8)]
    results: dict[str, object] = {}

    def _run(run_id: str, key: str) -> None:
        plan = _plan(run_id, key)
        # single retry, no backoff -- keep the failing-run threads fast
        for node in plan.nodes:
            node.max_retries = 1
            node.retry_backoff_seconds = 0
        results[run_id] = executor.execute_plan(plan)

    threads = [threading.Thread(target=_run, args=(rid, ok_key)) for rid in run_ids_ok]
    threads += [threading.Thread(target=_run, args=(rid, fail_key)) for rid in run_ids_fail]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for run_id in run_ids_ok:
        assert results[run_id].overall_status == "completed"
    for run_id in run_ids_fail:
        assert results[run_id].overall_status == "failed"

    summaries = run_store.list_run_summaries()
    for run_id in run_ids_ok:
        assert summaries[run_id]["overall_status"] == "completed"
    for run_id in run_ids_fail:
        assert summaries[run_id]["overall_status"] == "failed"
