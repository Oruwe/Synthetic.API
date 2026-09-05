"""Transcript -> DAGPlan.

The live path: any non-empty transcript is treated as a research question.
Tavily is called HERE, before the DAG is built (per the pivot spec — search
is plan-construction logic, not something Web-Navigator does at runtime
anymore), and the resulting candidate URLs are baked into the first node's
params. Web-Navigator still owns fetching/extracting (its responsibility is
unchanged) — it just now fetches whatever URLs the planner found instead of
navigating one hardcoded portal.

Two nodes: fetch_pages -> embed_pages. Both flow through the existing,
untouched DAG executor (retries/timeout/circuit-breaker), so nothing here
needs its own retry logic beyond what agents/web_navigator/page_fetcher.py
already does per-URL.

Older keyword-routed planning (shipping portal / DuckDuckGo+vision search)
lived in this file before this pivot and is fully retired from live
routing — see agents/web_navigator/handlers.py, research_handlers.py, and
their still-present, still-tested underlying modules for what's now dormant
rather than deleted.
"""

import uuid
from datetime import datetime, timezone

from agents.common import search_wrapper
from agents.common.config import settings
from agents.common.models.dag import DAGEdge, DAGNode, DAGPlan, NodeType


class PlannerInputError(ValueError):
    pass


def build_plan(transcript: str, run_id: str | None = None) -> DAGPlan:
    if not transcript or not transcript.strip():
        raise PlannerInputError("transcript is empty")

    run_id = run_id or str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    question = transcript.strip()

    # search_wrapper.search() never raises (see its own docstring) -- a
    # search-API outage still produces a plan; fetch_pages/embed_pages then
    # produce a "no sources found" answer rather than blocking planning.
    search_results = search_wrapper.search(question)

    nodes = [
        DAGNode(
            id="fetch",
            type=NodeType.FETCH_PAGES,
            name="Fetch and extract candidate pages",
            handler_key="fetch_pages",
            params={
                "question": question,
                "search_results": [r.model_dump(mode="json") for r in search_results],
            },
            timeout_seconds=100,
            max_retries=1,  # per-URL fast/fallback retry already happens inside the handler
        ),
        DAGNode(
            id="embed",
            type=NodeType.EMBED_PAGES,
            name="Chunk and embed fetched pages",
            handler_key="embed_pages",
            depends_on=["fetch"],
            timeout_seconds=60,
        ),
    ]
    edges = [DAGEdge(from_node="fetch", to_node="embed")]

    return DAGPlan(
        run_id=run_id,
        transcript=transcript,
        created_at=created_at,
        nodes=nodes,
        edges=edges,
        status="planned",
        circuit_breaker_threshold=settings.dag_circuit_breaker_threshold,
    )
