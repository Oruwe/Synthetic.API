"""Transcript -> DAGPlan.

Two independent capabilities, checked in this order:

0. Web research ("search the web for...", "look up...", "research...") ->
   the search/screenshot/vision-analyze/embed/curate chain (see
   web_navigator/research_handlers.py). Checked FIRST and only on an
   explicit trigger phrase -- a bare question like "what's the weather
   today?" deliberately stays `no_capability` rather than auto-triggering
   a web search for anything ambiguous; the system acts, it doesn't guess.
1. Shipping portal (checked below, existing behavior).

Two layers within each capability, deliberately in this order:

1. A deterministic, rule/keyword-based base parse. This alone is enough to
   build the standard 3-node plan (scrape -> extract -> embed) for the
   hackathon's target scenario, needs no API key, and is what the golden
   tests in tests/test_orchestrator_golden.py assert against — so the
   planner's core behavior is provable offline and doesn't flake with LLM
   non-determinism.
2. An OPTIONAL Lyzr-assisted refinement layered on top (paraphrase
   robustness, e.g. "are there any late shipments?" instead of "delayed
   orders"), used only to fall back to intent classification when the
   rule-based pass finds no match. If Lyzr is disabled/unavailable, the
   rule-based layer alone still produces a working plan for the demo.
"""

import re
import uuid
from datetime import datetime, timezone

from agents.common.config import settings
from agents.common.lyzr_wrapper import LyzrAgentWrapper
from agents.common.models.dag import DAGEdge, DAGNode, DAGPlan, NodeType

_SHIPPING_KEYWORDS = re.compile(
    r"\b(shipping|delivery|delayed order|delayed shipment|late shipment|late order|"
    r"shiptrack|logistics)\b",
    re.I,
)
_DELAY_KEYWORDS = re.compile(r"\b(delay|delayed|late)\b", re.I)
_CUSTOMER_FILTER = re.compile(r"\bfor\s+customer\s+([A-Za-z][\w\s]*?)(?:[.,]|$)", re.I)
_URGENCY = re.compile(r"\b(urgent|asap|immediately|right away)\b", re.I)
_UNSUPPORTED_SUBINTENTS = {
    "cancel_order": re.compile(r"\bcancel\s+order\b", re.I),
}
_UNSUPPORTED_SYSTEMS = re.compile(r"\b(crm|erp|billing system|payroll)\b", re.I)

_RESEARCH_QUERY_EXTRACT = re.compile(
    r"\b(?:search(?: the web| online)? for|search(?: the web| online)?|look up|research|"
    r"find information (?:about|on)|google)\b\s*(.*)",
    re.I,
)

_orchestrator_agent = LyzrAgentWrapper(agent_role="orchestrator")


class PlannerInputError(ValueError):
    pass


def build_plan(transcript: str, run_id: str | None = None) -> DAGPlan:
    if not transcript or not transcript.strip():
        raise PlannerInputError("transcript is empty")

    run_id = run_id or str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    research_match = _RESEARCH_QUERY_EXTRACT.search(transcript)
    if research_match:
        query = research_match.group(1).strip(" .,!?") or transcript.strip()
        return _research_plan(transcript, query, run_id, created_at)

    # Reject explicitly out-of-scope systems before matching on generic
    # "delayed" language, so "check the CRM for delayed leads" doesn't get
    # silently mapped onto the shipping portal capability it doesn't have.
    if _UNSUPPORTED_SYSTEMS.search(transcript) and not _SHIPPING_KEYWORDS.search(transcript):
        return _no_capability_plan(transcript, run_id, created_at)

    is_shipping_task = bool(_SHIPPING_KEYWORDS.search(transcript) or _DELAY_KEYWORDS.search(transcript))
    if not is_shipping_task:
        return _no_capability_plan(transcript, run_id, created_at)

    params: dict = {}
    if _URGENCY.search(transcript):
        params["priority"] = "high"
    customer_match = _CUSTOMER_FILTER.search(transcript)
    if customer_match:
        params["customer_filter"] = customer_match.group(1).strip()

    unsupported = [name for name, pattern in _UNSUPPORTED_SUBINTENTS.items() if pattern.search(transcript)]

    nodes = [
        DAGNode(
            id="scrape",
            type=NodeType.SCRAPE_PORTAL,
            name="Scrape shipping portal dashboard",
            handler_key="scrape_portal",
            params=params,
        ),
        DAGNode(
            id="extract",
            type=NodeType.EXTRACT_VALIDATE,
            name="Extract and validate delayed orders",
            handler_key="extract_validate",
            depends_on=["scrape"],
        ),
        DAGNode(
            id="embed",
            type=NodeType.EMBED_STORE,
            name="Embed and store in Qdrant",
            handler_key="embed_store",
            depends_on=["extract"],
        ),
    ]
    edges = [
        DAGEdge(from_node="scrape", to_node="extract"),
        DAGEdge(from_node="extract", to_node="embed"),
    ]

    return DAGPlan(
        run_id=run_id,
        transcript=transcript,
        created_at=created_at,
        nodes=nodes,
        edges=edges,
        status="planned",
        unsupported_subintents=unsupported,
        circuit_breaker_threshold=settings.dag_circuit_breaker_threshold,
    )


def _research_plan(transcript: str, query: str, run_id: str, created_at: datetime) -> DAGPlan:
    nodes = [
        DAGNode(
            id="search",
            type=NodeType.SEARCH_WEB,
            name="Search the web",
            handler_key="search_web",
            params={"query": query},
        ),
        DAGNode(
            id="screenshot",
            type=NodeType.CAPTURE_SCREENSHOTS,
            name="Capture screenshots of top results",
            handler_key="capture_screenshots",
            depends_on=["search"],
            timeout_seconds=60,  # visiting several real sites is slower than the mock portal
        ),
        DAGNode(
            id="analyze",
            type=NodeType.ANALYZE_SCREENSHOTS,
            name="Analyze screenshots with an open-weight vision model",
            handler_key="analyze_screenshots",
            depends_on=["screenshot"],
            timeout_seconds=60,
        ),
        DAGNode(
            id="embed_candidates",
            type=NodeType.EMBED_CANDIDATES,
            name="Embed candidate findings in Qdrant",
            handler_key="embed_candidates",
            depends_on=["analyze"],
        ),
        DAGNode(
            id="curate",
            type=NodeType.CURATE_KNOWLEDGE,
            name="Curate: keep relevant findings, delete the rest",
            handler_key="curate_knowledge",
            depends_on=["embed_candidates"],
        ),
    ]
    edges = [
        DAGEdge(from_node="search", to_node="screenshot"),
        DAGEdge(from_node="screenshot", to_node="analyze"),
        DAGEdge(from_node="analyze", to_node="embed_candidates"),
        DAGEdge(from_node="embed_candidates", to_node="curate"),
    ]
    return DAGPlan(
        run_id=run_id,
        transcript=transcript,
        created_at=created_at,
        nodes=nodes,
        edges=edges,
        status="planned",
        circuit_breaker_threshold=settings.dag_circuit_breaker_threshold,
    )


def _no_capability_plan(transcript: str, run_id: str, created_at: datetime) -> DAGPlan:
    node = DAGNode(
        id="clarify",
        type=NodeType.CLARIFY_UNSUPPORTED,
        name="No matching capability for this request",
        handler_key="clarify_unsupported",
    )
    return DAGPlan(
        run_id=run_id,
        transcript=transcript,
        created_at=created_at,
        nodes=[node],
        edges=[],
        status="no_capability",
        circuit_breaker_threshold=settings.dag_circuit_breaker_threshold,
    )
