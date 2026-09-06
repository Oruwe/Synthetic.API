"""The DAG execution plan as real data, not implicit control flow.

The Orchestrator's planner builds a `DAGPlan`; `orchestrator/executor.py`
walks it in topological order, persisting a `RunState` after every node
transition so a run is inspectable mid-flight (`data/runs/<run_id>.json`).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.common.models.page import Source


class NodeType(str, Enum):
    SCRAPE_PORTAL = "scrape_portal"
    EXTRACT_VALIDATE = "extract_validate"
    EMBED_STORE = "embed_store"
    CLARIFY_UNSUPPORTED = "clarify_unsupported"
    # Web-Researcher (DDG+vision) -- retired from live routing, kept for the
    # dormant pipeline/tests, see agents/web_navigator/research_handlers.py
    SEARCH_WEB = "search_web"
    CAPTURE_SCREENSHOTS = "capture_screenshots"
    ANALYZE_SCREENSHOTS = "analyze_screenshots"
    EMBED_CANDIDATES = "embed_candidates"
    CURATE_KNOWLEDGE = "curate_knowledge"
    # Live research path: Tavily search (done in the planner, before the
    # DAG is built) -> fetch (HTTP+trafilatura, Playwright fallback) -> chunk+embed
    FETCH_PAGES = "fetch_pages"
    EMBED_PAGES = "embed_pages"


class DAGNode(BaseModel):
    id: str
    type: NodeType
    name: str
    handler_key: str  # lookup key into orchestrator.executor.HANDLER_REGISTRY
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


class DAGEdge(BaseModel):
    from_node: str
    to_node: str


PlanStatus = Literal["planned", "no_capability", "invalid"]


class DAGPlan(BaseModel):
    run_id: str
    transcript: str
    created_at: datetime
    nodes: list[DAGNode] = Field(default_factory=list)
    edges: list[DAGEdge] = Field(default_factory=list)
    status: PlanStatus = "planned"
    # Callers should pass settings.dag_circuit_breaker_threshold explicitly
    # (see orchestrator/planner.py) -- this default only applies if a plan
    # is constructed without going through the planner (e.g. in a test).
    circuit_breaker_threshold: int = 2
    unsupported_subintents: list[str] = Field(default_factory=list)


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeExecutionState(BaseModel):
    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    attempts: int = 0
    last_error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result_summary: str | None = None


OverallStatus = Literal["running", "completed", "failed", "circuit_broken", "no_capability"]


class RunState(BaseModel):
    run_id: str
    plan: DAGPlan
    node_states: dict[str, NodeExecutionState] = Field(default_factory=dict)
    overall_status: OverallStatus = "running"
    failure_count: int = 0
    # timezone-aware, matching every other timestamp in this codebase
    # (FetchedPage.timestamp, NodeExecutionState.started_at, etc. all use
    # datetime.now(timezone.utc)) -- datetime.utcnow() returns a naive
    # datetime, which can't even be compared to an aware one without
    # raising TypeError, caught by run_store.save_run()'s own test after
    # that function started actually reassigning this field on every save.
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Set by the Synthesizer once it drafts an answer for this run (see
    # agents/synthesizer/main.py) and persisted back via run_store.save_run
    # so GET /runs/{run_id} can hand it back directly -- before this field
    # existed, the ONLY place the answer appeared was the agents-synthesizer
    # container's stdout/logs (see agents/common/notifier.py), which isn't
    # something an API caller can poll for.
    #
    # `answer` is the full text INCLUDING the "Sources used: ..." /
    # "Partial results: ..." footer -- kept exactly as before for backward
    # compatibility (notifier.notify() and any existing consumer still see
    # the same string). The fields below are additive, not a replacement:
    # `answer_text` is the same answer with that footer stripped (what a
    # UI's "read aloud" or a clean answer display should use -- reading
    # "Sources used: https://..." out loud verbatim was a real, silly bug
    # in the first version of ui/app.py), and `sources` is the same
    # citation list structured as {url, title, snippet, score} instead of
    # a comma-joined string a UI would have to re-parse.
    answer: str | None = None
    answer_text: str | None = None
    sources: list[Source] = Field(default_factory=list)
    sources_attempted: int | None = None
    sources_succeeded: int | None = None


class PlanValidationError(Exception):
    """Raised when a DAGPlan is not a valid DAG (cycle, dangling edge, etc.)."""
