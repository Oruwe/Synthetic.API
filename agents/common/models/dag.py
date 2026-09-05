"""The DAG execution plan as real data, not implicit control flow.

The Orchestrator's planner builds a `DAGPlan`; `orchestrator/executor.py`
walks it in topological order, persisting a `RunState` after every node
transition so a run is inspectable mid-flight (`data/runs/<run_id>.json`).
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    SCRAPE_PORTAL = "scrape_portal"
    EXTRACT_VALIDATE = "extract_validate"
    EMBED_STORE = "embed_store"
    CLARIFY_UNSUPPORTED = "clarify_unsupported"
    # Web-Researcher: search -> screenshot -> vision-analyze -> embed -> curate
    SEARCH_WEB = "search_web"
    CAPTURE_SCREENSHOTS = "capture_screenshots"
    ANALYZE_SCREENSHOTS = "analyze_screenshots"
    EMBED_CANDIDATES = "embed_candidates"
    CURATE_KNOWLEDGE = "curate_knowledge"


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
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PlanValidationError(Exception):
    """Raised when a DAGPlan is not a valid DAG (cycle, dangling edge, etc.)."""
