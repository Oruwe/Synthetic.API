"""Schema for the ambient RPA action path: intent -> observe/decide/act
loop -> a recorded, replayable workflow.

This is a genuinely different kind of thing from FetchedPage/VisionFinding
(read-only research records): an ActionStep is a real, physical action this
system took on a page with no API -- a click or a keystroke -- not just an
observation about one.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ActionKind = Literal["click", "type", "scroll", "done", "refused", "stuck"]


class ActionStep(BaseModel):
    """One decision the vision model made and (if not done/refused/stuck)
    this system actually executed. Coordinates are normalized 0-1000 on
    both axes (a common VLM grounding convention) rather than raw pixels,
    since the model was never shown the screenshot's actual pixel
    dimensions -- the executor maps these back to real page coordinates
    using the screenshot it took."""

    model_config = ConfigDict(extra="forbid")

    kind: ActionKind
    x: int | None = None  # 0-1000, None for scroll/done/refused/stuck
    y: int | None = None
    text: str | None = None  # for "type"
    reasoning: str = ""  # the model's own stated reason, kept for the audit trail
    screenshot_path: str | None = None  # the screenshot this decision was made from


class ActionWorkflow(BaseModel):
    """A full attempt at carrying out one intent -- successful or not.
    This is the EXECUTION record for a single run (one run_id), never
    stored keyed by itself in Qdrant anymore -- see WorkflowMemory below,
    which is the durable, trust-weighted memory an attempt feeds into.
    Each attempt is still worth keeping around on the RunState it belongs
    to (see agents/orchestrator/executor.py) for that run's own audit
    trail, independent of whether it moved the shared memory forward."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    intent: str
    start_url: str
    steps: list[ActionStep] = Field(default_factory=list)
    success: bool
    # Set when success=False because the intent looked like a
    # payment/purchase/checkout action -- refused on purpose, not a
    # failure of capability. See action_executor.py's _looks_like_payment.
    refused_reason: str | None = None
    created_at: datetime


class WorkflowMemory(BaseModel):
    """The durable, shared memory record ambient RPA actually learns from --
    ONE row per (domain, canonicalized intent), continuously updated in
    place as attempts happen, not one row per run. This is the difference
    between a log of what was tried and memory of what works.

    Design choices that make this "production," not just persistence:
    - `canonical_key` (see qdrant_store._canonical_key) is derived from
      the target domain + a normalized intent string, so 50 successful
      runs of "the same" task reinforce ONE record's trust instead of
      creating 50 near-duplicate points a similarity search has to sift
      through.
    - `steps` always holds the MOST RECENT SUCCESSFUL sequence, not
      whatever the most recent attempt was -- a failed attempt updates
      the counters below but never overwrites a known-good replay target
      with an unverified one.
    - `success_count`/`failure_count` are the trust signal a bad match
      degrades over time: qdrant_store.find_workflow_memory gates replay
      on both semantic similarity AND a minimum trust ratio, so a
      workflow that broke after a page redesign stops being blindly
      replayed once enough recent attempts have failed against it,
      without a human ever having to manually invalidate it.
    - `domain` is checked (not just intent similarity) before a replay is
      offered, so a semantically similar phrase for a genuinely different
      site can't accidentally trigger a replay against the wrong page.
    """

    model_config = ConfigDict(extra="forbid")

    canonical_key: str
    domain: str
    representative_intent: str  # the intent text this record is embedded/matched by
    start_url: str
    steps: list[ActionStep] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime
    last_used_at: datetime
    last_success_at: datetime | None = None

    @property
    def trust_ratio(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
