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
    Persisted to Qdrant (agents/common/qdrant_store.py's action_workflows
    collection) regardless of outcome: a failed/refused attempt is still
    useful signal against re-trying the exact same doomed approach, and a
    successful one is a literal replayable recording for a future
    semantically-similar intent (see agents/web_navigator/action_executor.py)."""

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
