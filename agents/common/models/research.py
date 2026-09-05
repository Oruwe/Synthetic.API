"""Schema for the Web-Researcher pipeline: search result -> screenshot ->
vision-model analysis -> Qdrant candidate -> curated (permanent) finding.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    url: str


class ScreenshotCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    screenshot_path: str
    captured_at: datetime
    # None when the capture itself failed (timeout, DNS error, etc.) --
    # kept as a record rather than silently dropped, so a run's screenshot
    # attempts are all visible in the run-state file even if some failed.
    error: str | None = None


class VisionFinding(BaseModel):
    """A vision-language model's read of one screenshot. `flags` follows the
    same convention as DelayedOrder.flags (agents/common/models/orders.py):
    guard-pattern hits are recorded, never used to silently strip content."""

    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    screenshot_path: str
    flags: list[str] = Field(default_factory=list)
    analyzed_at: datetime
