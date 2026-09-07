"""Schema for the search -> fetch pipeline: a page fetched via the fast
(HTTP + trafilatura) or fallback (Playwright) path, before chunking."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FetchedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    text: str
    timestamp: datetime
    fetch_method: str  # "http" | "playwright"
    # None on success. Set (and `text` left empty) when both the fast and
    # fallback paths failed for this URL -- the fetch is isolated per-URL,
    # so one bad site never fails the whole batch (see page_fetcher.py).
    error: str | None = None
    # Human-in-the-loop gate path (feature/ambient-rpa-action-bridge):
    # True when page_fetcher detected this page is showing a login/
    # subscribe/paywall notice instead of real content (see
    # page_fetcher._detect_gate_phrase). Distinct from `error`: a gate is
    # not a fetch FAILURE -- the fetch worked fine, the content is just
    # behind a wall a human can help get past (see
    # page_handlers.handle_fetch_pages, which pauses the run for exactly
    # this signal). `gate_reason` is the matched phrase, for the prompt
    # shown to the human and for debugging a false positive/negative.
    gated: bool = False
    gate_reason: str | None = None


class Source(BaseModel):
    """A structured citation for a drafted answer -- replaces the old
    "Sources used: url1, url2" flattened string (still kept, for backward
    compatibility, as part of RunState.answer's full text) with proper
    fields a UI can render as clickable cards instead of parsing prose.
    Built from the same retrieved Qdrant chunks the answer itself is
    drafted from -- see synthesizer/drafter.py's _build_sources()."""

    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    snippet: str | None = None
    score: float | None = None  # retrieval relevance (cosine similarity); highest first
