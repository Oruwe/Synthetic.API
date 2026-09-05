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
