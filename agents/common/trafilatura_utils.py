"""Shared helper for reading trafilatura's `bare_extraction()` result.

trafilatura types this return as `Document | dict[str, Any] | None` --
this codebase always calls it the same way (`with_metadata=...`) and, in
practice, always gets a `Document` object or `None` back, never a plain
dict. But that's an assumption about trafilatura's runtime behavior under
this codebase's specific call pattern, not something its own type stubs
guarantee -- three call sites (page_fetcher.py's HTTP and Playwright
paths, action_executor.py's page-text extraction) previously read
`.text`/`.title` as bare attribute access, which would raise AttributeError
on the dict case rather than degrading gracefully like every other
extraction failure in this codebase does. Centralized here so both shapes
are actually handled once, correctly, instead of assumed away three times.
"""

from typing import Any


def extract_text_and_title(document: Any) -> tuple[str, str | None]:
    """Returns (text, title) from a trafilatura bare_extraction() result,
    handling None, a real Document object, or (defensively) a plain dict
    -- text defaults to "" and title to None if genuinely absent, matching
    every call site's prior fallback behavior exactly."""
    if document is None:
        return "", None
    if isinstance(document, dict):
        text = document.get("text") or ""
        title = document.get("title") or None
    else:
        text = getattr(document, "text", None) or ""
        title = getattr(document, "title", None) or None
    return text, title
