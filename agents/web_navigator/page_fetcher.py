"""Fetches page text for a list of URLs, one of two ways:

- fast path: plain HTTP GET + trafilatura text extraction. Cheap, quick,
  works for the large majority of normal server-rendered pages.
- fallback path: full Playwright headless browser, used ONLY when the fast
  path fails outright or comes back with suspiciously little text (a common
  signal for a JS-rendered page that needs a real browser to populate).

Every URL is handled in total isolation: a failure (either path, or both)
is caught, logged, and recorded as a FetchedPage with `error` set rather
than raised -- one bad site must never fail the whole batch. This is the
same per-item isolation discipline already used in extractor.py (per-row)
and screenshotter.py (per-URL), reused here rather than reinvented.
"""

import os
from datetime import datetime, timezone

import httpx
import trafilatura
from playwright.sync_api import sync_playwright

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.page import FetchedPage
from agents.common.models.research import SearchResult

logger = get_logger(component="page_fetcher")

_CHROMIUM_EXECUTABLE_OVERRIDE = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
# Minimum extracted length below which the fast path is treated as having
# failed (likely a JS-rendered page whose real content trafilatura can't
# see in the raw HTML) and the Playwright fallback is tried instead.
_MIN_ACCEPTABLE_TEXT_LENGTH = 200


def fetch_pages(results: list[SearchResult], timeout_seconds: float | None = None) -> list[FetchedPage]:
    timeout_seconds = timeout_seconds or settings.page_fetch_timeout_seconds
    pages = [_fetch_one(result, timeout_seconds) for result in results]
    ok = sum(1 for p in pages if p.error is None)
    logger.info("pages_fetched", requested=len(results), succeeded=ok, failed=len(pages) - ok)
    return pages


def _fetch_one(result: SearchResult, timeout_seconds: float) -> FetchedPage:
    try:
        page = _fetch_fast(result, timeout_seconds)
        if page is not None:
            return page
    except Exception as exc:  # noqa: BLE001 - fast path failing must fall through to Playwright, not raise
        logger.info("fast_fetch_failed_trying_fallback", url=result.url, error=str(exc))

    try:
        return _fetch_with_playwright(result, timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - one bad URL must not fail the whole batch
        logger.warning("page_fetch_failed", url=result.url, error=str(exc))
        return FetchedPage(
            url=result.url,
            title=result.title,
            text="",
            timestamp=datetime.now(timezone.utc),
            fetch_method="http",
            error=str(exc),
        )


def _fetch_fast(result: SearchResult, timeout_seconds: float) -> FetchedPage | None:
    response = httpx.get(
        result.url,
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SyntheticAPI-Researcher/1.0)"},
    )
    response.raise_for_status()
    text = trafilatura.extract(response.text) or ""
    if len(text.strip()) < _MIN_ACCEPTABLE_TEXT_LENGTH:
        return None  # too little content -- let the caller try the Playwright fallback
    return FetchedPage(
        url=result.url,
        title=result.title,
        text=text.strip(),
        timestamp=datetime.now(timezone.utc),
        fetch_method="http",
    )


def _fetch_with_playwright(result: SearchResult, timeout_seconds: float) -> FetchedPage:
    timeout_ms = int(timeout_seconds * 1000)
    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if _CHROMIUM_EXECUTABLE_OVERRIDE:
            launch_kwargs["executable_path"] = _CHROMIUM_EXECUTABLE_OVERRIDE
        browser = p.chromium.launch(**launch_kwargs)
        try:
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(result.url, wait_until="load")
            html = page.content()
        finally:
            browser.close()

    text = (trafilatura.extract(html) or "").strip()
    return FetchedPage(
        url=result.url,
        title=result.title,
        text=text,
        timestamp=datetime.now(timezone.utc),
        fetch_method="playwright",
        error=None if text else "playwright fallback produced no extractable text",
    )
