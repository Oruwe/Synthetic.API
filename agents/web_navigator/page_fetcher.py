"""Fetches page text for a list of URLs, one of two ways:

- fast path: plain HTTP GET + trafilatura text extraction. Cheap, quick,
  works for the large majority of normal server-rendered pages.
- fallback path: full Playwright headless browser, used ONLY when the fast
  path fails outright or comes back with suspiciously little content (a
  common signal for a JS-rendered page that needs a real browser to
  populate).

Every URL is handled in total isolation: a failure (either path, or both,
or a robots.txt disallow) is caught, logged, and recorded as a FetchedPage
with `error` set rather than raised -- one bad site must never fail the
whole batch. This is the same per-item isolation discipline already used
in extractor.py (per-row) and screenshotter.py (per-URL), reused here
rather than reinvented.

Two courtesy/robustness measures apply to every real network attempt:
robots.txt is checked first (agents/web_navigator/robots.py), and a
per-domain rate limit is applied (agents/web_navigator/rate_limiter.py) --
neither is optional per-call, both fail open rather than block a
legitimate fetch on their own hiccup.

A third case, PDFs: Tavily search results occasionally point straight at a
PDF (a government report, a spec sheet). Neither of the two paths above can
read one -- trafilatura expects HTML, and Playwright's page.goto() on a PDF
response triggers a browser *download* event rather than rendering it,
which raises rather than returning content. So a PDF gets its own third
path (pypdf, text extraction only -- no OCR, so a scanned/image-only PDF
still comes back empty and isolated like any other failure), tried whenever
either the URL itself looks like a PDF or the server says so via
Content-Type, in preference to ever reaching the Playwright fallback for
one.
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import trafilatura
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.page import FetchedPage
from agents.common.models.research import SearchResult
from agents.web_navigator import rate_limiter, robots

logger = get_logger(component="page_fetcher")

_CHROMIUM_EXECUTABLE_OVERRIDE = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
# Minimum word count below which the fast path is treated as having failed
# (likely a JS-rendered page whose real content trafilatura can't see in
# the raw HTML, or a boilerplate/nav-only page) and the Playwright
# fallback is tried instead. Word count, not raw character count: a
# repeated-short-token page (nav links, tag clouds) can pass a character
# threshold while being useless content -- word count is a slightly better
# proxy, though still a proxy, not a true quality judgment.
_MIN_ACCEPTABLE_WORD_COUNT = 40
# Bounds how many URLs are fetched at once. Each URL is fully isolated (see
# module docstring) so there's no correctness reason to serialize them --
# doing so was a real bug: fetching results one at a time meant N URLs each
# up to `timeout_seconds` (worse with a Playwright fallback per URL) could
# add up to well past the DAG "fetch" node's own outer timeout (100s, with
# only 1 retry -- see planner.py), timing out the whole node and discarding
# every page already fetched even though most individual fetches succeeded.
# Kept modest rather than "one worker per URL" so this respects the same
# per-domain rate limit/robots-fetch machinery without hammering a site
# that happens to dominate a result set.
_FETCH_CONCURRENCY = 4


def fetch_pages(results: list[SearchResult], timeout_seconds: float | None = None) -> list[FetchedPage]:
    timeout_seconds = timeout_seconds or settings.page_fetch_timeout_seconds
    if not results:
        return []

    with ThreadPoolExecutor(
        max_workers=min(_FETCH_CONCURRENCY, len(results)), thread_name_prefix="page-fetch"
    ) as pool:
        # pool.map preserves input order in its output, so callers can still
        # zip `results` against the returned pages positionally.
        pages = list(pool.map(lambda result: _fetch_one(result, timeout_seconds), results))

    ok = sum(1 for p in pages if p.error is None)
    logger.info("pages_fetched", requested=len(results), succeeded=ok, failed=len(pages) - ok)
    return pages


def _looks_like_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def _fetch_one(result: SearchResult, timeout_seconds: float) -> FetchedPage:
    if not robots.is_allowed(result.url):
        logger.info("fetch_skipped_disallowed_by_robots_txt", url=result.url)
        return FetchedPage(
            url=result.url,
            title=result.title,
            text="",
            timestamp=datetime.now(timezone.utc),
            fetch_method="http",
            error="disallowed by robots.txt",
        )

    if _looks_like_pdf_url(result.url):
        # Go straight to the PDF path -- never try Playwright on a URL that
        # already looks like a PDF, since that path's own goto() would just
        # raise on the resulting download event.
        try:
            return _fetch_pdf(result, timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - one bad URL must not fail the whole batch
            logger.warning("pdf_fetch_failed", url=result.url, error=str(exc))
            return FetchedPage(
                url=result.url,
                title=result.title,
                text="",
                timestamp=datetime.now(timezone.utc),
                fetch_method="pdf",
                error=str(exc),
            )

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
    rate_limiter.throttle(result.url)

    # Separate, tighter connect timeout: a stuck DNS lookup or TCP
    # handshake shouldn't get to eat the whole per-page budget when a
    # slow-but-progressing transfer legitimately needs more of it.
    timeout = httpx.Timeout(timeout_seconds, connect=min(4.0, timeout_seconds))
    response = httpx.get(
        result.url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SyntheticAPI-Researcher/1.0)"},
    )
    response.raise_for_status()

    # Some PDFs don't end in ".pdf" (the URL check in _fetch_one only
    # catches the obvious case) -- caught live: a Tavily result with no
    # ".pdf" extension came back with this content-type and, before this
    # check existed, silently failed both trafilatura (garbage binary text)
    # and the Playwright fallback (raised on the download event) with no
    # content ever extracted.
    content_type = response.headers.get("content-type", "")
    if "application/pdf" in content_type.lower():
        return _extract_pdf_page(result, response.content)

    document = trafilatura.bare_extraction(response.text, with_metadata=True)
    text = (document.text if document and document.text else "").strip()
    if len(text.split()) < _MIN_ACCEPTABLE_WORD_COUNT:
        return None  # too little content -- let the caller try the Playwright fallback

    title = (document.title if document and document.title else None) or result.title
    return FetchedPage(
        url=result.url,
        title=title,
        text=text,
        timestamp=datetime.now(timezone.utc),
        fetch_method="http",
    )


def _fetch_pdf(result: SearchResult, timeout_seconds: float) -> FetchedPage:
    """For a URL that already looks like a PDF (see _looks_like_pdf_url) --
    fetched directly rather than going through _fetch_fast, since there's
    no HTML to try trafilatura on first."""
    rate_limiter.throttle(result.url)
    timeout = httpx.Timeout(timeout_seconds, connect=min(4.0, timeout_seconds))
    response = httpx.get(
        result.url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SyntheticAPI-Researcher/1.0)"},
    )
    response.raise_for_status()
    return _extract_pdf_page(result, response.content)


def _extract_pdf_page(result: SearchResult, pdf_bytes: bytes) -> FetchedPage:
    """Text-only extraction via pypdf -- no OCR, so a scanned/image-only
    PDF still comes back with no usable text. Never raises: a
    corrupt/encrypted/scanned PDF is reported via `error` exactly like any
    other per-URL failure (see module docstring), and the caller must treat
    whatever this returns as final -- never fall through to the Playwright
    fallback for a confirmed PDF, since that path can't handle one either."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:  # noqa: BLE001 - corrupt/encrypted PDF must not fail the whole batch
        return FetchedPage(
            url=result.url,
            title=result.title,
            text="",
            timestamp=datetime.now(timezone.utc),
            fetch_method="pdf",
            error=f"pdf extraction failed: {exc}",
        )

    if len(text.split()) < _MIN_ACCEPTABLE_WORD_COUNT:
        return FetchedPage(
            url=result.url,
            title=result.title,
            text="",
            timestamp=datetime.now(timezone.utc),
            fetch_method="pdf",
            error="pdf produced too little extractable text (likely scanned/image-only -- no OCR)",
        )

    return FetchedPage(
        url=result.url,
        title=result.title,
        text=text,
        timestamp=datetime.now(timezone.utc),
        fetch_method="pdf",
    )


def _fetch_with_playwright(result: SearchResult, timeout_seconds: float) -> FetchedPage:
    rate_limiter.throttle(result.url)

    timeout_ms = int(timeout_seconds * 1000)
    with sync_playwright() as p:
        # `timeout=` here bounds the browser LAUNCH itself (process spawn),
        # which page.set_default_timeout() below does not cover -- that
        # only applies to page-level operations (goto/click/etc.) on an
        # already-running browser. A hung launch was a real gap: it's the
        # one Playwright operation with no timeout anywhere else in this
        # function, and thus the one that could genuinely defeat the DAG
        # node's own outer timeout.
        launch_kwargs = {"headless": True, "timeout": timeout_ms}
        if _CHROMIUM_EXECUTABLE_OVERRIDE:
            launch_kwargs["executable_path"] = _CHROMIUM_EXECUTABLE_OVERRIDE
        browser = p.chromium.launch(**launch_kwargs)
        try:
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            response = page.goto(result.url, wait_until="load")
            # page.goto() does NOT raise on an HTTP error status -- a 404
            # or 500 still "loads" as far as Playwright is concerned, so
            # without this check an error page's own HTML (its "not
            # found"/"internal server error" body) gets extracted and
            # returned as if it were real content. Caught live: a test
            # against a real 404 endpoint came back looking like a
            # successful fetch until this check was added.
            if response is not None and response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
            html = page.content()
        finally:
            browser.close()

    document = trafilatura.bare_extraction(html, with_metadata=True)
    text = (document.text if document and document.text else "").strip()
    title = (document.title if document and document.title else None) or result.title
    return FetchedPage(
        url=result.url,
        title=title,
        text=text,
        timestamp=datetime.now(timezone.utc),
        fetch_method="playwright",
        error=None if text else "playwright fallback produced no extractable text",
    )
