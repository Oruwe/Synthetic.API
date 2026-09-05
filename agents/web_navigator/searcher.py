"""Keyless web search via DuckDuckGo's HTML endpoint (no API key, fits the
project's no-vendor-lock-in stance -- consistent with routing the LLM
fallback through OpenRouter to an open-weight model rather than a closed API).

Only used to get a short list of candidate URLs. Reads results via Playwright
selectors into plain strings, same discipline as portal_client.py -- no page
content is ever handed to an LLM here, that only happens per-screenshot in
the vision-analysis step, and only as an image, never as scraped page text.
"""

import os
from urllib.parse import parse_qs, quote, unquote, urlparse

from playwright.sync_api import sync_playwright

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.research import SearchResult

logger = get_logger(component="searcher")

_RESULT_LINK_SELECTOR = "a.result__a"
_CHROMIUM_EXECUTABLE_OVERRIDE = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
_PAGE_DEFAULT_TIMEOUT_MS = 15_000


def _unwrap_ddg_redirect(href: str) -> str:
    """DuckDuckGo's HTML endpoint wraps result links in its own redirect
    (`//duckduckgo.com/l/?uddg=<encoded-target>&...`) -- unwrap to the real
    target URL so downstream screenshotting hits the actual site."""
    if "duckduckgo.com/l/" not in href:
        return href
    parsed = urlparse(href if "://" in href else f"https:{href}")
    target = parse_qs(parsed.query).get("uddg", [None])[0]
    return unquote(target) if target else href


def search_web(query: str, max_results: int | None = None) -> list[SearchResult]:
    max_results = max_results or settings.research_max_results

    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if _CHROMIUM_EXECUTABLE_OVERRIDE:
            launch_kwargs["executable_path"] = _CHROMIUM_EXECUTABLE_OVERRIDE
        browser = p.chromium.launch(**launch_kwargs)
        try:
            page = browser.new_page()
            page.set_default_timeout(_PAGE_DEFAULT_TIMEOUT_MS)
            # DDG's HTML endpoint accepts the query as a GET param; a built
            # URL avoids a form-fill round trip.
            page.goto(f"{settings.search_engine_url}?q={quote(query)}")

            results: list[SearchResult] = []
            for link in page.query_selector_all(_RESULT_LINK_SELECTOR):
                title = (link.inner_text() or "").strip()
                href = link.get_attribute("href") or ""
                url = _unwrap_ddg_redirect(href)
                if not title or not url.startswith("http"):
                    continue
                results.append(SearchResult(title=title[:300], url=url))
                if len(results) >= max_results:
                    break

            logger.info("web_search_completed", query=query, result_count=len(results))
            return results
        finally:
            browser.close()
