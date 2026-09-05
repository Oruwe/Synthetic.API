"""Isolation boundary for the web search API (Tavily), same pattern as
lyzr_wrapper.py / vision_wrapper.py: one place that knows the provider's
request/response shape, so a future provider swap touches only this file.

Non-negotiable per the pivot spec: a search failure must never raise past
this module. `search()` catches everything (timeout, HTTP error, malformed
response) and returns an empty list, logged, so the Orchestrator can still
build a plan (which then produces a "no sources found" answer downstream)
instead of the whole request failing.
"""

import httpx

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.research import SearchResult

logger = get_logger(component="search_wrapper")

_TAVILY_URL = "https://api.tavily.com/search"
_REQUEST_TIMEOUT_SECONDS = 10.0


def search(question: str, max_results: int | None = None) -> list[SearchResult]:
    max_results = max_results or settings.research_max_results

    if not settings.tavily_api_key:
        logger.warning("search_skipped_no_api_key", question=question)
        return []

    try:
        response = httpx.post(
            _TAVILY_URL,
            json={
                "api_key": settings.tavily_api_key,
                "query": question,
                "search_depth": "basic",
                "max_results": max_results,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - a dead search API must not fail plan-building
        logger.warning("search_failed", question=question, error=str(exc))
        return []

    results: list[SearchResult] = []
    for item in payload.get("results", [])[:max_results]:
        url = item.get("url")
        title = item.get("title")
        if not url or not title:
            continue
        results.append(SearchResult(title=title[:300], url=url, snippet=(item.get("content") or "")[:500] or None))

    logger.info("search_completed", question=question, result_count=len(results))
    return results
