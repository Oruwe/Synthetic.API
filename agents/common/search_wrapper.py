"""Isolation boundary for the web search API (Tavily), same pattern as
lyzr_wrapper.py / vision_wrapper.py: one place that knows the provider's
request/response shape, so a future provider swap touches only this file.

Non-negotiable per the pivot spec: a search failure must never raise past
this module. `search()` catches everything (timeout, HTTP error, malformed
response) and returns an empty list, logged, so the Orchestrator can still
build a plan (which then produces a "no sources found" answer downstream)
instead of the whole request failing.
"""

import time

import httpx

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.research import SearchResult

logger = get_logger(component="search_wrapper")

_TAVILY_URL = "https://api.tavily.com/search"
# Separate connect timeout: a stuck DNS lookup or TCP handshake is bounded
# tighter than a slow-but-progressing response body, so a genuinely dead
# network path fails fast rather than eating the full request budget.
_REQUEST_TIMEOUT = httpx.Timeout(10.0, connect=4.0)
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.5


def search(question: str, max_results: int | None = None) -> list[SearchResult]:
    max_results = max_results or settings.research_max_results

    if not settings.tavily_api_key:
        logger.warning("search_skipped_no_api_key", question=question)
        return []

    payload = _post_with_retry(question, max_results)
    if payload is None:
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


def _post_with_retry(question: str, max_results: int) -> dict | None:
    """Retries transient failures (timeout/connection error/5xx) up to
    _MAX_ATTEMPTS times with a short backoff; a 4xx (bad API key, bad
    request) fails immediately -- retrying a request that will never
    succeed just burns time and, on some plans, quota."""
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                _TAVILY_URL,
                json={
                    "api_key": settings.tavily_api_key,
                    "query": question,
                    "search_depth": "basic",
                    "max_results": max_results,
                },
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code < 500:
                logger.warning("search_failed_non_retryable", question=question, status=exc.response.status_code)
                return None
            logger.info("search_attempt_failed_retrying", question=question, attempt=attempt, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - a dead search API must not fail plan-building
            last_error = exc
            logger.info("search_attempt_failed_retrying", question=question, attempt=attempt, error=str(exc))

        if attempt < _MAX_ATTEMPTS:
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    logger.warning("search_failed", question=question, error=str(last_error), attempts=_MAX_ATTEMPTS)
    return None
