"""Captures a full-page screenshot per candidate URL -- this is the core
"how we read the web" decision for the researcher: pages are read visually
(screenshot -> vision model), not by scraping DOM text, so paywalled,
JS-rendered, or visually-laid-out content (tables, charts, images) is
captured the way a human would see it rather than lost in a text dump.

Each URL is captured in isolation: one slow/broken/blocking site must not
fail the whole batch (same lesson as extractor.py's per-row isolation --
applied here to per-URL capture instead of per-row parsing).
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.research import ScreenshotCapture, SearchResult

logger = get_logger(component="screenshotter")

_CHROMIUM_EXECUTABLE_OVERRIDE = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
_PAGE_DEFAULT_TIMEOUT_MS = 15_000


def capture_screenshots(results: list[SearchResult], run_id: str) -> list[ScreenshotCapture]:
    out_dir = Path(settings.screenshot_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    captures: list[ScreenshotCapture] = []
    with sync_playwright() as p:
        # `timeout=` bounds the browser LAUNCH itself, not covered by
        # page.set_default_timeout() (page-level operations only).
        launch_kwargs = {"headless": True, "timeout": _PAGE_DEFAULT_TIMEOUT_MS}
        if _CHROMIUM_EXECUTABLE_OVERRIDE:
            launch_kwargs["executable_path"] = _CHROMIUM_EXECUTABLE_OVERRIDE
        browser = p.chromium.launch(**launch_kwargs)
        try:
            for i, result in enumerate(results):
                captures.append(_capture_one(browser, result, out_dir, i))
        finally:
            browser.close()

    ok = sum(1 for c in captures if c.error is None)
    logger.info("screenshots_captured", requested=len(results), succeeded=ok, failed=len(captures) - ok)
    return captures


def _capture_one(browser, result: SearchResult, out_dir: Path, index: int) -> ScreenshotCapture:
    screenshot_path = str(out_dir / f"{index:02d}.png")
    try:
        page = browser.new_page()
        try:
            page.set_default_timeout(_PAGE_DEFAULT_TIMEOUT_MS)
            response = page.goto(result.url, wait_until="load")
            # page.goto() does NOT raise on an HTTP error status (a 404
            # still "loads" as far as Playwright is concerned) -- without
            # this check a dead link gets screenshotted and sent to the
            # vision model as if it were real content, wasting a call on
            # an error page we already know isn't useful from the status
            # code alone. Same bug class caught live in page_fetcher.py's
            # equivalent fallback path.
            if response is not None and response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
            page.screenshot(path=screenshot_path, full_page=True)
        finally:
            page.close()
        return ScreenshotCapture(
            url=result.url,
            title=result.title,
            screenshot_path=screenshot_path,
            captured_at=datetime.now(timezone.utc),
        )
    except Exception as exc:  # noqa: BLE001 - one bad site must not fail the whole batch
        logger.warning("screenshot_capture_failed", url=result.url, error=str(exc))
        return ScreenshotCapture(
            url=result.url,
            title=result.title,
            screenshot_path="",
            captured_at=datetime.now(timezone.utc),
            error=str(exc),
        )
