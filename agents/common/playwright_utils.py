"""Shared Playwright browser-lifecycle helper.

Every synchronous-Playwright caller in this codebase (portal_client.py,
searcher.py, screenshotter.py) needs the same three things: an optional
executable-path override for sandboxes with a pre-installed Chromium whose
revision doesn't match this Playwright version, a launch-level timeout
(page.set_default_timeout() only covers page operations, not the browser
launch itself), and reliable browser.close() on the way out. This was
previously copy-pasted verbatim in each of those modules; centralized here
so the override/timeout logic has exactly one definition to fix or tune.
"""

import os
from contextlib import contextmanager

from playwright.sync_api import Browser, sync_playwright

# Optional override for environments with a pre-installed browser binary
# whose revision doesn't match what this playwright version expects (e.g.
# a shared sandbox image) -- normally unset; the Dockerfile runs
# `playwright install --with-deps chromium` so this isn't needed there.
CHROMIUM_EXECUTABLE_OVERRIDE = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")

# Kept comfortably below the DAG node's own timeout_seconds (default 30s in
# agents/common/models/dag.py) so a hung page raises *inside* the caller
# and the executor's per-node timeout / retry logic runs the common case.
PAGE_DEFAULT_TIMEOUT_MS = 15_000


@contextmanager
def launched_browser(timeout_ms: int = PAGE_DEFAULT_TIMEOUT_MS):
    """Launches a headless Chromium browser and guarantees it's closed.

    `timeout_ms` bounds the browser LAUNCH itself, which
    page.set_default_timeout() does not cover (that only applies to
    page-level operations on an already-running browser).
    """
    with sync_playwright() as p:
        launch_kwargs = {"headless": True, "timeout": timeout_ms}
        if CHROMIUM_EXECUTABLE_OVERRIDE:
            launch_kwargs["executable_path"] = CHROMIUM_EXECUTABLE_OVERRIDE
        browser: Browser = p.chromium.launch(**launch_kwargs)
        try:
            yield browser
        finally:
            browser.close()
