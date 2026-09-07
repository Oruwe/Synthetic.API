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
        # Explicit named kwargs, not a **dict unpack -- the previous version
        # built launch_kwargs as one dict mixing bool/int/list[str] values,
        # which mypy can only see as dict[str, object] once unpacked,
        # losing per-parameter type-checking against .launch()'s real
        # signature entirely. `or None` on executable_path preserves the
        # exact original behavior (an unset OR EMPTY-STRING override both
        # fall through to Playwright's own default, matching the old
        # `if CHROMIUM_EXECUTABLE_OVERRIDE:` truthiness check).
        browser: Browser = p.chromium.launch(
            headless=True,
            timeout=timeout_ms,
            # Docker's default /dev/shm is 64MB regardless of the
            # container's own memory limit -- a classic Chromium-in-
            # container crash cause (SIGBUS/renderer crashes on
            # anything past a trivial page) that this codebase's own
            # docker-compose.yml doesn't work around via shm_size
            # either. This flag makes Chromium use /tmp instead of
            # /dev/shm for shared memory -- slightly slower, much more
            # reliable under a container's real memory constraints
            # (most acute on a free-tier host like Render's 512MB, but
            # a real risk on every deployment, not something specific
            # to any one of them). Harmless outside a container too.
            args=["--disable-dev-shm-usage"],
            executable_path=CHROMIUM_EXECUTABLE_OVERRIDE or None,
        )
        try:
            yield browser
        finally:
            browser.close()
