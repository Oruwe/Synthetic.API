"""Playwright browser lifecycle: log into the (mock) legacy portal and read
the dashboard's DOM into plain dicts of field -> text.

Deliberately reads via specific `[data-field]` selectors and returns plain
strings — never `page.content()` or anything that would hand raw page
markup to a later LLM step. See extractor.py for why that boundary matters.
"""

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.playwright_utils import PAGE_DEFAULT_TIMEOUT_MS, launched_browser

logger = get_logger(component="portal_client")

_ORDER_ROW_SELECTOR = '[data-testid="order-row"]'

# Kept comfortably below the DAG node's own timeout_seconds (default 30s in
# agents/common/models/dag.py) so a hung page raises *inside* this function
# and the executor's per-node timeout / retry logic runs the common case --
# a Python thread that's genuinely stuck cannot be force-stopped once
# started (see executor.py's node_thread_possibly_orphaned log), so the
# real mitigation is making that outcome rare, not handling it after the fact.
_PAGE_DEFAULT_TIMEOUT_MS = PAGE_DEFAULT_TIMEOUT_MS


def scrape_dashboard_rows() -> list[dict[str, str]]:
    with launched_browser(_PAGE_DEFAULT_TIMEOUT_MS) as browser:
        page = browser.new_page()
        page.set_default_timeout(_PAGE_DEFAULT_TIMEOUT_MS)
        page.goto(f"{settings.portal_base_url}/login")
        page.fill('[data-testid="username-input"]', settings.portal_username)
        page.fill('[data-testid="password-input"]', settings.portal_password)
        page.click('[data-testid="login-submit"]')
        page.wait_for_selector('[data-testid="orders-table"]', timeout=10_000)

        rows: list[dict[str, str]] = []
        for row in page.query_selector_all(_ORDER_ROW_SELECTOR):
            cells = row.query_selector_all("[data-field]")
            row_data = {cell.get_attribute("data-field"): cell.inner_text() for cell in cells}
            rows.append(row_data)

        logger.info("dashboard_scraped", row_count=len(rows))
        return rows
