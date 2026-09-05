"""Real (non-mocked) exercise of page_fetcher against a local HTTP server:
actual httpx GET, actual trafilatura extraction, actual headless Chromium
navigation for the fallback path. No mocking of any of it.

Skipped by default -- this needs a real Chromium binary, which CI (and
most local `pytest` runs without Docker) doesn't have. Opt in with
RUN_LIVE_FETCH_TESTS=1, which is what the Dockerfile's browser install
guarantees is available. This is deliberately the one place in the test
suite that talks to a real (if local) HTTP server and a real browser --
everything else in the offline suite mocks those boundaries; this is the
closest thing to a live proof this sandbox can offer, given it has no
outbound network access to test the actual internet with.
"""

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_FETCH_TESTS"),
    reason="needs a real Chromium binary; set RUN_LIVE_FETCH_TESTS=1 to opt in (see agents/Dockerfile)",
)

_NORMAL_PAGE_HTML = f"""<html><head><title>Real Article</title></head>
<body><article><h1>Real Article</h1><p>{'This is genuine article content fetched over real HTTP. ' * 20}</p></article></body></html>"""

# Content only appears after JS runs -- httpx+trafilatura sees an empty
# shell; Playwright, which actually executes the script, sees the real text.
_JS_ONLY_PAGE_HTML = """<html><head><title>JS App</title></head>
<body><div id="root"></div>
<script>
document.getElementById('root').innerHTML =
  '<article><h1>Rendered Article</h1><p>' + 'This content only exists after JavaScript runs. '.repeat(20) + '</p></article>';
</script>
</body></html>"""

_ROBOTS_TXT = "User-agent: *\nDisallow: /disallowed\n"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        routes = {
            "/normal": (200, "text/html", _NORMAL_PAGE_HTML),
            "/js-only": (200, "text/html", _JS_ONLY_PAGE_HTML),
            "/robots.txt": (200, "text/plain", _ROBOTS_TXT),
            "/disallowed": (200, "text/html", _NORMAL_PAGE_HTML),
        }
        status, content_type, body = routes.get(self.path, (404, "text/plain", "not found"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):  # noqa: A002 - silence default request logging in test output
        pass


@pytest.fixture(scope="module")
def local_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _chromium_path(monkeypatch):
    override = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if override:
        import agents.web_navigator.page_fetcher as page_fetcher

        monkeypatch.setattr(page_fetcher, "_CHROMIUM_EXECUTABLE_OVERRIDE", override)


def test_fast_path_really_fetches_and_extracts_a_normal_page(local_server):
    import agents.web_navigator.robots as robots
    from agents.common.models.research import SearchResult
    from agents.web_navigator.page_fetcher import _fetch_fast

    robots._cache.clear()
    result = SearchResult(title="fallback title", url=f"{local_server}/normal")

    page = _fetch_fast(result, timeout_seconds=9)

    assert page is not None
    assert page.fetch_method == "http"
    assert "genuine article content" in page.text
    assert page.title == "Real Article"


def test_fallback_path_really_renders_js_via_playwright(local_server):
    import agents.web_navigator.robots as robots
    from agents.common.models.page import FetchedPage
    from agents.common.models.research import SearchResult
    from agents.web_navigator.page_fetcher import _fetch_one

    robots._cache.clear()
    result = SearchResult(title="fallback title", url=f"{local_server}/js-only")

    page = _fetch_one(result, timeout_seconds=15)

    assert isinstance(page, FetchedPage)
    assert page.error is None
    assert page.fetch_method == "playwright"  # fast path must have bailed on this one
    assert "only exists after JavaScript runs" in page.text


def test_a_real_404_is_isolated_as_a_failed_fetch(local_server):
    import agents.web_navigator.robots as robots
    from agents.common.models.research import SearchResult
    from agents.web_navigator.page_fetcher import fetch_pages

    robots._cache.clear()
    results = [SearchResult(title="missing", url=f"{local_server}/does-not-exist")]

    pages = fetch_pages(results, timeout_seconds=9)

    assert len(pages) == 1
    assert pages[0].error is not None


def test_real_robots_txt_is_fetched_and_respected(local_server):
    import agents.web_navigator.robots as robots

    robots._cache.clear()

    assert robots.is_allowed(f"{local_server}/disallowed") is False
    assert robots.is_allowed(f"{local_server}/normal") is True


def test_full_batch_against_a_real_server_mixed_outcomes(local_server):
    import agents.web_navigator.robots as robots
    from agents.common.models.research import SearchResult
    from agents.web_navigator.page_fetcher import fetch_pages

    robots._cache.clear()
    results = [
        SearchResult(title="normal", url=f"{local_server}/normal"),
        SearchResult(title="js", url=f"{local_server}/js-only"),
        SearchResult(title="missing", url=f"{local_server}/does-not-exist"),
        SearchResult(title="blocked", url=f"{local_server}/disallowed"),
    ]

    pages = fetch_pages(results, timeout_seconds=15)

    by_url = {p.url: p for p in pages}
    assert by_url[f"{local_server}/normal"].error is None
    assert by_url[f"{local_server}/normal"].fetch_method == "http"
    assert by_url[f"{local_server}/js-only"].error is None
    assert by_url[f"{local_server}/js-only"].fetch_method == "playwright"
    assert by_url[f"{local_server}/does-not-exist"].error is not None
    assert by_url[f"{local_server}/disallowed"].error == "disallowed by robots.txt"
