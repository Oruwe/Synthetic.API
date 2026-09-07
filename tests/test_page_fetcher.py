"""Tests for page_fetcher's fast-path/fallback decision, per-URL
isolation, and its robots.txt / rate-limit courtesy checks. httpx,
trafilatura, robots, and the rate limiter are all mocked -- no real
network or browser needed, consistent with the rest of the offline suite.
"""

from types import SimpleNamespace

import httpx
import pytest

import agents.web_navigator.page_fetcher as page_fetcher
from agents.common.models.research import SearchResult


def _result(url="https://example.test", title="Example"):
    return SearchResult(title=title, url=url)


def _doc(text, title=None):
    return SimpleNamespace(text=text, title=title)


class _FakeResponse:
    def __init__(self, text, content_type="text/html", content=b""):
        self.text = text
        self.content = content
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        pass


@pytest.fixture(autouse=True)
def _allow_and_no_throttle(monkeypatch):
    """Every test in this file exercises fetch logic, not the robots/rate-
    limit checks themselves (those have their own test files) -- default
    them to a no-op so tests stay fast and deterministic regardless of
    sandbox network conditions."""
    monkeypatch.setattr(page_fetcher.robots, "is_allowed", lambda url: True)
    monkeypatch.setattr(page_fetcher.rate_limiter, "throttle", lambda url: None)


def test_fast_path_succeeds_for_normal_page(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResponse("<html>...</html>"))
    monkeypatch.setattr(page_fetcher.trafilatura, "bare_extraction", lambda html, **kw: _doc("word " * 100, title="A Title"))

    page = page_fetcher._fetch_fast(_result(), timeout_seconds=9)

    assert page is not None
    assert page.fetch_method == "http"
    assert page.error is None
    assert page.title == "A Title"


def test_fast_path_returns_none_when_word_count_too_low(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResponse("<html>...</html>"))
    monkeypatch.setattr(page_fetcher.trafilatura, "bare_extraction", lambda html, **kw: _doc("too short"))

    page = page_fetcher._fetch_fast(_result(), timeout_seconds=9)

    assert page is None  # signals the caller to try the Playwright fallback


def test_fast_path_detects_a_gate_instead_of_falling_back_to_playwright(monkeypatch):
    """A real subscribe-wall/paywall notice is a distinct outcome from a
    generic low-content page: gated, not just "try Playwright" -- the
    gate text usually renders server-side too, so a browser wouldn't see
    anything different."""
    html = "<html><body><div class='gate'>Subscribe to continue reading this article.</div></body></html>"
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResponse(html))
    monkeypatch.setattr(page_fetcher.trafilatura, "bare_extraction", lambda html, **kw: _doc(""))

    page = page_fetcher._fetch_fast(_result(), timeout_seconds=9)

    assert page is not None
    assert page.gated is True
    assert page.gate_reason is not None
    assert page.error is None  # a gate is not a fetch failure


def test_fast_path_with_thin_content_and_no_gate_phrase_still_falls_back(monkeypatch):
    """Regression guard for the detector's own false-positive risk: thin
    content alone, with no matching phrase, must still behave exactly as
    before this feature existed -- try Playwright, not a spurious gate."""
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResponse("<html><body>nav home about</body></html>"))
    monkeypatch.setattr(page_fetcher.trafilatura, "bare_extraction", lambda html, **kw: _doc("too short"))

    page = page_fetcher._fetch_fast(_result(), timeout_seconds=9)

    assert page is None


def test_fast_path_does_not_flag_a_gate_when_real_content_was_extracted(monkeypatch):
    """A long, legitimately-fetched article that happens to mention one
    of the gate phrases somewhere (a footer, a related-article teaser)
    must not be flagged -- the word-count check gates the gate check."""
    html = "<html><body>" + "real article content. " * 100 + " by the way, subscribe to continue reading our newsletter.</body></html>"
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResponse(html))
    monkeypatch.setattr(page_fetcher.trafilatura, "bare_extraction", lambda html, **kw: _doc("real article content. " * 100))

    page = page_fetcher._fetch_fast(_result(), timeout_seconds=9)

    assert page is not None
    assert page.gated is False


@pytest.mark.parametrize(
    "phrase",
    [
        "Sign in to continue reading",
        "Log in to view this content",
        "Subscribe to read the full story",
        "Register to continue",
        "Create a free account to unlock this article",
        "Enter your email to continue",
        "This content is for subscribers only",
        "Members-only content",
    ],
)
def test_detect_gate_phrase_recognizes_common_gate_wording(phrase):
    assert page_fetcher._detect_gate_phrase(f"<div>{phrase}</div>") is not None


def test_detect_gate_phrase_returns_none_for_ordinary_text():
    assert page_fetcher._detect_gate_phrase("<div>Welcome to our homepage. Read our latest updates.</div>") is None


def test_fast_path_falls_back_to_search_result_title_when_none_extracted(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResponse("<html>...</html>"))
    monkeypatch.setattr(page_fetcher.trafilatura, "bare_extraction", lambda html, **kw: _doc("word " * 100, title=None))

    page = page_fetcher._fetch_fast(_result(title="Fallback Title"), timeout_seconds=9)

    assert page.title == "Fallback Title"


def test_fetch_one_skips_urls_disallowed_by_robots_txt(monkeypatch):
    monkeypatch.setattr(page_fetcher.robots, "is_allowed", lambda url: False)
    called = {"count": 0}
    monkeypatch.setattr(page_fetcher, "_fetch_fast", lambda *a, **kw: called.__setitem__("count", called["count"] + 1))

    page = page_fetcher._fetch_one(_result(url="https://disallowed.test"), timeout_seconds=9)

    assert called["count"] == 0  # never even attempted
    assert page.error == "disallowed by robots.txt"
    assert page.url == "https://disallowed.test"


def test_fetch_one_falls_back_to_playwright_when_fast_path_fails(monkeypatch):
    monkeypatch.setattr(page_fetcher, "_fetch_fast", lambda result, timeout: None)
    called = {}

    def fake_playwright_fetch(result, timeout):
        called["url"] = result.url
        from datetime import datetime, timezone

        from agents.common.models.page import FetchedPage

        return FetchedPage(
            url=result.url, title=result.title, text="from playwright",
            timestamp=datetime.now(timezone.utc), fetch_method="playwright",
        )

    monkeypatch.setattr(page_fetcher, "_fetch_with_playwright", fake_playwright_fetch)

    page = page_fetcher._fetch_one(_result(url="https://needs-js.test"), timeout_seconds=9)

    assert called["url"] == "https://needs-js.test"
    assert page.fetch_method == "playwright"
    assert page.text == "from playwright"


def test_fetch_one_isolates_total_failure_without_raising(monkeypatch):
    def _raise_fast(result, timeout):
        raise httpx.ConnectTimeout("timed out")

    def _raise_fallback(result, timeout):
        raise RuntimeError("playwright also failed")

    monkeypatch.setattr(page_fetcher, "_fetch_fast", _raise_fast)
    monkeypatch.setattr(page_fetcher, "_fetch_with_playwright", _raise_fallback)

    page = page_fetcher._fetch_one(_result(url="https://totally-broken.test"), timeout_seconds=9)

    assert page.error is not None
    assert page.url == "https://totally-broken.test"
    assert page.text == ""


def test_playwright_fallback_treats_http_error_status_as_a_failure(monkeypatch):
    """Regression test: page.goto() does NOT raise on an HTTP error status
    (a 404/500 still "loads" as far as Playwright is concerned) -- caught
    live against a real server (test_page_fetcher_live.py) returning what
    looked like a successful fetch of a 404 page's own error-page HTML.
    Without the status check, this silently corrupts an answer with a
    "not found" page treated as real content."""

    class _FakeResponse:
        status = 404

    class _FakePage:
        def set_default_timeout(self, ms):
            pass

        def goto(self, url, wait_until="load"):
            return _FakeResponse()

        def content(self):
            raise AssertionError("must not read page content after a 4xx/5xx status")

    class _FakeBrowser:
        def new_page(self):
            return _FakePage()

        def close(self):
            pass

    class _FakeChromium:
        def launch(self, **kwargs):
            return _FakeBrowser()

    class _FakePlaywrightContext:
        def __enter__(self):
            return SimpleNamespace(chromium=_FakeChromium())

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(page_fetcher, "sync_playwright", lambda: _FakePlaywrightContext())

    with pytest.raises(RuntimeError, match="404"):
        page_fetcher._fetch_with_playwright(_result(), timeout_seconds=9)


def test_fetch_one_isolates_a_playwright_http_error_status_without_raising(monkeypatch):
    monkeypatch.setattr(page_fetcher, "_fetch_fast", lambda result, timeout: None)
    monkeypatch.setattr(
        page_fetcher, "_fetch_with_playwright", lambda result, timeout: (_ for _ in ()).throw(RuntimeError("HTTP 404"))
    )

    page = page_fetcher._fetch_one(_result(url="https://dead-link.test"), timeout_seconds=9)

    assert page.error == "HTTP 404"


def test_fetch_pages_isolates_one_bad_url_from_the_rest(monkeypatch):
    def fake_fetch_one(result, timeout):
        from datetime import datetime, timezone

        from agents.common.models.page import FetchedPage

        if "bad" in result.url:
            return FetchedPage(
                url=result.url, title=result.title, text="", timestamp=datetime.now(timezone.utc),
                fetch_method="http", error="simulated failure",
            )
        return FetchedPage(
            url=result.url, title=result.title, text="good content", timestamp=datetime.now(timezone.utc),
            fetch_method="http",
        )

    monkeypatch.setattr(page_fetcher, "_fetch_one", fake_fetch_one)

    results = [_result(url="https://good1.test"), _result(url="https://bad.test"), _result(url="https://good2.test")]
    pages = page_fetcher.fetch_pages(results, timeout_seconds=9)

    assert len(pages) == 3
    assert sum(1 for p in pages if p.error is None) == 2
    assert [p.url for p in pages if p.error is not None] == ["https://bad.test"]


# --- PDF handling ---------------------------------------------------------
# Neither trafilatura (HTML-only) nor Playwright (a PDF response triggers a
# browser download event and raises) can read a PDF -- see page_fetcher.py's
# module docstring. These cover both ways a PDF is detected: an obvious
# ".pdf" URL, and a URL without that extension whose server reports
# Content-Type: application/pdf (the case caught live via Tavily).


def test_looks_like_pdf_url_detects_pdf_extension():
    assert page_fetcher._looks_like_pdf_url("https://example.test/report.pdf")
    assert page_fetcher._looks_like_pdf_url("https://example.test/report.PDF?x=1")
    assert not page_fetcher._looks_like_pdf_url("https://example.test/page.html")


def test_fetch_one_routes_pdf_url_straight_to_pdf_path_never_playwright(monkeypatch):
    monkeypatch.setattr(page_fetcher, "_fetch_pdf", lambda result, timeout: page_fetcher.FetchedPage(
        url=result.url, title=result.title, text="pdf content " * 10,
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc), fetch_method="pdf",
    ))
    monkeypatch.setattr(page_fetcher, "_fetch_fast", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not try HTML fast path for a PDF URL")))
    monkeypatch.setattr(page_fetcher, "_fetch_with_playwright", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not try Playwright for a PDF URL")))

    page = page_fetcher._fetch_one(_result(url="https://example.test/report.pdf"), timeout_seconds=9)

    assert page.fetch_method == "pdf"
    assert page.error is None


def test_extract_pdf_page_returns_extracted_text_on_success():
    # A blank-page PDF extracts no real text via pypdf, so patch extract_text
    # directly rather than relying on real PDF content generation here.
    class _FakePage:
        def extract_text(self):
            return "real extracted words " * 10

    class _FakeReader:
        def __init__(self, *a, **kw):
            self.pages = [_FakePage(), _FakePage()]

    import agents.web_navigator.page_fetcher as pf_module
    original = pf_module.PdfReader
    pf_module.PdfReader = _FakeReader
    try:
        page = page_fetcher._extract_pdf_page(_result(), b"%PDF-fake-bytes")
    finally:
        pf_module.PdfReader = original

    assert page.fetch_method == "pdf"
    assert page.error is None
    assert "real extracted words" in page.text


def test_extract_pdf_page_reports_error_on_corrupt_pdf():
    page = page_fetcher._extract_pdf_page(_result(), b"not a real pdf")

    assert page.error is not None
    assert page.text == ""
    assert page.fetch_method == "pdf"


def test_extract_pdf_page_reports_error_when_too_little_text_extracted():
    """The scanned/image-only-PDF case: pypdf has no OCR, so extraction
    "succeeds" with empty/near-empty text. Must be reported as a failure,
    not silently returned as a near-empty success."""
    import agents.web_navigator.page_fetcher as pf_module

    class _FakePage:
        def extract_text(self):
            return "scan"

    class _FakeReader:
        def __init__(self, *a, **kw):
            self.pages = [_FakePage()]

    original = pf_module.PdfReader
    pf_module.PdfReader = _FakeReader
    try:
        page = page_fetcher._extract_pdf_page(_result(), b"%PDF-fake-bytes")
    finally:
        pf_module.PdfReader = original

    assert page.error is not None
    assert "scanned" in page.error or "little" in page.error


def test_fetch_fast_routes_to_pdf_extraction_when_content_type_says_pdf(monkeypatch):
    """The URL doesn't end in .pdf, but the server says it's one -- must not
    be handed to trafilatura, and must not fall through to Playwright."""
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **kw: _FakeResponse("ignored", content_type="application/pdf; charset=binary", content=b"%PDF-fake"),
    )
    monkeypatch.setattr(
        page_fetcher, "_extract_pdf_page",
        lambda result, pdf_bytes: page_fetcher.FetchedPage(
            url=result.url, title=result.title, text="pdf text " * 10,
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc), fetch_method="pdf",
        ),
    )

    page = page_fetcher._fetch_fast(_result(url="https://example.test/download?id=42"), timeout_seconds=9)

    assert page is not None
    assert page.fetch_method == "pdf"
