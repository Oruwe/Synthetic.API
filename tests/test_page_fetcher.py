"""Tests for page_fetcher's fast-path/fallback decision and per-URL
isolation. httpx and the Playwright fallback are both mocked -- no real
network or browser needed, consistent with the rest of the offline suite.
"""

import httpx

import agents.web_navigator.page_fetcher as page_fetcher
from agents.common.models.research import SearchResult


def _result(url="https://example.test", title="Example"):
    return SearchResult(title=title, url=url)


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_fast_path_succeeds_for_normal_page(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResponse("<html>...</html>"))
    monkeypatch.setattr(page_fetcher.trafilatura, "extract", lambda html: "x" * 500)

    page = page_fetcher._fetch_fast(_result(), timeout_seconds=9)

    assert page is not None
    assert page.fetch_method == "http"
    assert page.error is None
    assert len(page.text) == 500


def test_fast_path_returns_none_when_extracted_text_too_short(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResponse("<html>...</html>"))
    monkeypatch.setattr(page_fetcher.trafilatura, "extract", lambda html: "too short")

    page = page_fetcher._fetch_fast(_result(), timeout_seconds=9)

    assert page is None  # signals the caller to try the Playwright fallback


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
