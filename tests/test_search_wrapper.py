"""Tests for the Tavily search wrapper's non-negotiable fail-safe
behavior: no API key, an HTTP error, or a malformed response must all
degrade to an empty list, logged, never a raised exception."""

import httpx

import agents.common.search_wrapper as search_wrapper
from agents.common.config import settings


def test_no_api_key_returns_empty_list_without_calling_out(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "")
    called = {"count": 0}
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: called.__setitem__("count", called["count"] + 1))

    results = search_wrapper.search("some question")

    assert results == []
    assert called["count"] == 0


def test_successful_response_is_parsed_into_search_results(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": "First result", "url": "https://a.test", "content": "preview a"},
                    {"title": "Second result", "url": "https://b.test", "content": "preview b"},
                ]
            }

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: FakeResponse())

    results = search_wrapper.search("some question", max_results=5)

    assert [r.url for r in results] == ["https://a.test", "https://b.test"]
    assert results[0].snippet == "preview a"


def test_http_error_returns_empty_list_not_raise(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")

    def _raise(*a, **kw):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(httpx, "post", _raise)

    results = search_wrapper.search("some question")
    assert results == []


def test_malformed_json_returns_empty_list_not_raise(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")

    class BadResponse:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: BadResponse())

    results = search_wrapper.search("some question")
    assert results == []


def test_results_missing_url_or_title_are_skipped(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"title": "No URL"}, {"url": "https://ok.test", "title": "OK"}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: FakeResponse())

    results = search_wrapper.search("some question")
    assert [r.url for r in results] == ["https://ok.test"]


def test_max_results_caps_the_returned_list(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"title": f"R{i}", "url": f"https://{i}.test"} for i in range(10)]}

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: FakeResponse())

    results = search_wrapper.search("some question", max_results=3)
    assert len(results) == 3
