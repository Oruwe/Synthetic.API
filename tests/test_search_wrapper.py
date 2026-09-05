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
    monkeypatch.setattr(search_wrapper.time, "sleep", lambda s: None)

    def _raise(*a, **kw):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(httpx, "post", _raise)

    results = search_wrapper.search("some question")
    assert results == []


def test_transient_failure_is_retried_up_to_max_attempts(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")
    monkeypatch.setattr(search_wrapper.time, "sleep", lambda s: None)

    calls = {"count": 0}

    def _raise(*a, **kw):
        calls["count"] += 1
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(httpx, "post", _raise)

    results = search_wrapper.search("some question")

    assert results == []
    assert calls["count"] == search_wrapper._MAX_ATTEMPTS


def test_transient_failure_succeeds_on_a_later_attempt(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")
    monkeypatch.setattr(search_wrapper.time, "sleep", lambda s: None)

    calls = {"count": 0}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"title": "OK", "url": "https://ok.test"}]}

    def _flaky(*a, **kw):
        calls["count"] += 1
        if calls["count"] < 2:
            raise httpx.ConnectTimeout("timed out")
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", _flaky)

    results = search_wrapper.search("some question")

    assert [r.url for r in results] == ["https://ok.test"]
    assert calls["count"] == 2


def test_client_error_status_is_not_retried(monkeypatch):
    """A 401 (bad API key) or 400 (bad request) will never succeed on
    retry -- fail immediately instead of burning attempts/quota."""
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")
    monkeypatch.setattr(search_wrapper.time, "sleep", lambda s: None)

    calls = {"count": 0}

    class UnauthorizedResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError("401", request=None, response=httpx.Response(401, request=httpx.Request("POST", "https://api.tavily.com/search")))

    def _unauthorized(*a, **kw):
        calls["count"] += 1
        return UnauthorizedResponse()

    monkeypatch.setattr(httpx, "post", _unauthorized)

    results = search_wrapper.search("some question")

    assert results == []
    assert calls["count"] == 1  # no retries on a non-retryable client error


def test_server_error_status_is_retried(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")
    monkeypatch.setattr(search_wrapper.time, "sleep", lambda s: None)

    calls = {"count": 0}

    class ServerErrorResponse:
        def raise_for_status(self):
            calls["count"] += 1
            raise httpx.HTTPStatusError("503", request=None, response=httpx.Response(503, request=httpx.Request("POST", "https://api.tavily.com/search")))

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: ServerErrorResponse())

    results = search_wrapper.search("some question")

    assert results == []
    assert calls["count"] == search_wrapper._MAX_ATTEMPTS


def test_malformed_json_returns_empty_list_not_raise(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")
    monkeypatch.setattr(search_wrapper.time, "sleep", lambda s: None)

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
