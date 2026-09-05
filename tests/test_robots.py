"""Tests for robots.txt respect: allow/disallow decisions, per-domain
caching, and fail-open behavior on any fetch/parse problem -- a robots.txt
outage must never block an otherwise-legitimate fetch."""

import httpx

import agents.web_navigator.robots as robots


def _fake_response(status_code, text=""):
    class _R:
        def __init__(self):
            self.status_code = status_code
            self.text = text

    return _R()


def test_disallowed_path_is_blocked(monkeypatch):
    robots._cache.clear()
    robots_txt = "User-agent: *\nDisallow: /private/\n"
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _fake_response(200, robots_txt))

    assert robots.is_allowed("https://example.test/private/secret") is False
    assert robots.is_allowed("https://example.test/public/page") is True


def test_missing_robots_txt_allows_everything(monkeypatch):
    robots._cache.clear()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _fake_response(404))

    assert robots.is_allowed("https://example.test/anything") is True


def test_robots_txt_fetch_failure_fails_open(monkeypatch):
    robots._cache.clear()

    def _raise(*a, **kw):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(httpx, "get", _raise)

    assert robots.is_allowed("https://unreachable.test/page") is True


def test_robots_txt_is_cached_per_domain(monkeypatch):
    robots._cache.clear()
    calls = {"count": 0}

    def _get(*a, **kw):
        calls["count"] += 1
        return _fake_response(200, "User-agent: *\nAllow: /\n")

    monkeypatch.setattr(httpx, "get", _get)

    robots.is_allowed("https://example.test/a")
    robots.is_allowed("https://example.test/b")
    robots.is_allowed("https://example.test/c")

    assert calls["count"] == 1  # fetched once, reused for all three checks


def test_different_domains_are_fetched_separately(monkeypatch):
    robots._cache.clear()
    calls = {"count": 0}

    def _get(*a, **kw):
        calls["count"] += 1
        return _fake_response(200, "User-agent: *\nAllow: /\n")

    monkeypatch.setattr(httpx, "get", _get)

    robots.is_allowed("https://a.test/page")
    robots.is_allowed("https://b.test/page")

    assert calls["count"] == 2
