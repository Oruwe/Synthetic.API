"""Tests for the per-domain rate limiter: back-to-back requests to the
SAME domain get throttled; different domains don't block each other."""

import agents.web_navigator.rate_limiter as rate_limiter


def test_second_request_to_same_domain_is_throttled(monkeypatch):
    rate_limiter._last_request_at.clear()
    sleeps = []
    monkeypatch.setattr(rate_limiter.time, "sleep", lambda s: sleeps.append(s))

    rate_limiter.throttle("https://example.test/page1")
    rate_limiter.throttle("https://example.test/page2")

    assert len(sleeps) == 1
    assert sleeps[0] > 0


def test_different_domains_are_not_throttled_against_each_other(monkeypatch):
    rate_limiter._last_request_at.clear()
    sleeps = []
    monkeypatch.setattr(rate_limiter.time, "sleep", lambda s: sleeps.append(s))

    rate_limiter.throttle("https://a.test/page")
    rate_limiter.throttle("https://b.test/page")

    assert sleeps == []


def test_request_after_the_minimum_gap_is_not_throttled(monkeypatch):
    rate_limiter._last_request_at.clear()
    sleeps = []
    monkeypatch.setattr(rate_limiter.time, "sleep", lambda s: sleeps.append(s))

    fake_clock = {"t": 0.0}
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: fake_clock["t"])

    rate_limiter.throttle("https://example.test/page1")
    fake_clock["t"] += rate_limiter._MIN_GAP_SECONDS + 0.1
    rate_limiter.throttle("https://example.test/page2")

    assert sleeps == []


def test_url_with_no_domain_is_a_noop():
    rate_limiter._last_request_at.clear()
    rate_limiter.throttle("not-a-url")  # must not raise
