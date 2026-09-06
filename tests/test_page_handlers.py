"""Tests for the live DAG node handlers: fetch_pages (delegates to
page_fetcher, handles zero search results gracefully) and embed_pages
(per-page isolation -- one page's embedding failing must not lose the
others)."""

from datetime import datetime, timezone

import agents.web_navigator.page_handlers as page_handlers
from agents.common.models.page import FetchedPage
from agents.orchestrator.executor import RunContext


def test_fetch_pages_handles_zero_search_results(monkeypatch):
    called = {"count": 0}
    monkeypatch.setattr(page_handlers.page_fetcher, "fetch_pages", lambda results: called.__setitem__("count", called["count"] + 1))

    node = type("N", (), {"params": {"question": "q", "search_results": []}})()
    ctx = RunContext(run_id="r1")
    result = page_handlers.handle_fetch_pages(node, ctx)

    assert called["count"] == 0  # never even calls the fetcher with nothing to fetch
    assert ctx.data["fetched_pages"] == []
    assert "no search results" in result


def test_fetch_pages_delegates_to_page_fetcher_with_results(monkeypatch):
    fake_pages = [
        FetchedPage(url="https://a.test", title="A", text="content", timestamp=datetime.now(timezone.utc), fetch_method="http")
    ]
    monkeypatch.setattr(page_handlers.page_fetcher, "fetch_pages", lambda results: fake_pages)

    node = type("N", (), {"params": {"question": "q", "search_results": [{"title": "A", "url": "https://a.test"}]}})()
    ctx = RunContext(run_id="r1")
    result = page_handlers.handle_fetch_pages(node, ctx)

    assert ctx.data["fetched_pages"] == fake_pages
    assert "1/1" in result
    assert node.params["sources_succeeded"] == 1


def test_fetch_pages_records_the_actual_success_count_onto_node_params(monkeypatch):
    """synthesizer/main.py reads this back to compute sources_succeeded --
    must be the real fetch-success count, not an approximation derived
    later from however many distinct URLs made it into the top-k
    semantically-retrieved chunks (which undercounts whenever more URLs
    fetch successfully than settings.research_top_k can represent)."""
    fake_pages = [
        FetchedPage(url="https://a.test", title="A", text="content", timestamp=datetime.now(timezone.utc), fetch_method="http"),
        FetchedPage(url="https://b.test", title="B", text="", timestamp=datetime.now(timezone.utc), fetch_method="http", error="404"),
        FetchedPage(url="https://c.test", title="C", text="content", timestamp=datetime.now(timezone.utc), fetch_method="http"),
    ]
    monkeypatch.setattr(page_handlers.page_fetcher, "fetch_pages", lambda results: fake_pages)

    search_results = [{"title": "A", "url": "https://a.test"}, {"title": "B", "url": "https://b.test"}, {"title": "C", "url": "https://c.test"}]
    node = type("N", (), {"params": {"question": "q", "search_results": search_results}})()
    ctx = RunContext(run_id="r1")
    page_handlers.handle_fetch_pages(node, ctx)

    assert node.params["sources_succeeded"] == 2  # a and c succeeded, b failed


def test_embed_pages_isolates_one_page_failure_from_the_rest(monkeypatch):
    good_page = FetchedPage(url="https://good.test", title="Good", text="content", timestamp=datetime.now(timezone.utc), fetch_method="http")
    bad_page = FetchedPage(url="https://bad.test", title="Bad", text="content", timestamp=datetime.now(timezone.utc), fetch_method="http")

    def fake_upsert(page, question, run_id):
        if page.url == "https://bad.test":
            raise RuntimeError("qdrant write failed")
        return ["point-1", "point-2"]

    monkeypatch.setattr(page_handlers.qdrant_store, "upsert_page_chunks", fake_upsert)

    ctx = RunContext(run_id="r1", data={"question": "q", "fetched_pages": [good_page, bad_page]})
    result = page_handlers.handle_embed_pages(node=None, ctx=ctx)

    assert "2 chunks" in result
    assert "1/2 pages" in result
