"""Tests for the live DAG node handlers: fetch_pages (delegates to
page_fetcher, handles zero search results gracefully, and owns the
human-in-the-loop gated-content pause/resume) and embed_pages (per-page
isolation -- one page's embedding failing must not lose the others)."""

from datetime import datetime, timezone

import pytest

import agents.web_navigator.page_handlers as page_handlers
from agents.common.models.action import ActionWorkflow
from agents.common.models.page import FetchedPage
from agents.orchestrator.executor import AwaitingHumanInputError, RunContext


def test_fetch_pages_handles_zero_search_results(monkeypatch):
    called = {"count": 0}
    monkeypatch.setattr(page_handlers.page_fetcher, "fetch_pages", lambda results: called.__setitem__("count", called["count"] + 1))

    node = type("N", (), {"id": "fetch", "params": {"question": "q", "search_results": []}})()
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

    node = type("N", (), {"id": "fetch", "params": {"question": "q", "search_results": [{"title": "A", "url": "https://a.test"}]}})()
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
    node = type("N", (), {"id": "fetch", "params": {"question": "q", "search_results": search_results}})()
    ctx = RunContext(run_id="r1")
    page_handlers.handle_fetch_pages(node, ctx)

    assert node.params["sources_succeeded"] == 2  # a and c succeeded, b failed


def _gated_page(gate_reason="Subscribe to continue reading"):
    return FetchedPage(
        url="https://gated.test", title="Gated", text="", timestamp=datetime.now(timezone.utc),
        fetch_method="http", gated=True, gate_reason=gate_reason,
    )


def _node(search_results, node_id="fetch"):
    return type("N", (), {"id": node_id, "params": {"question": "q", "search_results": search_results}})()


def test_fetch_pages_pauses_when_a_gated_page_is_the_only_source(monkeypatch):
    monkeypatch.setattr(page_handlers.page_fetcher, "fetch_pages", lambda results: [_gated_page()])
    node = _node([{"title": "Gated", "url": "https://gated.test"}])
    ctx = RunContext(run_id="r1")

    with pytest.raises(AwaitingHumanInputError) as exc_info:
        page_handlers.handle_fetch_pages(node, ctx)

    assert exc_info.value.fields == ["email"]
    assert exc_info.value.url == "https://gated.test"


def test_fetch_pages_requests_a_password_for_a_login_style_gate(monkeypatch):
    monkeypatch.setattr(
        page_handlers.page_fetcher, "fetch_pages", lambda results: [_gated_page(gate_reason="Sign in to continue")]
    )
    node = _node([{"title": "Gated", "url": "https://gated.test"}])

    with pytest.raises(AwaitingHumanInputError) as exc_info:
        page_handlers.handle_fetch_pages(node, RunContext(run_id="r1"))

    assert exc_info.value.fields == ["email", "password"]


def test_fetch_pages_does_not_pause_when_another_source_already_has_content(monkeypatch):
    """A gate on ONE of several candidate sources must not interrupt the
    run if the OTHERS already answer the question -- only pause when the
    gate is actually blocking the answer."""
    good_page = FetchedPage(
        url="https://good.test", title="Good", text="real content here", timestamp=datetime.now(timezone.utc),
        fetch_method="http",
    )
    gated_page = _gated_page()
    monkeypatch.setattr(page_handlers.page_fetcher, "fetch_pages", lambda results: [good_page, gated_page])
    node = _node([{"title": "Good", "url": "https://good.test"}, {"title": "Gated", "url": "https://gated.test"}])
    ctx = RunContext(run_id="r1")

    result = page_handlers.handle_fetch_pages(node, ctx)  # must not raise

    assert "fetched" in result
    assert ctx.data["fetched_pages"] == [good_page, gated_page]


def test_fetch_pages_uses_the_action_executor_on_resume_with_an_email(monkeypatch):
    monkeypatch.setattr(page_handlers.page_fetcher, "fetch_pages", lambda results: [_gated_page()])
    captured = {}

    def fake_loop(intent, start_url, run_id, on_success_extract=None):
        captured["intent"] = intent
        captured["start_url"] = start_url
        return ActionWorkflow(
            run_id=run_id, intent=intent, start_url=start_url, steps=[], success=True, refused_reason=None,
            created_at=datetime.now(timezone.utc), extracted_text="the real unlocked article text",
        )

    monkeypatch.setattr(page_handlers.action_executor, "execute_action_loop", fake_loop)

    node = _node([{"title": "Gated", "url": "https://gated.test"}])
    ctx = RunContext(run_id="r1", data={"human_provided_inputs": {"fetch": {"email": "judge@example.com"}}})

    page_handlers.handle_fetch_pages(node, ctx)

    assert "judge@example.com" in captured["intent"]
    assert captured["start_url"] == "https://gated.test"
    unlocked = ctx.data["fetched_pages"][0]
    assert unlocked.text == "the real unlocked article text"
    assert unlocked.error is None
    assert unlocked.gated is False  # the returned FetchedPage is a fresh, ungated one


def test_fetch_pages_uses_login_and_extract_on_resume_with_a_password(monkeypatch):
    monkeypatch.setattr(
        page_handlers.page_fetcher, "fetch_pages", lambda results: [_gated_page(gate_reason="Sign in to continue")]
    )
    captured = {}

    def fake_login(email, password, start_url, run_id, on_success_extract=None):
        captured["email"] = email
        captured["password"] = password
        return ActionWorkflow(
            run_id=run_id, intent="log in", start_url=start_url, steps=[], success=True, refused_reason=None,
            created_at=datetime.now(timezone.utc), extracted_text="member-only content",
        )

    monkeypatch.setattr(page_handlers.action_executor, "execute_login_and_extract", fake_login)

    node = _node([{"title": "Gated", "url": "https://gated.test"}])
    ctx = RunContext(
        run_id="r1",
        data={"human_provided_inputs": {"fetch": {"email": "judge@example.com", "password": "hunter2"}}},
    )

    page_handlers.handle_fetch_pages(node, ctx)

    assert captured["email"] == "judge@example.com"
    assert captured["password"] == "hunter2"
    assert ctx.data["fetched_pages"][0].text == "member-only content"


def test_fetch_pages_reports_a_clean_failure_when_the_gate_could_not_be_passed(monkeypatch):
    monkeypatch.setattr(page_handlers.page_fetcher, "fetch_pages", lambda results: [_gated_page()])
    monkeypatch.setattr(
        page_handlers.action_executor,
        "execute_action_loop",
        lambda intent, start_url, run_id, on_success_extract=None: ActionWorkflow(
            run_id=run_id, intent=intent, start_url=start_url, steps=[], success=False,
            refused_reason=None, created_at=datetime.now(timezone.utc),
        ),
    )

    node = _node([{"title": "Gated", "url": "https://gated.test"}])
    ctx = RunContext(run_id="r1", data={"human_provided_inputs": {"fetch": {"email": "judge@example.com"}}})

    page_handlers.handle_fetch_pages(node, ctx)

    unlocked = ctx.data["fetched_pages"][0]
    assert unlocked.error is not None
    assert unlocked.text == ""


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
