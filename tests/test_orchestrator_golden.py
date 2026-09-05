"""Golden cases for the live planner: any non-empty transcript becomes a
2-node fetch_pages -> embed_pages plan built around whatever Tavily
returned. search_wrapper.search() is mocked so these stay offline and
deterministic -- no network, no API key, consistent with the rest of the
test suite.
"""

import pytest

import agents.orchestrator.planner as planner
from agents.common.models.dag import NodeType
from agents.common.models.research import SearchResult


def _node_types(plan):
    return [n.type for n in plan.nodes]


def test_any_transcript_produces_fetch_then_embed_plan(monkeypatch):
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: [])
    plan = planner.build_plan("Check the shipping portal for delayed orders and email me a summary")
    assert plan.status == "planned"
    assert _node_types(plan) == [NodeType.FETCH_PAGES, NodeType.EMBED_PAGES]
    assert [e.from_node for e in plan.edges] == ["fetch"]
    assert [e.to_node for e in plan.edges] == ["embed"]


def test_bare_question_also_produces_a_plan_not_no_capability(monkeypatch):
    """Regression boundary from before the pivot: previously a bare
    question without a trigger phrase stayed `no_capability`. The new
    design has no keyword gate -- any non-empty transcript is a question
    worth searching for."""
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: [])
    plan = planner.build_plan("What's the weather today?")
    assert plan.status == "planned"


def test_empty_transcript_still_raises():
    with pytest.raises(planner.PlannerInputError):
        planner.build_plan("")


def test_whitespace_only_transcript_still_raises():
    with pytest.raises(planner.PlannerInputError):
        planner.build_plan("   ")


def test_search_results_are_embedded_into_fetch_node_params(monkeypatch):
    fake_results = [
        SearchResult(title="Result A", url="https://a.test", snippet="preview a"),
        SearchResult(title="Result B", url="https://b.test"),
    ]
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: fake_results)

    plan = planner.build_plan("Search the web for open source vision models")
    fetch_node = plan.nodes[0]
    assert fetch_node.params["question"] == "Search the web for open source vision models"
    urls = [r["url"] for r in fetch_node.params["search_results"]]
    assert urls == ["https://a.test", "https://b.test"]


def test_search_failure_still_produces_a_runnable_plan(monkeypatch):
    """search_wrapper.search() never raises (it catches everything itself),
    but this asserts the planner doesn't additionally assume a non-empty
    result list anywhere -- a zero-result search must still produce a
    valid, executable plan (see test_dag_executor / page_handlers for what
    happens to it: a graceful "no sources found" answer, not a crash)."""
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: [])
    plan = planner.build_plan("Some obscure question with no results")
    assert plan.status == "planned"
    assert plan.nodes[0].params["search_results"] == []


def test_plan_uses_configured_circuit_breaker_threshold(monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: [])
    plan = planner.build_plan("Any question")
    assert plan.circuit_breaker_threshold == settings.dag_circuit_breaker_threshold


def test_fetch_node_has_a_generous_timeout_and_single_retry(monkeypatch):
    """fetch_pages can visit several real URLs with an internal fast/fallback
    retry per URL already -- a full node-level retry on top adds little and
    would double an already-generous timeout in the worst case, so this is
    intentionally 1, not the model default of 3."""
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: [])
    plan = planner.build_plan("Any question")
    assert plan.nodes[0].max_retries == 1
    assert plan.nodes[0].timeout_seconds >= 60
