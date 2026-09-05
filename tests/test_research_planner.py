"""Golden cases for the Web-Researcher intent: transcript -> DAGPlan.
Deterministic/offline, same philosophy as test_orchestrator_golden.py."""

from agents.common.models.dag import NodeType
from agents.orchestrator import planner

_EXPECTED_RESEARCH_CHAIN = [
    NodeType.SEARCH_WEB,
    NodeType.CAPTURE_SCREENSHOTS,
    NodeType.ANALYZE_SCREENSHOTS,
    NodeType.EMBED_CANDIDATES,
    NodeType.CURATE_KNOWLEDGE,
]


def _node_types(plan):
    return [n.type for n in plan.nodes]


def test_search_the_web_for_triggers_research_chain_with_extracted_query():
    plan = planner.build_plan("Search the web for the latest ISRO hackathon rules")
    assert plan.status == "planned"
    assert _node_types(plan) == _EXPECTED_RESEARCH_CHAIN
    assert plan.nodes[0].params["query"] == "the latest ISRO hackathon rules"


def test_look_up_triggers_research_chain():
    plan = planner.build_plan("Look up the current price of DeepSeek API access")
    assert _node_types(plan) == _EXPECTED_RESEARCH_CHAIN
    assert "DeepSeek" in plan.nodes[0].params["query"]


def test_research_verb_triggers_research_chain():
    plan = planner.build_plan("Research quantum computing breakthroughs in 2026")
    assert _node_types(plan) == _EXPECTED_RESEARCH_CHAIN
    assert plan.nodes[0].params["query"] == "quantum computing breakthroughs in 2026"


def test_find_information_about_triggers_research_chain():
    plan = planner.build_plan("Find information about Qwen2.5-VL")
    assert _node_types(plan) == _EXPECTED_RESEARCH_CHAIN
    assert plan.nodes[0].params["query"] == "Qwen2.5-VL"


def test_research_chain_edges_are_linear():
    plan = planner.build_plan("Search the web for open source vision models")
    pairs = [(e.from_node, e.to_node) for e in plan.edges]
    assert pairs == [
        ("search", "screenshot"),
        ("screenshot", "analyze"),
        ("analyze", "embed_candidates"),
        ("embed_candidates", "curate"),
    ]


def test_bare_question_without_trigger_phrase_stays_no_capability():
    """A plain question must NOT silently trigger a web search -- only an
    explicit research trigger phrase does. Same golden case as
    test_orchestrator_golden.py, re-asserted here since it's the boundary
    this feature must not regress."""
    plan = planner.build_plan("What's the weather today?")
    assert plan.status == "no_capability"


def test_shipping_request_is_unaffected_by_research_detection():
    plan = planner.build_plan("Check the shipping portal for delayed orders and email me a summary")
    assert _node_types(plan) == [NodeType.SCRAPE_PORTAL, NodeType.EXTRACT_VALIDATE, NodeType.EMBED_STORE]


def test_research_plan_uses_configured_circuit_breaker_threshold():
    from agents.common.config import settings

    plan = planner.build_plan("Research the history of Qdrant")
    assert plan.circuit_breaker_threshold == settings.dag_circuit_breaker_threshold
