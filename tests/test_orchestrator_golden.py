"""Golden cases for transcript -> DAGPlan. planner.py's base layer is
rule-based and needs no LLM/API key, so these assertions are fully
deterministic and offline -- no mocking required for the cases below.
(If/when an LLM-assisted refinement layer is added on top for paraphrase
robustness beyond these patterns, mock agents.common.lyzr_wrapper there.)
"""

from agents.common.models.dag import NodeType
from agents.orchestrator import planner


def _node_types(plan):
    return [n.type for n in plan.nodes]


def test_standard_shipping_request_produces_scrape_extract_embed_plan():
    plan = planner.build_plan("Check the shipping portal for delayed orders and email me a summary")
    assert plan.status == "planned"
    assert _node_types(plan) == [NodeType.SCRAPE_PORTAL, NodeType.EXTRACT_VALIDATE, NodeType.EMBED_STORE]
    assert [e.from_node for e in plan.edges] == ["scrape", "extract"]


def test_urgent_request_sets_priority_param():
    plan = planner.build_plan("Check the shipping portal for delayed orders, this is urgent")
    assert _node_types(plan) == [NodeType.SCRAPE_PORTAL, NodeType.EXTRACT_VALIDATE, NodeType.EMBED_STORE]
    assert plan.nodes[0].params["priority"] == "high"


def test_unrelated_request_returns_no_capability():
    plan = planner.build_plan("What's the weather today?")
    assert plan.status == "no_capability"
    assert _node_types(plan) == [NodeType.CLARIFY_UNSUPPORTED]


def test_unconfigured_system_returns_no_capability_not_shipping_plan():
    plan = planner.build_plan("Check the CRM for delayed leads")
    assert plan.status == "no_capability"
    assert _node_types(plan) == [NodeType.CLARIFY_UNSUPPORTED]


def test_supported_task_with_unsupported_subintent_is_flagged_not_dropped():
    plan = planner.build_plan("Check delayed orders and also cancel order 1234")
    assert plan.status == "planned"
    assert _node_types(plan) == [NodeType.SCRAPE_PORTAL, NodeType.EXTRACT_VALIDATE, NodeType.EMBED_STORE]
    assert plan.unsupported_subintents == ["cancel_order"]


def test_paraphrased_request_maps_to_same_plan_shape():
    plan = planner.build_plan("Are there any late shipments? Let me know")
    assert plan.status == "planned"
    assert _node_types(plan) == [NodeType.SCRAPE_PORTAL, NodeType.EXTRACT_VALIDATE, NodeType.EMBED_STORE]


def test_empty_transcript_raises_without_crashing():
    import pytest

    with pytest.raises(planner.PlannerInputError):
        planner.build_plan("")


def test_customer_filter_is_extracted():
    plan = planner.build_plan("Check delayed orders for customer Acme")
    assert plan.nodes[0].params["customer_filter"] == "Acme"
