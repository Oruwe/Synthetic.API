"""Golden cases for the planner's action-vs-question routing: a physical-
task transcript ("book", "buy", "sign me up for", ...) must produce a
single execute_action node instead of the research fetch_pages ->
embed_pages chain, and a fresh action needs Tavily's top result as a
starting URL -- with no search results at all, there's nothing to act on,
so it must fall back to no_capability rather than handing the action
executor a start_url of None.

search_wrapper.search() is mocked so these stay offline and deterministic,
same convention as test_orchestrator_golden.py.
"""

import agents.orchestrator.planner as planner
from agents.common.models.dag import NodeType
from agents.common.models.research import SearchResult


def _search(url="https://portal.test/signup"):
    return [SearchResult(title="Sign Up", url=url, snippet="Create an account")]


def test_book_intent_produces_a_single_execute_action_node(monkeypatch):
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: _search())
    plan = planner.build_plan("Book a table for two at an Italian restaurant tonight")
    assert plan.status == "planned"
    assert [n.type for n in plan.nodes] == [NodeType.EXECUTE_ACTION]


def test_action_node_carries_intent_and_start_url(monkeypatch):
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: _search("https://x.test/book"))
    plan = planner.build_plan("Book a table for two tonight")
    node = plan.nodes[0]
    assert node.params["intent"] == "Book a table for two tonight"
    assert node.params["start_url"] == "https://x.test/book"
    assert node.handler_key == "execute_action"


def test_action_node_never_retries_and_has_a_generous_timeout(monkeypatch):
    """A retried click/type is not idempotent like a retried HTTP fetch --
    this must stay 1 no matter what."""
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: _search())
    plan = planner.build_plan("Sign me up for the newsletter")
    node = plan.nodes[0]
    assert node.max_retries == 1
    assert node.timeout_seconds >= 60


def test_action_intent_with_no_search_results_falls_back_to_no_capability(monkeypatch):
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: [])
    plan = planner.build_plan("Buy me a new pair of running shoes")
    assert plan.status == "no_capability"
    assert plan.nodes == []


def test_various_action_verbs_are_all_recognized(monkeypatch):
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: _search())
    phrases = [
        "Buy a new pair of shoes",
        "Purchase a plane ticket to Goa",
        "Order a pizza for dinner",
        "Register for the ISRO hackathon",
        "Reserve a table at the cafe",
        "Schedule a dentist appointment",
        "Apply for the internship posting",
        "Subscribe to the newsletter",
        "Fill out the contact form",
        "Submit the application",
        "Add this shirt to cart",
        "Renew my library membership",
    ]
    for phrase in phrases:
        plan = planner.build_plan(phrase)
        assert plan.status == "planned", phrase
        assert [n.type for n in plan.nodes] == [NodeType.EXECUTE_ACTION], phrase


def test_research_questions_are_not_misrouted_as_actions(monkeypatch):
    """Regression coverage for the classifier's two false-positive guards:
    a leading question word always wins regardless of what follows (the
    "best watch to buy" case is a real query from this project's own bug
    history -- see ui/app.py's polling-race fix), and a substring match
    inside a longer word (\\border\\b must not fire on "orders") never
    counts as the verb either way."""
    monkeypatch.setattr(planner.search_wrapper, "search", lambda q, max_results=None: [])
    questions = [
        "What is the best watch to buy under $200?",
        "Check the shipping portal for delayed orders",
        "What's the weather today?",
        "Who won the last cricket world cup?",
        "How do I register a business in India?",
    ]
    for question in questions:
        plan = planner.build_plan(question)
        assert [n.type for n in plan.nodes] == [NodeType.FETCH_PAGES, NodeType.EMBED_PAGES], question
