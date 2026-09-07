"""Tests for vision_wrapper's JSON parsing and per-screenshot error
isolation. The actual OpenRouter vision call is mocked -- no network,
no API key needed, consistent with the rest of the offline test suite.
"""

import agents.common.vision_wrapper as vision_wrapper
from agents.common.models.action import ActionStep
from agents.common.vision_wrapper import _parse_json_response, analyze_screenshot, decide_next_action


def test_parse_json_response_plain_json():
    parsed = _parse_json_response('{"title": "X", "summary": "Y", "key_facts": ["a", "b"]}')
    assert parsed == {"title": "X", "summary": "Y", "key_facts": ["a", "b"]}


def test_parse_json_response_strips_markdown_fence():
    parsed = _parse_json_response('```json\n{"title": "X", "summary": "Y"}\n```')
    assert parsed["title"] == "X"


def test_parse_json_response_falls_back_to_raw_text_summary():
    parsed = _parse_json_response("this is not json")
    assert parsed == {"summary": "this is not json"}


def test_analyze_screenshot_returns_valid_finding(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vision_wrapper._vision_agent,
        "analyze",
        lambda image_ref, prompt, *, run_id, node_id: '{"title": "Example", "summary": "It works.", "key_facts": ["fact one"]}',
    )
    finding = analyze_screenshot(
        "https://example.test", "Example Title", str(tmp_path / "shot.png"), "some query", run_id="r1", node_id="n1"
    )
    assert finding.url == "https://example.test"
    assert finding.title == "Example"
    assert finding.summary == "It works."
    assert finding.key_facts == ["fact one"]
    assert finding.flags == []


def test_analyze_screenshot_isolates_failure_without_raising(monkeypatch, tmp_path):
    """Regression-style test mirroring extractor.py's per-row isolation:
    one screenshot's analysis failing must not raise out of the function,
    since the batch handler (research_handlers.py) relies on that to keep
    processing the rest of the screenshots."""

    def _boom(image_ref, prompt, *, run_id, node_id):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(vision_wrapper._vision_agent, "analyze", _boom)

    finding = analyze_screenshot(
        "https://broken.test", "Broken", str(tmp_path / "shot.png"), "q", run_id="r1", node_id="n1"
    )
    assert finding.url == "https://broken.test"
    assert "analysis failed" in finding.summary
    assert finding.key_facts == []


# --- decide_next_action (ambient RPA action path) ------------------------


def test_decide_next_action_returns_a_valid_click_step(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vision_wrapper._vision_agent,
        "decide_action",
        lambda image_ref, prompt, *, run_id, node_id: '{"kind": "click", "x": 500, "y": 500, "reasoning": "click search"}',
    )

    step = decide_next_action(str(tmp_path / "shot.png"), "find the cheapest flight", [], run_id="r1", node_id="n1")

    assert step.kind == "click"
    assert step.x == 500
    assert step.y == 500
    assert step.reasoning == "click search"


def test_decide_next_action_returns_done_step(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vision_wrapper._vision_agent,
        "decide_action",
        lambda image_ref, prompt, *, run_id, node_id: '{"kind": "done", "reasoning": "goal satisfied"}',
    )

    step = decide_next_action(str(tmp_path / "shot.png"), "find the cheapest flight", [], run_id="r1", node_id="n1")

    assert step.kind == "done"
    assert step.x is None


def test_decide_next_action_falls_back_to_stuck_on_unrecognized_kind(monkeypatch, tmp_path):
    """The model returning something outside the known action vocabulary
    (e.g. it invents "submit") must not propagate an invalid ActionStep --
    it must come back as "stuck" so the loop stops cleanly instead of the
    executor later hitting an unhandled kind."""
    monkeypatch.setattr(
        vision_wrapper._vision_agent,
        "decide_action",
        lambda image_ref, prompt, *, run_id, node_id: '{"kind": "submit", "reasoning": "submit the form"}',
    )

    step = decide_next_action(str(tmp_path / "shot.png"), "buy the item", [], run_id="r1", node_id="n1")

    assert step.kind == "stuck"


def test_decide_next_action_surfaces_the_raw_response_when_kind_is_unrecognized(monkeypatch, tmp_path):
    """Regression test: found live, against a real free-tier model, that
    an unparseable/empty response produced a "stuck" step with an EMPTY
    reasoning field and nothing in the logs beyond `kind=None` -- no way
    to tell whether the model call returned truly empty content, JSON
    shaped differently than expected, or something else entirely. The
    resulting ActionStep's reasoning must actually surface what came
    back, not just say nothing happened."""
    monkeypatch.setattr(
        vision_wrapper._vision_agent,
        "decide_action",
        lambda image_ref, prompt, *, run_id, node_id: '{"unexpected_field": "the model ignored the schema"}',
    )

    step = decide_next_action(str(tmp_path / "shot.png"), "buy the item", [], run_id="r1", node_id="n1")

    assert step.kind == "stuck"
    assert step.reasoning != ""
    assert "unexpected_field" in step.reasoning


def test_decide_next_action_falls_back_to_stuck_on_malformed_response(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vision_wrapper._vision_agent,
        "decide_action",
        lambda image_ref, prompt, *, run_id, node_id: "not json at all",
    )

    step = decide_next_action(str(tmp_path / "shot.png"), "buy the item", [], run_id="r1", node_id="n1")

    assert step.kind == "stuck"


def test_decide_next_action_isolates_a_raising_call_as_stuck(monkeypatch, tmp_path):
    def _boom(image_ref, prompt, *, run_id, node_id):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(vision_wrapper._vision_agent, "decide_action", _boom)

    step = decide_next_action(str(tmp_path / "shot.png"), "buy the item", [], run_id="r1", node_id="n1")

    assert step.kind == "stuck"
    assert "vision call failed" in step.reasoning


def test_decide_next_action_includes_prior_history_in_the_prompt(monkeypatch, tmp_path):
    """The model needs to know what's already been tried, or it will repeat
    the same action forever -- verify the history actually reaches the
    prompt text passed to the model."""
    captured = {}

    def fake_decide(image_ref, prompt, *, run_id, node_id):
        captured["prompt"] = prompt
        return '{"kind": "done", "reasoning": "done"}'

    monkeypatch.setattr(vision_wrapper._vision_agent, "decide_action", fake_decide)
    history = [ActionStep(kind="click", x=100, y=200, reasoning="clicked search box")]

    decide_next_action(str(tmp_path / "shot.png"), "find a flight", history, run_id="r1", node_id="n1")

    assert "click" in captured["prompt"]
    assert "(100,200)" in captured["prompt"]
    assert "find a flight" in captured["prompt"]
