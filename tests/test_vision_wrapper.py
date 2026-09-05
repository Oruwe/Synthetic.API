"""Tests for vision_wrapper's JSON parsing and per-screenshot error
isolation. The actual OpenRouter vision call is mocked -- no network,
no API key needed, consistent with the rest of the offline test suite.
"""

import agents.common.vision_wrapper as vision_wrapper
from agents.common.vision_wrapper import _parse_json_response, analyze_screenshot


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
