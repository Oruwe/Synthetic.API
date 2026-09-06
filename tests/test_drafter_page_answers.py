"""Tests for drafter.draft_answer: must always cite sources used and
always surface a partial-results caveat, regardless of whether the LLM
call succeeds or falls back to the template -- these are appended
deterministically, not left to the model's own compliance.

draft_answer() returns a DraftedAnswer (not a bare string): `.full` is
the backward-compatible flattened string (answer + footer, what
RunState.answer/notifier.notify() carry), `.text` is the same answer
with the footer stripped (what a UI should display/read aloud), and
`.sources` is the same citations as structured Source objects instead of
a string a caller would have to re-parse.
"""

from types import SimpleNamespace

from agents.synthesizer import drafter


def _point(url, title="T", text="some excerpt text", chunk_index=0, score=0.9):
    return SimpleNamespace(payload={"url": url, "title": title, "text": text, "chunk_index": chunk_index}, score=score)


def test_empty_chunks_returns_no_sources_found_message():
    result = drafter.draft_answer([], run_id="r1", question="my question", sources_attempted=3, sources_succeeded=0)
    assert "couldn't find" in result.full
    assert "my question" in result.full
    assert "0/3" in result.full
    assert result.text == result.full  # no footer to strip when there's nothing to cite
    assert result.sources == []


def test_answer_always_cites_sources_used_via_template_fallback(monkeypatch):
    monkeypatch.setattr(
        drafter._synthesizer_agent, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    chunks = [_point("https://a.test"), _point("https://b.test")]

    result = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=2, sources_succeeded=2)

    assert "https://a.test" in result.full
    assert "https://b.test" in result.full
    assert "Sources used:" in result.full
    assert {s.url for s in result.sources} == {"https://a.test", "https://b.test"}


def test_partial_results_caveat_appears_when_fewer_succeeded_than_attempted(monkeypatch):
    monkeypatch.setattr(drafter._synthesizer_agent, "run", lambda *a, **kw: "a drafted answer")
    chunks = [_point("https://a.test")]

    result = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=5, sources_succeeded=1)

    assert "Partial results" in result.full
    assert "1/5" in result.full
    assert result.sources_attempted == 5
    assert result.sources_succeeded == 1


def test_no_caveat_when_all_sources_succeeded(monkeypatch):
    monkeypatch.setattr(drafter._synthesizer_agent, "run", lambda *a, **kw: "a drafted answer")
    chunks = [_point("https://a.test")]

    result = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=1, sources_succeeded=1)

    assert "Partial results" not in result.full


def test_duplicate_urls_across_chunks_are_deduplicated_in_sources_list(monkeypatch):
    monkeypatch.setattr(
        drafter._synthesizer_agent, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    chunks = [_point("https://a.test", chunk_index=0), _point("https://a.test", chunk_index=1)]

    result = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=1, sources_succeeded=1)

    assert result.full.count("https://a.test") == 2  # once in the template body, once in "Sources used"
    assert len(result.sources) == 1  # deduplicated in the structured list, unlike the flattened text


def test_text_field_has_no_footer_but_full_does(monkeypatch):
    monkeypatch.setattr(drafter._synthesizer_agent, "run", lambda *a, **kw: "a clean drafted answer")
    chunks = [_point("https://a.test")]

    result = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=1, sources_succeeded=1)

    assert result.text == "a clean drafted answer"
    assert "Sources used:" not in result.text
    assert "Sources used:" in result.full


def test_sources_carry_title_snippet_and_score(monkeypatch):
    monkeypatch.setattr(drafter._synthesizer_agent, "run", lambda *a, **kw: "an answer")
    chunks = [_point("https://a.test", title="A Title", text="a snippet of text", score=0.75)]

    result = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=1, sources_succeeded=1)

    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.url == "https://a.test"
    assert source.title == "A Title"
    assert source.snippet == "a snippet of text"
    assert source.score == 0.75
