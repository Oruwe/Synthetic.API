"""Tests for drafter.draft_answer: must always cite sources used and
always surface a partial-results caveat, regardless of whether the LLM
call succeeds or falls back to the template -- these are appended
deterministically, not left to the model's own compliance."""

from types import SimpleNamespace

from agents.synthesizer import drafter


def _point(url, title="T", text="some excerpt text", chunk_index=0):
    return SimpleNamespace(payload={"url": url, "title": title, "text": text, "chunk_index": chunk_index})


def test_empty_chunks_returns_no_sources_found_message():
    answer = drafter.draft_answer([], run_id="r1", question="my question", sources_attempted=3, sources_succeeded=0)
    assert "couldn't find" in answer
    assert "my question" in answer
    assert "0/3" in answer


def test_answer_always_cites_sources_used_via_template_fallback(monkeypatch):
    monkeypatch.setattr(
        drafter._synthesizer_agent, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    chunks = [_point("https://a.test"), _point("https://b.test")]

    answer = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=2, sources_succeeded=2)

    assert "https://a.test" in answer
    assert "https://b.test" in answer
    assert "Sources used:" in answer


def test_partial_results_caveat_appears_when_fewer_succeeded_than_attempted(monkeypatch):
    monkeypatch.setattr(drafter._synthesizer_agent, "run", lambda *a, **kw: "a drafted answer")
    chunks = [_point("https://a.test")]

    answer = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=5, sources_succeeded=1)

    assert "Partial results" in answer
    assert "1/5" in answer


def test_no_caveat_when_all_sources_succeeded(monkeypatch):
    monkeypatch.setattr(drafter._synthesizer_agent, "run", lambda *a, **kw: "a drafted answer")
    chunks = [_point("https://a.test")]

    answer = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=1, sources_succeeded=1)

    assert "Partial results" not in answer


def test_duplicate_urls_across_chunks_are_deduplicated_in_sources_list(monkeypatch):
    monkeypatch.setattr(
        drafter._synthesizer_agent, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    chunks = [_point("https://a.test", chunk_index=0), _point("https://a.test", chunk_index=1)]

    answer = drafter.draft_answer(chunks, run_id="r1", question="q", sources_attempted=1, sources_succeeded=1)

    assert answer.count("https://a.test") == 2  # once in the template body, once in "Sources used"
