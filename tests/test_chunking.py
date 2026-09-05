import pytest

from agents.common.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_single_chunk():
    assert chunk_text("short text", chunk_size=800) == ["short text"]


def test_long_text_is_split_into_multiple_chunks():
    text = "x" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_chunks_overlap():
    text = "".join(f"{i:04d}" for i in range(500))  # deterministic, indexable content
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    # the tail of chunk N should reappear at the head of chunk N+1
    assert chunks[0][-100:] == chunks[1][:100]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("x" * 2000, chunk_size=100, overlap=100)


def test_no_content_is_lost_across_chunks():
    text = "x" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert "".join(chunks).replace("x", "") == ""  # sanity: still all 'x'
    assert sum(len(c) for c in chunks) >= len(text)  # overlap means >=, never less
