"""Pure text chunker -- no I/O, deliberately simple and offline-testable.

Fixed-size character windows with overlap rather than sentence/paragraph-
aware splitting: good enough for embedding-based semantic retrieval over
a handful of fetched pages, and simple enough to reason about and test
exhaustively rather than depending on a heavier NLP splitter.
"""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if not text or not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            # Reached the end of the text with this window -- stop here.
            # Stepping again would only ever re-slice a tail that's already
            # wholly contained in the chunk just appended (e.g. a 1450-char
            # text with the defaults produced a 3rd "chunk" that was a pure
            # substring of the 2nd), wasting an embed call/Qdrant point on a
            # duplicate with no new content and diluting semantic search.
            break
        start += step
    return chunks
