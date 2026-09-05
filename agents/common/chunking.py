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
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
