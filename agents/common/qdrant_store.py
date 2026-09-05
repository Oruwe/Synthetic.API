"""Qdrant as the shared, asynchronous coordination layer between agents.

Two independent pipelines write here, each its own collection:
- Web-Navigator (shipping): upserts extracted orders tagged `status=delayed`.
- Web-Researcher: upserts screenshot analyses tagged `status=candidate`,
  then curate_candidates() promotes relevant ones to `status=permanent`
  and hard-deletes the rest ("majority junk") based on relevance to the
  original query.

The Synthesizer polls for new `delayed`/`permanent` points rather than
being called directly — that indirection is the point: agents coordinate
through shared memory, not a direct function/RPC call between them.

Embeddings use FastEmbed (local, CPU, no API key) so this works offline
and doesn't burn a limited hackathon LLM credit budget on every row/page.
"""

import math
import uuid
from datetime import datetime, timezone

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.orders import DelayedOrder
from agents.common.models.research import VisionFinding

logger = get_logger(component="qdrant_store")

_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_VECTOR_SIZE = 384  # bge-small-en-v1.5 output dimension

_client: QdrantClient | None = None
_embedder: TextEmbedding | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=_EMBEDDING_MODEL_NAME)
    return _embedder


def embed_text(text: str) -> list[float]:
    return next(iter(get_embedder().embed([text]))).tolist()


def ensure_collection(collection_name: str, client: QdrantClient | None = None) -> None:
    client = client or get_client()
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qm.VectorParams(size=_VECTOR_SIZE, distance=qm.Distance.COSINE),
        )


def _stable_uuid(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# --- Shipping orders (delayed_orders collection) ----------------------------


def _embedding_text(order: DelayedOrder) -> str:
    # Built ONLY from structured fields we already parsed and (for free-text
    # fields) guard-scanned — never the raw scraped HTML/DOM text.
    parts = [
        f"Order {order.order_id} for {order.customer_name}",
        f"destination {order.destination}",
        f"status {order.status}",
    ]
    if order.carrier:
        parts.append(f"carrier {order.carrier}")
    if order.delay_reason:
        parts.append(f"delay reason: {order.delay_reason}")
    return "; ".join(parts)


def upsert_order(order: DelayedOrder, run_id: str, client: QdrantClient | None = None) -> str:
    client = client or get_client()
    ensure_collection(settings.qdrant_collection, client)
    vector = embed_text(_embedding_text(order))
    point_id = f"{run_id}:{order.order_id}"
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            qm.PointStruct(
                id=_stable_uuid(point_id),
                vector=vector,
                payload={
                    "run_id": run_id,
                    "order_id": order.order_id,
                    "status": order.status,
                    "customer_name": order.customer_name,
                    "destination": order.destination,
                    "carrier": order.carrier,
                    "delay_reason": order.delay_reason,
                    "flags": order.flags,
                    "extracted_at": order.extracted_at.isoformat(),
                    "point_key": point_id,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    return point_id


def scroll_new_delayed(seen_point_ids: set[str]) -> list[qm.Record]:
    """Return delayed-order points not already in `seen_point_ids`."""
    scroll_filter = qm.Filter(must=[qm.FieldCondition(key="status", match=qm.MatchValue(value="delayed"))])
    return _scroll_all_matching(settings.qdrant_collection, scroll_filter, seen_point_ids=seen_point_ids)


# --- Web-Researcher (web_knowledge collection) ------------------------------


def _finding_embedding_text(finding: VisionFinding) -> str:
    parts = [finding.title, finding.summary, *finding.key_facts]
    return "; ".join(p for p in parts if p)


def upsert_candidate(finding: VisionFinding, run_id: str, query: str, client: QdrantClient | None = None) -> str:
    """Stores a vision-model finding as a `status=candidate` point -- every
    screenshot analysis lands here first ("entire data is stored in
    Qdrant"), and curate_candidates() below decides what survives."""
    client = client or get_client()
    ensure_collection(settings.qdrant_research_collection, client)
    vector = embed_text(_finding_embedding_text(finding))
    point_id = f"{run_id}:{finding.url}"
    client.upsert(
        collection_name=settings.qdrant_research_collection,
        points=[
            qm.PointStruct(
                id=_stable_uuid(point_id),
                vector=vector,
                payload={
                    "run_id": run_id,
                    "query": query,
                    "url": finding.url,
                    "title": finding.title,
                    "summary": finding.summary,
                    "key_facts": finding.key_facts,
                    "screenshot_path": finding.screenshot_path,
                    "flags": finding.flags,
                    "status": "candidate",
                    "point_key": point_id,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    return point_id


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def should_retain(similarity: float, threshold: float | None = None) -> bool:
    """Pure decision function (unit-testable in isolation from Qdrant I/O):
    is this candidate relevant enough to the query to keep permanently?"""
    threshold = threshold if threshold is not None else settings.research_relevance_threshold
    return similarity >= threshold


def curate_candidates(run_id: str, query: str, threshold: float | None = None) -> dict:
    """The lifecycle step: fetch every `candidate` point for this run,
    score each against the query embedding, promote relevant ones to
    `status=permanent` (kept forever), and hard-delete the rest -- "the
    relevant data is stored permanently and the majority junk is deleted."
    """
    client = get_client()
    ensure_collection(settings.qdrant_research_collection, client)
    query_vector = embed_text(query)

    scroll_filter = qm.Filter(
        must=[
            qm.FieldCondition(key="run_id", match=qm.MatchValue(value=run_id)),
            qm.FieldCondition(key="status", match=qm.MatchValue(value="candidate")),
        ]
    )
    candidates = _scroll_all_matching(
        settings.qdrant_research_collection, scroll_filter, seen_point_ids=None, with_vectors=True
    )

    promote_ids: list[str] = []
    delete_ids: list[str] = []
    for record in candidates:
        similarity = cosine_similarity(query_vector, record.vector)
        if should_retain(similarity, threshold):
            promote_ids.append(record.id)
        else:
            delete_ids.append(record.id)

    if promote_ids:
        client.set_payload(
            collection_name=settings.qdrant_research_collection,
            payload={"status": "permanent"},
            points=promote_ids,
        )
    if delete_ids:
        client.delete(
            collection_name=settings.qdrant_research_collection,
            points_selector=qm.PointIdsList(points=delete_ids),
        )

    logger.info("candidates_curated", run_id=run_id, promoted=len(promote_ids), deleted=len(delete_ids))
    return {"promoted": len(promote_ids), "deleted": len(delete_ids)}


def scroll_new_permanent_research(seen_point_ids: set[str]) -> list[qm.Record]:
    """Return `status=permanent` research findings not already seen --
    what the Synthesizer's watcher polls to notice a curated research run."""
    scroll_filter = qm.Filter(must=[qm.FieldCondition(key="status", match=qm.MatchValue(value="permanent"))])
    return _scroll_all_matching(settings.qdrant_research_collection, scroll_filter, seen_point_ids=seen_point_ids)


# --- Shared pagination helper ------------------------------------------------

_SCROLL_PAGE_SIZE = 100
_SCROLL_MAX_PAGES = 50  # safety cap: 50 * 100 = 5,000 points per call, not unbounded


def _scroll_all_matching(
    collection_name: str,
    scroll_filter: qm.Filter,
    seen_point_ids: set[str] | None,
    with_vectors: bool = False,
) -> list[qm.Record]:
    """Walks every page of a scroll query via Qdrant's `next_page_offset`
    cursor rather than reading a single page: a single unpaginated page
    (an earlier version of this module) silently dropped any matching
    point that happened to land past the first `limit` results once a
    collection held more than that many matches.

    `seen_point_ids=None` returns every matching record (used by
    curate_candidates, which needs the full candidate set for a run);
    a set filters out already-seen points (used by the two watch-for-new
    functions above).
    """
    client = get_client()
    ensure_collection(collection_name, client)

    matched: list[qm.Record] = []
    offset = None
    for _ in range(_SCROLL_MAX_PAGES):
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        if seen_point_ids is None:
            matched.extend(records)
        else:
            matched.extend(r for r in records if r.payload.get("point_key") not in seen_point_ids)
        if next_offset is None:
            break
        offset = next_offset
    return matched
