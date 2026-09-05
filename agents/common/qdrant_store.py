"""Qdrant as the shared, asynchronous coordination layer between agents.

Web-Navigator upserts extracted orders here; Synthesizer polls for new
points tagged `status=delayed` rather than being called directly — that
indirection is the point: agents coordinate through shared memory, not
through a direct function/RPC call between them.

Embeddings use FastEmbed (local, CPU, no API key) so this works offline
and doesn't burn a limited hackathon LLM credit budget on every order row.
"""

from datetime import datetime, timezone

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from agents.common.config import settings
from agents.common.models.orders import DelayedOrder

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


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qm.VectorParams(size=_VECTOR_SIZE, distance=qm.Distance.COSINE),
        )


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
    ensure_collection(client)
    vector = next(iter(get_embedder().embed([_embedding_text(order)]))).tolist()
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


def scroll_new_delayed(seen_point_ids: set[str], limit: int = 100) -> list[qm.Record]:
    """Return delayed-order points not already in `seen_point_ids`.

    A simple scroll-and-diff instead of Qdrant's native change-stream
    (not available in the OSS single-node image) — sufficient for the
    polling cadence the Synthesizer needs, and keeps this dependency-free.
    """
    client = get_client()
    ensure_collection(client)
    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=qm.Filter(must=[qm.FieldCondition(key="status", match=qm.MatchValue(value="delayed"))]),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [r for r in records if r.payload.get("point_key") not in seen_point_ids]


def _stable_uuid(key: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
