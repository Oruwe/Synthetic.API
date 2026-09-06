"""Qdrant as the shared, asynchronous coordination layer between agents.

THREE pipelines have written here across this project's iterations, each
its own collection — only the third is live/routed today, the first two
are retired but their code and collections are left in place:
- Web-Navigator (shipping, RETIRED from live routing): extracted orders
  tagged `status=delayed`.
- Web-Researcher (DDG+vision, RETIRED from live routing): screenshot
  analyses tagged `status=candidate`, curated via curate_candidates().
- Search+fetch+chunk (LIVE): chunked page text, upsert_page_chunk() below,
  retrieved via semantic_search_pages() rather than an exact payload filter.

Agents coordinate through this shared memory rather than calling each
other directly — the Synthesizer reads what Web-Navigator wrote, with no
direct call between them.

Embeddings use FastEmbed (local, CPU, no API key) so this works offline
and doesn't burn a limited hackathon LLM credit budget on every row/page/chunk.
"""

import math
import threading
import uuid
from datetime import datetime, timedelta, timezone

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from agents.common.chunking import chunk_text
from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.action import ActionWorkflow
from agents.common.models.orders import DelayedOrder
from agents.common.models.page import FetchedPage
from agents.common.models.research import VisionFinding

logger = get_logger(component="qdrant_store")

_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_VECTOR_SIZE = 384  # bge-small-en-v1.5 output dimension

_client: QdrantClient | None = None
_embedder: TextEmbedding | None = None
_client_lock = threading.Lock()
_embedder_lock = threading.Lock()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        # Double-checked locking: multiple node handlers can call this
        # concurrently (the DAG executor's ThreadPoolExecutor, or several
        # FastAPI requests at once) -- without the lock, two threads racing
        # the first call each build and discard a QdrantClient/connection.
        with _client_lock:
            if _client is None:
                _client = QdrantClient(url=settings.qdrant_url)
    return _client


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = TextEmbedding(model_name=_EMBEDDING_MODEL_NAME)
    return _embedder


def embed_text(text: str) -> list[float]:
    return next(iter(get_embedder().embed([text]))).tolist()


def ensure_collection(collection_name: str, client: QdrantClient | None = None) -> None:
    client = client or get_client()
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qm.VectorParams(size=_VECTOR_SIZE, distance=qm.Distance.COSINE),
            )
        except UnexpectedResponse as exc:
            # TOCTOU: two callers can both see the collection missing (e.g.
            # two concurrent first requests right after a fresh `docker
            # compose up`, since FastAPI runs these sync handlers in a
            # thread pool) and both race to create it -- Qdrant 409s the
            # loser. That's fine, the collection exists either way; only
            # re-raise if creation failed for some other reason.
            if exc.status_code != 409:
                raise


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


# --- Search + fetch + chunk (web_pages collection, the LIVE path) ----------


def upsert_page_chunks(page: FetchedPage, question: str, run_id: str, client: QdrantClient | None = None) -> list[str]:
    """Chunks one fetched page's text and upserts each chunk as its own
    point -- "exact query, chunked page text" per the pivot spec, as opposed
    to one embedding for a whole page or a whole structured record.

    A page with `error` set (fetch failed) or empty text is a no-op: there's
    nothing to embed, and the caller (page_handlers.py) already logs the
    failure -- this function doesn't need to re-raise or re-log it.

    Embeds and upserts the whole page's chunks in ONE call each, not one
    per chunk. Caught live: a real Wikipedia-length article chunks into
    50-100+ pieces, and the previous version did one embed() call plus one
    separate network round-trip to Qdrant PER CHUNK -- fully serial, so 5
    real fetched pages took long enough (worse on virtualized/WSL2 Docker
    networking) to exhaust embed_pages' 60s node timeout on all 3 retries.
    fastembed batches a list far more efficiently than N single-item calls
    in one ONNX inference session, and one qdrant upsert() with all of a
    page's points removes N-1 network round-trips outright.
    """
    if page.error is not None or not page.text.strip():
        return []

    client = client or get_client()
    ensure_collection(settings.qdrant_pages_collection, client)

    chunks = chunk_text(page.text)
    if not chunks:
        return []

    vectors = [vector.tolist() for vector in get_embedder().embed(chunks)]

    point_ids: list[str] = []
    points: list[qm.PointStruct] = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point_key = f"{run_id}:{page.url}:{i}"
        points.append(
            qm.PointStruct(
                id=_stable_uuid(point_key),
                vector=vector,
                payload={
                    "run_id": run_id,
                    "question": question,
                    "url": page.url,
                    "title": page.title,
                    "text": chunk,
                    "chunk_index": i,
                    "fetch_method": page.fetch_method,
                    "timestamp": page.timestamp.isoformat(),
                    "point_key": point_key,
                },
            )
        )
        point_ids.append(point_key)

    client.upsert(collection_name=settings.qdrant_pages_collection, points=points)
    return point_ids


def semantic_search_pages(run_id: str, question: str, top_k: int | None = None) -> list[qm.ScoredPoint]:
    """Top-k semantic retrieval over this run's chunks, scoped to `run_id` --
    replaces the old exact-payload-filter read pattern (see
    scroll_new_delayed/scroll_new_permanent_research above) with a real
    vector query using the question's own embedding.

    Never raises: a Qdrant outage or a run with zero successfully-embedded
    chunks both come back as an empty list, logged, so the Synthesizer can
    still produce a caveated "couldn't retrieve results" answer instead of
    crashing its poll loop.
    """
    top_k = top_k or settings.research_top_k
    try:
        client = get_client()
        ensure_collection(settings.qdrant_pages_collection, client)
        query_vector = embed_text(question)
        response = client.query_points(
            collection_name=settings.qdrant_pages_collection,
            query=query_vector,
            query_filter=qm.Filter(must=[qm.FieldCondition(key="run_id", match=qm.MatchValue(value=run_id))]),
            limit=top_k,
            with_payload=True,
        )
        return response.points
    except Exception as exc:  # noqa: BLE001 - retrieval failing must not crash the Synthesizer
        logger.warning("semantic_search_failed", run_id=run_id, error=str(exc))
        return []


def prune_old_page_chunks(max_age_hours: float | None = None) -> int:
    """Deletes web_pages chunks older than `max_age_hours` (by their stored
    `timestamp` payload field) -- bounds the otherwise-unbounded growth of
    the live collection. Called periodically from the Synthesizer's poll
    loop (see watcher.py), alongside run_store.prune_old_runs(). Never
    raises: a Qdrant outage here is logged and the sweep just does nothing
    this cycle, same fail-open discipline as every other Qdrant call.
    """
    max_age_hours = max_age_hours if max_age_hours is not None else settings.run_retention_hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    try:
        client = get_client()
        ensure_collection(settings.qdrant_pages_collection, client)
        old_filter = qm.Filter(
            must=[qm.FieldCondition(key="timestamp", range=qm.DatetimeRange(lt=cutoff))]
        )
        old_points = _scroll_all_matching(settings.qdrant_pages_collection, old_filter, seen_point_ids=None)
        if not old_points:
            return 0
        client.delete(
            collection_name=settings.qdrant_pages_collection,
            points_selector=qm.PointIdsList(points=[p.id for p in old_points]),
        )
        logger.info("page_chunks_pruned", count=len(old_points), max_age_hours=max_age_hours)
        return len(old_points)
    except Exception as exc:  # noqa: BLE001 - a pruning failure must not crash the poll loop
        logger.warning("page_chunk_prune_failed", error=str(exc))
        return 0


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


# --- Ambient RPA action path (action_workflows collection) -----------------


def upsert_action_workflow(workflow: ActionWorkflow, client: QdrantClient | None = None) -> str:
    """Stores one attempted ActionWorkflow -- successful or not -- embedded
    by its intent text. Persisted regardless of outcome: a failed/refused
    attempt is still useful signal (find_similar_workflow below only
    replays successful ones), and every attempt should be auditable.
    Never raises: a Qdrant outage here must not lose the fact that a real
    browser action was already taken in the physical world, so the caller
    (action_executor.py) logs the workflow either way and this failing is
    a secondary, best-effort concern -- same fail-open discipline as
    every other Qdrant write in this module.
    """
    client = client or get_client()
    point_id = f"{workflow.run_id}:{workflow.intent}"
    try:
        ensure_collection(settings.qdrant_action_workflows_collection, client)
        vector = embed_text(workflow.intent)
        client.upsert(
            collection_name=settings.qdrant_action_workflows_collection,
            points=[
                qm.PointStruct(
                    id=_stable_uuid(point_id),
                    vector=vector,
                    payload={
                        "run_id": workflow.run_id,
                        "intent": workflow.intent,
                        "start_url": workflow.start_url,
                        "steps": [s.model_dump(mode="json") for s in workflow.steps],
                        "success": workflow.success,
                        "refused_reason": workflow.refused_reason,
                        "created_at": workflow.created_at.isoformat(),
                        "point_key": point_id,
                    },
                )
            ],
        )
    except Exception as exc:  # noqa: BLE001 - the real-world action already happened; storage is secondary
        logger.warning("action_workflow_upsert_failed", run_id=workflow.run_id, error=str(exc))
    return point_id


def find_similar_workflow(intent: str, min_score: float | None = None) -> ActionWorkflow | None:
    """Semantic search for a past SUCCESSFUL workflow whose intent is close
    enough to `intent` to trust replaying it outright. Deliberately more
    conservative than semantic_search_pages' read-only top-k retrieval: a
    wrong match here means executing real clicks/keystrokes on a real page
    on the strength of a bad vector match, not just citing a slightly-off
    source. Returns None (never raises) on any failure, an empty result,
    or a best match below `min_score` -- all three mean "explore fresh,"
    handled identically by the caller.
    """
    min_score = min_score if min_score is not None else settings.action_workflow_replay_min_score
    try:
        client = get_client()
        ensure_collection(settings.qdrant_action_workflows_collection, client)
        query_vector = embed_text(intent)
        response = client.query_points(
            collection_name=settings.qdrant_action_workflows_collection,
            query=query_vector,
            query_filter=qm.Filter(must=[qm.FieldCondition(key="success", match=qm.MatchValue(value=True))]),
            limit=1,
            with_payload=True,
        )
        if not response.points or response.points[0].score < min_score:
            return None
        payload = dict(response.points[0].payload or {})
        # "point_key" is a dedup/lookup field we stow on every payload in
        # this module (see upsert_action_workflow above) -- it isn't part
        # of the ActionWorkflow schema, and the model is extra="forbid", so
        # it must be dropped before validating back into the model or this
        # would raise on every real point and silently look identical to
        # "no similar workflow found."
        payload.pop("point_key", None)
        return ActionWorkflow.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - a lookup failure must fall through to fresh exploration, not crash
        logger.warning("find_similar_workflow_failed", intent=intent, error=str(exc))
        return None
