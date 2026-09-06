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
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from agents.common.chunking import chunk_text
from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.action import ActionWorkflow, WorkflowMemory
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

# Per-canonical-key locks guarding record_workflow_outcome's read-merge-
# write below -- see that function's docstring for why a plain upsert
# isn't safe here. One lock per key (not one global lock like
# run_store._index_lock) because record_workflow_outcome's actual hazard
# is two attempts racing on the SAME (domain, intent) pair; two DIFFERENT
# tasks completing at the same moment touch different Qdrant points and
# have no reason to serialize behind each other.
#
# This alone only protects one process (node handlers racing inside one
# orchestrator's ThreadPoolExecutor, see executor.py) -- sufficient for
# how this system runs today (one process, one replica), but NOT for a
# scaled-out orchestrator with more than one process/replica, where two
# instances could both pass this same-process lock and still race against
# each other on the same Qdrant point. That cross-process case is what
# _distributed_lock_for_workflow (below) closes via Redis when
# settings.redis_url is configured; this in-process lock remains its
# fallback (and the only thing used at all in local dev/tests, which run
# with REDIS_URL unset) -- see that function's docstring.
_workflow_memory_locks: dict[str, threading.Lock] = {}
_workflow_memory_locks_meta_lock = threading.Lock()


def _lock_for_workflow(point_id: str) -> threading.Lock:
    lock = _workflow_memory_locks.get(point_id)
    if lock is None:
        with _workflow_memory_locks_meta_lock:
            lock = _workflow_memory_locks.setdefault(point_id, threading.Lock())
    return lock


_redis_client = None  # type: ignore[var-annotated]  # "redis.Redis | None" -- untyped to keep redis import lazy
_redis_client_lock = threading.Lock()

# Released via a Lua script rather than a plain DEL so a caller can never
# release a lock it doesn't actually hold anymore -- if this holder's TTL
# already expired and a different caller acquired the key in the
# meantime, blindly DELing would release THEIR lock out from under them.
# The script is atomic (Redis runs it single-threaded), so the
# check-then-delete itself can't race.
_REDIS_RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def _get_redis_client():
    """None when settings.redis_url is unset (the default -- local dev
    and every existing test run entirely on the in-process lock above,
    no Redis required) or when constructing the client fails. Double-
    checked locking, same pattern as get_client()/get_embedder()."""
    global _redis_client
    if not settings.redis_url:
        return None
    if _redis_client is None:
        with _redis_client_lock:
            if _redis_client is None:
                import redis as redis_lib

                _redis_client = redis_lib.Redis.from_url(
                    settings.redis_url, socket_timeout=2, socket_connect_timeout=2
                )
    return _redis_client


@contextmanager
def _distributed_lock_for_workflow(point_id: str):
    """The cross-process half of record_workflow_outcome's concurrency
    story. When settings.redis_url is configured, this is a real
    distributed lock (SET NX PX, released only by a matching token via
    the Lua script above) -- correct for a scaled-out orchestrator with
    more than one process/replica, which the in-process lock alone cannot
    protect (two different processes each pass their own in-process lock
    and still race on the same Qdrant point).

    Every failure mode here degrades rather than blocks the write,
    matching this codebase's fail-open discipline everywhere else
    (Tavily, Playwright, every other Qdrant call): Redis not configured,
    Redis unreachable, and lock-acquisition timing out (another holder
    has it) all fall through to either the in-process lock (same-process
    safety only) or, as a last resort, no lock at all -- logged loudly,
    since a real browser action already happened in the physical world
    and refusing to ever record it because a lock is contended would be a
    worse outcome than a rare, observable unprotected write.
    """
    client = _get_redis_client()
    if client is None:
        with _lock_for_workflow(point_id):
            yield
        return

    lock_key = f"workflow_lock:{point_id}"
    token = uuid.uuid4().hex
    try:
        acquired = _acquire_redis_lock(client, lock_key, token)
    except Exception as exc:  # noqa: BLE001 - Redis being unreachable must degrade, not block the write
        logger.warning("redis_lock_unreachable_falling_back_to_in_process_lock", point_id=point_id, error=str(exc))
        with _lock_for_workflow(point_id):
            yield
        return

    if not acquired:
        logger.warning("redis_lock_acquire_timed_out_proceeding_without_a_lock", point_id=point_id)
        yield
        return

    try:
        yield
    finally:
        try:
            client.eval(_REDIS_RELEASE_LOCK_SCRIPT, 1, lock_key, token)
        except Exception as exc:  # noqa: BLE001 - a failed release just means the TTL expires it later
            logger.warning("redis_lock_release_failed", point_id=point_id, error=str(exc))


def _acquire_redis_lock(client, lock_key: str, token: str) -> bool:
    """Spin-waits (short sleep, not a busy loop) for up to
    settings.redis_lock_acquire_timeout_seconds -- contention here is
    expected to be rare and short-lived (one Qdrant retrieve + one local
    embed + one Qdrant upsert), so a simple poll is the right amount of
    machinery, not a pub/sub wakeup scheme."""
    deadline = time.monotonic() + settings.redis_lock_acquire_timeout_seconds
    while True:
        if client.set(lock_key, token, nx=True, px=settings.redis_lock_ttl_ms):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


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
#
# v1 of this collection (superseded) stored one point per RUN -- every
# attempt of "the same" task created a new near-duplicate point, with no
# way for repeated success to build trust or repeated failure to erode
# it. This is the "engineer the memory" rebuild: ONE point per
# (domain, canonicalized intent) pair, continuously updated in place, so
# the memory actually gets more (or less) trustworthy over time instead
# of just accumulating a log. See WorkflowMemory's own docstring
# (agents/common/models/action.py) for the full design rationale.


def _normalize_intent(intent: str) -> str:
    """Collapses phrasing differences that shouldn't create a distinct
    memory record ("Book a table!" vs "book a table") -- NOT a substitute
    for semantic matching (find_workflow_memory below still does a real
    vector search), just what makes the canonical key for the exact-ish
    same request stable."""
    text = re.sub(r"[^\w\s]", "", intent.strip().lower())
    return re.sub(r"\s+", " ", text).strip()


def _domain_of(url: str) -> str:
    """Never raises: an unparseable start_url just means an empty domain,
    which still lets a fresh record be created (matched only by intent
    similarity going forward) rather than blocking the write."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.removeprefix("www.") if netloc else netloc
    except Exception:  # noqa: BLE001
        return ""


def _canonical_key(domain: str, intent: str) -> str:
    return f"{domain}:{_normalize_intent(intent)}"


def _load_workflow_memory(client: QdrantClient, point_id: str) -> WorkflowMemory | None:
    records = client.retrieve(
        collection_name=settings.qdrant_action_workflows_collection, ids=[point_id], with_payload=True
    )
    if not records:
        return None
    return WorkflowMemory.model_validate(records[0].payload or {})


def record_workflow_outcome(workflow: ActionWorkflow, client: QdrantClient | None = None) -> WorkflowMemory:
    """The write side of ambient RPA's memory: folds one execution attempt
    into the durable WorkflowMemory record for its (domain, intent) pair,
    reinforcing an existing record in place rather than creating a new
    near-duplicate point per run. A successful attempt replaces the
    replay target (`steps`/`start_url`) with what it just verified works;
    a failed attempt updates the trust counters but NEVER overwrites a
    known-good step sequence with an unverified one -- so one bad attempt
    against an otherwise-reliable workflow erodes its trust score without
    destroying the thing that made it trustworthy in the first place.

    Concurrency: two attempts against the SAME (domain, intent) pair
    completing around the same time both do read-merge-write against one
    Qdrant point -- without serializing that critical section, the
    classic lost-update race applies (both read success_count=N, both
    write back N+1, one increment vanishes). Qdrant has no compare-and-set
    primitive to lean on instead, so this is closed with a lock around the
    critical section: `_distributed_lock_for_workflow` (see its docstring)
    uses Redis when settings.redis_url is configured (correct across
    multiple orchestrator processes/replicas), falling back to the
    in-process `_lock_for_workflow` otherwise -- which is also all that
    local dev and every existing test run against, so this needs no Redis
    to develop against or to verify offline.

    Never raises: a Qdrant outage here must not lose the fact that a real
    browser action already happened in the physical world, so the caller
    (action_handlers.py) has the workflow either way and persisting the
    memory of it is a secondary, best-effort concern -- same fail-open
    discipline as every other Qdrant write in this module. On failure,
    returns a same-shaped, in-memory-only WorkflowMemory (not persisted)
    so the caller never has to special-case a storage failure just to log
    the outcome.
    """
    client = client or get_client()
    now = datetime.now(timezone.utc)
    domain = _domain_of(workflow.start_url)
    key = _canonical_key(domain, workflow.intent)
    point_id = _stable_uuid(key)

    def _fresh_memory() -> WorkflowMemory:
        return WorkflowMemory(
            canonical_key=key,
            domain=domain,
            representative_intent=workflow.intent,
            start_url=workflow.start_url,
            steps=workflow.steps if workflow.success else [],
            success_count=1 if workflow.success else 0,
            failure_count=0 if workflow.success else 1,
            created_at=now,
            last_used_at=now,
            last_success_at=now if workflow.success else None,
        )

    try:
        ensure_collection(settings.qdrant_action_workflows_collection, client)
        with _distributed_lock_for_workflow(point_id):
            existing = _load_workflow_memory(client, point_id)

            if existing is None:
                memory = _fresh_memory()
            else:
                memory = existing.model_copy(
                    update={
                        "last_used_at": now,
                        "success_count": existing.success_count + (1 if workflow.success else 0),
                        "failure_count": existing.failure_count + (0 if workflow.success else 1),
                    }
                )
                if workflow.success:
                    memory.steps = workflow.steps
                    memory.start_url = workflow.start_url
                    memory.last_success_at = now

            # The upsert (the "write" half of read-merge-write) MUST stay
            # inside the lock too -- releasing it after the merge and
            # before the write would reopen exactly the race this lock
            # exists to close.
            vector = embed_text(memory.representative_intent)
            client.upsert(
                collection_name=settings.qdrant_action_workflows_collection,
                points=[qm.PointStruct(id=point_id, vector=vector, payload=memory.model_dump(mode="json"))],
            )
        logger.info(
            "workflow_memory_recorded",
            canonical_key=key,
            success_count=memory.success_count,
            failure_count=memory.failure_count,
            trust_ratio=round(memory.trust_ratio, 3),
        )
        return memory
    except Exception as exc:  # noqa: BLE001 - the real-world action already happened; storage is secondary
        logger.warning("record_workflow_outcome_failed", intent=workflow.intent, error=str(exc))
        return _fresh_memory()


def find_workflow_memory(
    intent: str, start_url: str | None = None, min_score: float | None = None
) -> WorkflowMemory | None:
    """Semantic search for a WorkflowMemory trustworthy enough to replay
    outright. Gated on THREE independent conditions, ALL of which must
    hold -- similarity alone is not enough, since a wrong replay here
    means executing real clicks/keystrokes on a real page, not just
    citing a slightly-off source:
      1. cosine similarity >= min_score (as before).
      2. at least `settings.action_workflow_min_success_count` verified
         successes -- one lucky run is not enough trust to replay blind.
      3. a trust ratio (success/(success+failure)) >=
         `settings.action_workflow_min_trust_ratio` -- a workflow that
         broke after a page redesign stops being offered once enough
         recent attempts against it have failed, with no human ever
         needing to manually invalidate it.
    When `start_url` is known (a fresh action always has a candidate
    start_url from the planner's own search), the query is also filtered
    to the SAME domain -- a semantically similar phrase for a different
    site must never trigger a replay against the wrong page.

    Returns None (never raises) on any failure or a non-qualifying best
    match -- both mean "explore fresh," handled identically by the caller.
    """
    min_score = min_score if min_score is not None else settings.action_workflow_replay_min_score
    try:
        client = get_client()
        ensure_collection(settings.qdrant_action_workflows_collection, client)
        query_vector = embed_text(intent)
        query_filter = None
        if start_url:
            domain = _domain_of(start_url)
            if domain:
                query_filter = qm.Filter(must=[qm.FieldCondition(key="domain", match=qm.MatchValue(value=domain))])
        response = client.query_points(
            collection_name=settings.qdrant_action_workflows_collection,
            query=query_vector,
            query_filter=query_filter,
            limit=1,
            with_payload=True,
        )
        if not response.points or response.points[0].score < min_score:
            return None
        memory = WorkflowMemory.model_validate(response.points[0].payload or {})
        if memory.success_count < settings.action_workflow_min_success_count:
            return None
        if memory.trust_ratio < settings.action_workflow_min_trust_ratio:
            return None
        return memory
    except Exception as exc:  # noqa: BLE001 - a lookup failure must fall through to fresh exploration, not crash
        logger.warning("find_workflow_memory_failed", intent=intent, error=str(exc))
        return None


def prune_stale_workflows(max_age_hours: float | None = None, client: QdrantClient | None = None) -> int:
    """Deletes workflow memories that are both OLD (unused for
    `max_age_hours`) AND untrustworthy (never succeeded, or a trust ratio
    below `settings.action_workflow_min_trust_ratio`) -- bounds the
    otherwise-unbounded growth of this collection without ever deleting a
    record that's actively working, no matter how old it is. A workflow
    with real trust built up is exactly the thing this memory exists to
    keep; only the dead weight gets swept. Never raises: a Qdrant outage
    here is logged and the sweep just does nothing this cycle, same
    fail-open discipline as every other Qdrant call in this module.
    """
    max_age_hours = max_age_hours if max_age_hours is not None else settings.action_workflow_retention_hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    try:
        client = client or get_client()
        ensure_collection(settings.qdrant_action_workflows_collection, client)
        old_filter = qm.Filter(must=[qm.FieldCondition(key="last_used_at", range=qm.DatetimeRange(lt=cutoff))])
        old_points = _scroll_all_matching(settings.qdrant_action_workflows_collection, old_filter, seen_point_ids=None)
        stale_ids = []
        for point in old_points:
            memory = WorkflowMemory.model_validate(point.payload or {})
            if memory.trust_ratio < settings.action_workflow_min_trust_ratio:
                stale_ids.append(point.id)
        if not stale_ids:
            return 0
        client.delete(
            collection_name=settings.qdrant_action_workflows_collection,
            points_selector=qm.PointIdsList(points=stale_ids),
        )
        logger.info("stale_workflows_pruned", count=len(stale_ids), max_age_hours=max_age_hours)
        return len(stale_ids)
    except Exception as exc:  # noqa: BLE001 - a pruning failure must not crash the caller
        logger.warning("workflow_prune_failed", error=str(exc))
        return 0
