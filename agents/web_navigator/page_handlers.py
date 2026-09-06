"""Registers the live search+fetch+chunk pipeline's two DAG node handlers
into the orchestrator's HANDLER_REGISTRY: fetch_pages -> embed_pages.
Imported once (for side effects) by agents/orchestrator/main.py at startup,
replacing the old web_navigator/handlers.py + research_handlers.py imports
(those modules are still present and still tested, just no longer wired
into the live app).
"""

from agents.common import qdrant_store
from agents.common.logging import get_logger
from agents.common.models.dag import DAGNode
from agents.common.models.research import SearchResult
from agents.orchestrator.executor import RunContext, register_handler
from agents.web_navigator import page_fetcher

logger = get_logger(component="page_handlers")


@register_handler("fetch_pages")
def handle_fetch_pages(node: DAGNode, ctx: RunContext) -> str:
    question = node.params.get("question", "")
    search_results = [SearchResult.model_validate(r) for r in node.params.get("search_results", [])]

    ctx.data["question"] = question
    if not search_results:
        # search_wrapper.search() already logged why (no API key, API
        # error, or genuinely zero results) -- nothing to fetch, but this
        # is not a node failure: embed_pages/the drafter turn an empty
        # page list into a "no sources found" answer, not a crash.
        ctx.data["fetched_pages"] = []
        return "no search results to fetch (see search_wrapper logs)"

    pages = page_fetcher.fetch_pages(search_results)
    ctx.data["fetched_pages"] = pages
    ok = sum(1 for p in pages if p.error is None)
    # `node` is the same object living in run.plan.nodes (not a copy), so
    # this mutation survives run_store.save_run() and is readable later by
    # synthesizer/main.py -- the actual fetch-success count, not the
    # distinct-URLs-among-the-top-k-retrieved-chunks approximation it used
    # before this existed, which undercounted whenever fetch succeeded on
    # more URLs than settings.research_top_k (default 5) chunks could
    # represent, showing a false "(Partial results: ...)" caveat on a
    # fully successful fetch.
    node.params["sources_succeeded"] = ok
    return f"fetched {ok}/{len(pages)} pages for question {question!r}"


@register_handler("embed_pages")
def handle_embed_pages(node: DAGNode, ctx: RunContext) -> str:
    question = ctx.data.get("question", "")
    pages = ctx.data.get("fetched_pages", [])

    total_chunks = 0
    pages_embedded = 0
    for page in pages:
        try:
            point_ids = qdrant_store.upsert_page_chunks(page, question, ctx.run_id)
        except Exception as exc:  # noqa: BLE001 - one page's embedding failing must not fail the batch
            logger.warning("page_embedding_failed", url=page.url, error=str(exc))
            continue
        if point_ids:
            pages_embedded += 1
        total_chunks += len(point_ids)

    return f"embedded {total_chunks} chunks from {pages_embedded}/{len(pages)} pages"
