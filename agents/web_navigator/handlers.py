"""Registers the Web-Navigator's three DAG node handlers into the
orchestrator's HANDLER_REGISTRY. Imported once (for side effects) by
agents/orchestrator/main.py at startup.
"""

from agents.common.config import settings
from agents.common.models.dag import DAGNode
from agents.orchestrator.executor import RunContext, register_handler
from agents.web_navigator import embedder, extractor, portal_client


@register_handler("scrape_portal")
def handle_scrape_portal(node: DAGNode, ctx: RunContext) -> str:
    rows = portal_client.scrape_dashboard_rows()
    ctx.data["scraped_rows"] = rows
    return f"scraped {len(rows)} rows"


@register_handler("extract_validate")
def handle_extract_validate(node: DAGNode, ctx: RunContext) -> str:
    raw_rows = ctx.data.get("scraped_rows", [])
    result = extractor.extract_orders(raw_rows, page_url=f"{settings.portal_base_url}/dashboard")
    ctx.data["extraction_result"] = result
    return f"extracted {result.extracted_count} delayed orders, {result.guard_flags_total} guard flags"


@register_handler("embed_store")
def handle_embed_store(node: DAGNode, ctx: RunContext) -> str:
    result = ctx.data["extraction_result"]
    point_ids = embedder.embed_and_store(result.orders, ctx.run_id)
    return f"stored {len(point_ids)} points in qdrant"
