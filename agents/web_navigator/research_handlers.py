"""Registers the Web-Researcher's five DAG node handlers into the
orchestrator's HANDLER_REGISTRY: search -> screenshot -> vision-analyze ->
embed -> curate. Imported once (for side effects) by
agents/orchestrator/main.py at startup, alongside web_navigator/handlers.py.
"""

from agents.common import qdrant_store
from agents.common.guard import scan_for_injection
from agents.common.logging import get_logger
from agents.common.models.dag import DAGNode
from agents.common.vision_wrapper import analyze_screenshot
from agents.orchestrator.executor import RunContext, register_handler
from agents.web_navigator import screenshotter, searcher

logger = get_logger(component="research_handlers")

# Same free-text guard-scan discipline as web_navigator/extractor.py, applied
# to the vision model's own output: a real web page is adversarial territory
# (unlike the mock portal, it's not content we control), so anything it
# writes into a screenshot that the VLM then transcribes is treated as
# untrusted, exactly like a scraped DOM field.
#
# key_facts is scanned per-item (it's a list, not a scalar string) below --
# it's VLM-transcribed free text from the same untrusted page and was
# previously left out here entirely, which meant an injection payload that
# only showed up in a "key fact" (title/summary staying clean) went
# unflagged and reached drafter.py's prompt unredacted.
_SCANNED_FIELDS = ("title", "summary")


@register_handler("search_web")
def handle_search_web(node: DAGNode, ctx: RunContext) -> str:
    query = node.params.get("query", ctx.run_id)
    results = searcher.search_web(query)
    ctx.data["search_results"] = results
    ctx.data["research_query"] = query
    return f"found {len(results)} candidate URLs for query {query!r}"


@register_handler("capture_screenshots")
def handle_capture_screenshots(node: DAGNode, ctx: RunContext) -> str:
    results = ctx.data.get("search_results", [])
    captures = screenshotter.capture_screenshots(results, ctx.run_id)
    ctx.data["screenshot_captures"] = captures
    ok = sum(1 for c in captures if c.error is None)
    return f"captured {ok}/{len(captures)} screenshots"


@register_handler("analyze_screenshots")
def handle_analyze_screenshots(node: DAGNode, ctx: RunContext) -> str:
    query = ctx.data.get("research_query", "")
    captures = ctx.data.get("screenshot_captures", [])
    findings = []
    for i, capture in enumerate(captures):
        if capture.error is not None:
            logger.warning("skipping_failed_capture", url=capture.url, error=capture.error)
            continue
        finding = analyze_screenshot(
            capture.url, capture.title, capture.screenshot_path, query, run_id=ctx.run_id, node_id=f"analyze-{i}"
        )
        for field_name in _SCANNED_FIELDS:
            for hit in scan_for_injection(getattr(finding, field_name), field_name):
                finding.flags.append(f"{hit.pattern_name}:{hit.field_name}")
                logger.warning(
                    "guard_hit",
                    url=finding.url,
                    pattern_name=hit.pattern_name,
                    field_name=hit.field_name,
                    matched_text=hit.matched_text,
                )
        for key_fact in finding.key_facts:
            for hit in scan_for_injection(key_fact, "key_facts"):
                finding.flags.append(f"{hit.pattern_name}:{hit.field_name}")
                logger.warning(
                    "guard_hit",
                    url=finding.url,
                    pattern_name=hit.pattern_name,
                    field_name=hit.field_name,
                    matched_text=hit.matched_text,
                )
        findings.append(finding)

    ctx.data["vision_findings"] = findings
    flags_total = sum(len(f.flags) for f in findings)
    return f"analyzed {len(findings)} screenshots, {flags_total} guard flags"


@register_handler("embed_candidates")
def handle_embed_candidates(node: DAGNode, ctx: RunContext) -> str:
    query = ctx.data.get("research_query", "")
    findings = ctx.data.get("vision_findings", [])
    point_ids = [qdrant_store.upsert_candidate(f, ctx.run_id, query) for f in findings]
    return f"stored {len(point_ids)} research candidates in qdrant"


@register_handler("curate_knowledge")
def handle_curate_knowledge(node: DAGNode, ctx: RunContext) -> str:
    query = ctx.data.get("research_query", "")
    result = qdrant_store.curate_candidates(ctx.run_id, query)
    return f"promoted {result['promoted']} findings to permanent, deleted {result['deleted']} as junk"
