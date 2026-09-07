"""Registers the live search+fetch+chunk pipeline's two DAG node handlers
into the orchestrator's HANDLER_REGISTRY: fetch_pages -> embed_pages.
Imported once (for side effects) by agents/orchestrator/main.py at startup,
replacing the old web_navigator/handlers.py + research_handlers.py imports
(those modules are still present and still tested, just no longer wired
into the live app).

fetch_pages also owns the human-in-the-loop gated-content path
(feature/ambient-rpa-action-bridge): when a fetched page turns out to be a
login/subscribe/paywall notice instead of real content (see
page_fetcher._detect_gate_phrase), and nothing else already fetched
answers the question, this pauses the run (AwaitingHumanInputError) and
asks a human for what's needed to get past it. On resume, it uses the
ambient RPA action executor -- the SAME engine and safety rails proven
against demo_target earlier -- to get past the gate and read what was
behind it, in one continuous browser session.
"""

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from agents.common import qdrant_store
from agents.common.logging import get_logger
from agents.common.models.dag import DAGNode
from agents.common.models.page import FetchedPage
from agents.common.models.research import SearchResult
from agents.orchestrator.executor import AwaitingHumanInputError, RunContext, register_handler
from agents.web_navigator import action_executor, page_fetcher

logger = get_logger(component="page_handlers")

# Which of the two supported fields a gate needs, guessed from the SAME
# matched phrase page_fetcher already found -- deliberately simple and
# rule-based, matching this codebase's established style for this kind
# of signal. "sign in"/"log in"/"members" imply an account with a
# password; everything else (subscribe/register/create account/enter
# your email/paywall) is treated as a lower-friction email-capture gate.
# Wrong guesses aren't dangerous, just an extra round trip: a login page
# that only actually needed an email still gets an email field filled in
# by execute_login_and_extract's own field-location logic; the reverse
# (a true login wall guessed as email-only) fails cleanly and reports
# "could not get past" rather than doing anything unsafe.
_LOGIN_STYLE_GATE = re.compile(r"\b(sign\s*in|log\s*in|members?)\b", re.IGNORECASE)


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

    human_inputs = ctx.data.get("human_provided_inputs", {}).get(node.id, {})
    gated_pages = [p for p in pages if p.gated]
    # A gate on ONE candidate source doesn't need a human's help if other
    # sources already answer the question -- only pause when the gate is
    # actually standing between the run and an answer.
    has_other_usable_content = any(p.error is None and not p.gated and p.text for p in pages)

    if gated_pages and not has_other_usable_content:
        gate_page = gated_pages[0]
        if not human_inputs:
            domain = urlparse(gate_page.url).netloc
            fields = ["email", "password"] if _LOGIN_STYLE_GATE.search(gate_page.gate_reason or "") else ["email"]
            prompt = (
                f'{domain} needs {"a login" if len(fields) > 1 else "an email"} to show this content '
                f'(detected: "{gate_page.gate_reason}"). What should I use?'
            )
            logger.info("fetch_pages_pausing_for_gate", url=gate_page.url, fields=fields)
            raise AwaitingHumanInputError(fields=fields, prompt=prompt, url=gate_page.url)

        # Resumed: the human answered -- use the SAME ambient RPA action
        # engine (and its payment guard, step ceiling, screenshot audit
        # trail) already proven against demo_target to get past the gate
        # and read what's now visible, in one continuous browser session.
        index = pages.index(gate_page)
        pages[index] = _pass_gate_and_extract(gate_page, human_inputs, ctx.run_id)

    ctx.data["fetched_pages"] = pages
    ok = sum(1 for p in pages if p.error is None and p.text)
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


def _pass_gate_and_extract(gate_page: FetchedPage, human_inputs: dict, run_id: str) -> FetchedPage:
    """Turns a gated FetchedPage into a real one (or a clearly-failed one)
    using whatever the human supplied. `human_inputs` may contain
    "password" -- never logged, never written back onto anything this
    function returns (see execute_login_and_extract's own docstring for
    where that guarantee actually lives)."""
    email = human_inputs.get("email")
    password = human_inputs.get("password")

    if password:
        workflow = action_executor.execute_login_and_extract(
            email=email,
            password=password,
            start_url=gate_page.url,
            run_id=run_id,
            on_success_extract=action_executor.extract_visible_text,
        )
    else:
        workflow = action_executor.execute_action_loop(
            intent=f"enter the email address {email} into the newsletter/subscription/signup field and submit it",
            start_url=gate_page.url,
            run_id=run_id,
            on_success_extract=action_executor.extract_visible_text,
        )

    if workflow.success and workflow.extracted_text:
        return FetchedPage(
            url=gate_page.url,
            title=gate_page.title,
            text=workflow.extracted_text,
            timestamp=datetime.now(timezone.utc),
            fetch_method="action_gate_bypass",
        )
    return FetchedPage(
        url=gate_page.url,
        title=gate_page.title,
        text="",
        timestamp=datetime.now(timezone.utc),
        fetch_method="action_gate_bypass",
        error=workflow.refused_reason or "could not get past the content gate with the information provided",
    )


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
