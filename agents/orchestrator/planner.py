"""Transcript -> DAGPlan.

The live path: any non-empty transcript is classified as either an ACTION
intent (a physical task -- "book", "buy", "sign me up for", ...) or, by
default, a research QUESTION. A question flows into the existing
fetch_pages -> embed_pages chain unchanged. An action intent flows into
the newer, separate ambient-RPA execute_action node (see
agents/web_navigator/action_handlers.py) -- Qdrant either supplies a past
successful workflow to replay, or a live vision-guided browser loop
explores fresh. Both shapes call Tavily HERE, before the DAG is built (per
the pivot spec — search is plan-construction logic, not something
Web-Navigator does at runtime anymore): for a question it produces
candidate pages to fetch; for a fresh action it produces a starting URL to
act on (a replay skips this, since it already has the prior workflow's
start_url).

Web-Navigator still owns fetching/extracting/acting (its responsibility is
unchanged) — it just now fetches or acts on whatever URL(s) the planner
found instead of navigating one hardcoded portal.

Both DAG shapes flow through the existing, untouched DAG executor
(retries/timeout/circuit-breaker), so nothing here needs its own retry
logic beyond what the handlers already do -- though note the action node
is built with max_retries=1 deliberately (see _build_action_plan): a
node that clicks/types on a real page is not safe to retry like an HTTP
fetch is.

Older keyword-routed planning (shipping portal / DuckDuckGo+vision search)
lived in this file before this pivot and is fully retired from live
routing — see agents/web_navigator/handlers.py, research_handlers.py, and
their still-present, still-tested underlying modules for what's now dormant
rather than deleted.
"""

import re
import uuid
from datetime import datetime, timezone

from agents.common import search_wrapper
from agents.common.config import settings
from agents.common.models.dag import DAGEdge, DAGNode, DAGPlan, NodeType

# Deliberately simple and rule-based, matching this codebase's established
# "rule-based base, no LLM in the hot path of planning" approach (the
# original shipping-portal/DDG planners worked the same way). False
# negatives just fall through to a research answer (safe); the vision
# model's own refusal instruction plus action_executor.py's payment guard
# are the actual safety backstops, not this classifier.
_ACTION_INTENT_PATTERN = re.compile(
    r"\b(book|buy|purchase|order|sign\s*me\s*up|sign\s*up|register|reserve|schedule|"
    r"apply\s*for|subscribe|fill\s*out|submit|cart|check\s*out|renew)\b",
    re.IGNORECASE,
)
# A leading interrogative means this is a question ABOUT something, even if
# an action verb shows up later in it -- e.g. "What is the best watch to
# buy under $200?" (a real query from this project's own history) must not
# be misrouted into clicking "buy" on a page. Checked before the verb scan
# below, not merged into one pattern, so it always wins regardless of
# where in the transcript an action verb appears.
_QUESTION_STARTER_PATTERN = re.compile(
    r"^\s*(what|who|when|where|why|how|which|is|are|was|were|do|does|did|can|could|would|should|will)\b",
    re.IGNORECASE,
)
# Imperative commands lead with the verb ("book a table", "sign me up for
# the newsletter") -- an action verb only found several words in is much
# more likely to be an object of a question/statement ("...to buy a
# house") than a command. Six words gives a little room for a short
# leading phrase ("Hey, can you book...") without letting a verb buried
# deep in a longer sentence trigger a false positive.
_ACTION_VERB_LEAD_WORDS = 6


def _looks_like_action_intent(question: str) -> bool:
    if _QUESTION_STARTER_PATTERN.match(question):
        return False
    lead = " ".join(question.split()[:_ACTION_VERB_LEAD_WORDS])
    return bool(_ACTION_INTENT_PATTERN.search(lead))


class PlannerInputError(ValueError):
    pass


def build_plan(transcript: str, run_id: str | None = None) -> DAGPlan:
    if not transcript or not transcript.strip():
        raise PlannerInputError("transcript is empty")

    run_id = run_id or str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    question = transcript.strip()

    if _looks_like_action_intent(question):
        return _build_action_plan(transcript, question, run_id, created_at)

    # search_wrapper.search() never raises (see its own docstring) -- a
    # search-API outage still produces a plan; fetch_pages/embed_pages then
    # produce a "no sources found" answer rather than blocking planning.
    search_results = search_wrapper.search(question)

    nodes = [
        DAGNode(
            id="fetch",
            type=NodeType.FETCH_PAGES,
            name="Fetch and extract candidate pages",
            handler_key="fetch_pages",
            params={
                "question": question,
                "search_results": [r.model_dump(mode="json") for r in search_results],
            },
            timeout_seconds=100,
            max_retries=1,  # per-URL fast/fallback retry already happens inside the handler
        ),
        DAGNode(
            id="embed",
            type=NodeType.EMBED_PAGES,
            name="Chunk and embed fetched pages",
            handler_key="embed_pages",
            depends_on=["fetch"],
            # 60s was too tight for real content: caught live, embedding 5
            # real fetched pages (one a full Wikipedia article -> 50-100+
            # chunks) exhausted all 3 retries at exactly 60s each before
            # qdrant_store.upsert_page_chunks() was batched (one embed()
            # and one qdrant upsert() per page instead of per chunk, see
            # its docstring). 180s is a safety margin on top of that fix,
            # not a substitute for it -- a slow/unbatched path would still
            # time out eventually on a big enough page.
            timeout_seconds=180,
        ),
    ]
    # Derived from each node's depends_on rather than hand-listed a second
    # time -- the executor's own graph building (_build_graph) only reads
    # depends_on, so a hand-maintained `edges` list here was a second,
    # disconnected source of truth that could silently drift out of sync
    # with the actual dependency a future edit adds to a node.
    edges = [DAGEdge(from_node=dep, to_node=node.id) for node in nodes for dep in node.depends_on]

    return DAGPlan(
        run_id=run_id,
        transcript=transcript,
        created_at=created_at,
        nodes=nodes,
        edges=edges,
        status="planned",
        circuit_breaker_threshold=settings.dag_circuit_breaker_threshold,
    )


def _build_action_plan(transcript: str, question: str, run_id: str, created_at: datetime) -> DAGPlan:
    """A fresh action needs somewhere to start acting -- reuse the same
    Tavily search the research path uses, taking its top result as the
    starting page. (A replay -- decided later, inside
    action_handlers.py, once Qdrant is actually queried -- ignores this
    and uses the prior workflow's own start_url instead; the search here
    is cheap and always safe to do up front regardless of which path
    ends up running.)

    No candidate URL -> nothing to act on -> routed as `no_capability`
    with a clarifying message, same as an unsupported request, rather
    than handing the action executor a start_url of None.
    """
    search_results = search_wrapper.search(question)
    if not search_results:
        return DAGPlan(
            run_id=run_id,
            transcript=transcript,
            created_at=created_at,
            nodes=[],
            edges=[],
            status="no_capability",
            circuit_breaker_threshold=settings.dag_circuit_breaker_threshold,
        )

    start_url = search_results[0].url
    nodes = [
        DAGNode(
            id="act",
            type=NodeType.EXECUTE_ACTION,
            name="Execute (or replay) the requested action",
            handler_key="execute_action",
            params={"intent": question, "start_url": start_url},
            timeout_seconds=180,  # up to action_max_steps screenshots+vision calls+page settle waits
            # MUST stay 1: unlike an HTTP fetch, a click/type on a real
            # page is not idempotent -- retrying could re-submit an
            # action the first attempt already performed for real.
            max_retries=1,
        ),
    ]
    return DAGPlan(
        run_id=run_id,
        transcript=transcript,
        created_at=created_at,
        nodes=nodes,
        edges=[],
        status="planned",
        circuit_breaker_threshold=settings.dag_circuit_breaker_threshold,
    )
