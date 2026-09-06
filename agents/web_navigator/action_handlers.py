"""DAG node handler for the ambient RPA action path.

Given an intent and a starting URL (both baked into the node's params by
orchestrator/planner.py's action-intent branch), this handler:
1. Checks Qdrant for a semantically similar past SUCCESSFUL workflow
   (qdrant_store.find_similar_workflow) -- if one scores above
   settings.action_workflow_replay_min_score, its recorded steps are
   replayed deterministically instead of re-exploring from scratch.
2. Otherwise runs the live observe/decide/act loop
   (action_executor.execute_action_loop).

Either way, the resulting ActionWorkflow is persisted to Qdrant (so this
attempt itself becomes future replay/audit material) and stashed on the
RunContext so executor.execute_plan can compose the final answer directly
once the DAG finishes -- action runs report synchronously, unlike the live
research path's async Synthesizer draft, since there's no LLM drafting
step needed for a deterministic step-by-step outcome.

Safety: this node's max_retries MUST stay at 1 (enforced in planner.py,
where the node is built) -- retrying a node that clicks/types on a real
page is not idempotent like retrying an HTTP fetch, and could re-submit
an action the first attempt already performed.
"""

from agents.common import qdrant_store
from agents.common.logging import get_logger
from agents.common.models.dag import DAGNode
from agents.orchestrator.executor import RunContext, register_handler
from agents.web_navigator import action_executor

logger = get_logger(component="action_handlers")


@register_handler("execute_action")
def handle_execute_action(node: DAGNode, ctx: RunContext) -> bool:
    intent = node.params["intent"]
    start_url = node.params["start_url"]

    prior = qdrant_store.find_similar_workflow(intent)
    if prior is not None:
        logger.info("action_replaying_similar_workflow", intent=intent, prior_run_id=prior.run_id)
        workflow = action_executor.replay_workflow(prior, run_id=ctx.run_id)
    else:
        logger.info("action_exploring_fresh_workflow", intent=intent, start_url=start_url)
        workflow = action_executor.execute_action_loop(intent, start_url, run_id=ctx.run_id)

    qdrant_store.upsert_action_workflow(workflow)
    ctx.data["action_workflow"] = workflow
    return workflow.success
