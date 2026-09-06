"""DAG node handler for the ambient RPA action path.

Given an intent and a starting URL (both baked into the node's params by
orchestrator/planner.py's action-intent branch), this handler:
1. Checks Qdrant for a trustworthy WorkflowMemory for this (domain,
   intent) pair (qdrant_store.find_workflow_memory) -- gated on
   similarity AND accumulated trust, not similarity alone (see that
   function's docstring) -- and if one qualifies, its recorded steps are
   replayed deterministically instead of re-exploring from scratch.
2. Otherwise runs the live observe/decide/act loop
   (action_executor.execute_action_loop).

Either way, the resulting ActionWorkflow is folded back into the shared
WorkflowMemory (qdrant_store.record_workflow_outcome) -- a success
reinforces trust and refreshes the replay target, a failure erodes trust
without destroying a previously-verified good sequence -- and stashed on
the RunContext so executor.execute_plan can compose the final answer
directly once the DAG finishes. Action runs report synchronously, unlike
the live research path's async Synthesizer draft, since there's no LLM
drafting step needed for a deterministic step-by-step outcome.

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

    memory = qdrant_store.find_workflow_memory(intent, start_url=start_url)
    if memory is not None:
        logger.info(
            "action_replaying_trusted_workflow",
            intent=intent,
            canonical_key=memory.canonical_key,
            trust_ratio=round(memory.trust_ratio, 3),
        )
        workflow = action_executor.replay_workflow(memory, run_id=ctx.run_id)
    else:
        logger.info("action_exploring_fresh_workflow", intent=intent, start_url=start_url)
        workflow = action_executor.execute_action_loop(intent, start_url, run_id=ctx.run_id)

    qdrant_store.record_workflow_outcome(workflow)
    ctx.data["action_workflow"] = workflow
    return workflow.success
