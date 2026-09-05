"""Orchestrator-owned DAG node handlers (as opposed to the Web-Navigator's,
in agents/web_navigator/handlers.py)."""

from agents.common.models.dag import DAGNode
from agents.orchestrator.executor import RunContext, register_handler


@register_handler("clarify_unsupported")
def handle_clarify_unsupported(node: DAGNode, ctx: RunContext) -> str:
    return "no matching capability for this transcript"
