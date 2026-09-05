"""Single isolation boundary for the Lyzr Agent SDK.

TODO(verify against hackathon starter kit): the exact Lyzr SDK import,
Agent constructor signature, and call method (`.run` / `.chat` /
`.invoke`?) are NOT confirmed here — the starter kit distributed on the
hackathon platform is the source of truth, and only this file should need
to change once confirmed. Every other module calls `LyzrAgentWrapper.run`
and does not know or care whether that ends up calling the real Lyzr SDK
or the fallback path below.

Why a fallback: with ~3 days and unconfirmed SDK details, the pipeline
must not be dead in the water if `LYZR_ENABLED=false` (no key yet) or if
the real SDK call raises. The fallback goes through Langfuse tracing the
same as the primary path, so observability isn't lost either way.
"""

from abc import ABC, abstractmethod

from agents.common.config import settings
from agents.common.langfuse_tracer import traced_llm_call
from agents.common.logging import get_logger

logger = get_logger(component="lyzr_wrapper")


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_input: str) -> str: ...


class LyzrBackend(LLMBackend):
    """TODO(verify): replace with the real Lyzr Agent SDK call once the
    starter kit's exact API surface is confirmed. Keep the constructor and
    `complete` signature stable so callers never need to change."""

    def __init__(self, api_key: str, agent_role: str):
        self.api_key = api_key
        self.agent_role = agent_role
        # TODO(verify): e.g. `from lyzr import Agent` / `LyzrAgentAPI(...)`
        # and construct the real client here.

    def complete(self, system_prompt: str, user_input: str) -> str:
        raise NotImplementedError(
            "Lyzr SDK call not wired yet — verify method names against the "
            "hackathon starter kit, then implement here."
        )


class DirectLLMFallbackBackend(LLMBackend):
    """Calls an LLM directly (e.g. via the anthropic SDK) when Lyzr is
    disabled or fails. Kept intentionally minimal — this is a safety net,
    not the primary code path."""

    def complete(self, system_prompt: str, user_input: str) -> str:
        if not settings.llm_fallback_api_key:
            raise RuntimeError(
                "No LYZR_API_KEY and no LLM_FALLBACK_API_KEY configured — "
                "cannot complete an LLM call. Set one in .env."
            )
        import anthropic

        client = anthropic.Anthropic(api_key=settings.llm_fallback_api_key)
        response = client.messages.create(
            model=settings.llm_fallback_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class LyzrAgentWrapper:
    """The one object every agent module talks to for LLM calls."""

    def __init__(self, agent_role: str):
        self.agent_role = agent_role
        self._primary: LLMBackend | None = (
            LyzrBackend(settings.lyzr_api_key, agent_role) if settings.lyzr_enabled else None
        )
        self._fallback = DirectLLMFallbackBackend()

    @traced_llm_call(name="lyzr_agent_call")
    def run(self, system_prompt: str, user_input: str, *, run_id: str, node_id: str) -> str:
        if self._primary is not None:
            try:
                return self._primary.complete(system_prompt, user_input)
            except Exception as exc:  # noqa: BLE001 - deliberately broad: never let an
                # unconfirmed SDK failure take the whole pipeline down.
                logger.warning(
                    "lyzr_primary_call_failed_falling_back",
                    run_id=run_id,
                    node_id=node_id,
                    agent_role=self.agent_role,
                    error=str(exc),
                )
        return self._fallback.complete(system_prompt, user_input)
