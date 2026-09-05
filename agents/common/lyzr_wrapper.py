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

The fallback deliberately calls an OPEN-WEIGHT model via OpenRouter
(https://openrouter.ai) rather than a closed-source API — this project's
own "brain" is swappable, inspectable, and not locked to one vendor,
which matters for the open-source story as much as the code being public.
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


class OpenRouterBackend(LLMBackend):
    """Calls an open-weight model through OpenRouter (one API key, many
    open-source models) when Lyzr is disabled or fails. Kept intentionally
    minimal — this is a safety net, not the primary code path.

    Default model (`settings.openrouter_model`) is DeepSeek V3 — an
    MIT-licensed, open-weight model that benchmarks at or near closed
    frontier models on general reasoning/writing tasks, which is exactly
    the kind of task (summary drafting, transcript classification) this
    wrapper is used for. Swap it via OPENROUTER_MODEL in .env with no code
    change; check https://openrouter.ai/rankings for whatever currently
    tops the open-weight leaderboard before a demo, since new models ship
    often. A ":free" suffix on the model id (e.g.
    "deepseek/deepseek-chat-v3.1:free") uses OpenRouter's free tier.
    """

    def complete(self, system_prompt: str, user_input: str) -> str:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "No LYZR_API_KEY and no OPENROUTER_API_KEY configured — "
                "cannot complete an LLM call. Set one in .env."
            )
        from openai import OpenAI

        client = OpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)
        response = client.chat.completions.create(
            model=settings.openrouter_model,
            max_tokens=1024,
            extra_headers={
                # OpenRouter attribution headers (optional, but they're how
                # OpenRouter's public rankings credit this app).
                "HTTP-Referer": settings.openrouter_app_url,
                "X-Title": settings.openrouter_app_name,
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        return response.choices[0].message.content or ""


class LyzrAgentWrapper:
    """The one object every agent module talks to for LLM calls."""

    def __init__(self, agent_role: str):
        self.agent_role = agent_role
        self._primary: LLMBackend | None = (
            LyzrBackend(settings.lyzr_api_key, agent_role) if settings.lyzr_enabled else None
        )
        self._fallback = OpenRouterBackend()

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
