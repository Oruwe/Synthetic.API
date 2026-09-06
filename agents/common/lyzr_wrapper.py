"""Single isolation boundary for the Lyzr Agent SDK.

Verified against the public `lyzr-python-sdk` package (PyPI) and its
GitHub README (github.com/LyzrCore/lyzr-python) -- import, client
construction, agent creation, and the chat call's request shape are all
confirmed there. ONE thing is NOT verifiable from outside a real Lyzr
account: the exact shape of `client.inference.chat(...)`'s return value
(the docs show the request but only `print(chat_response)` for the
response, no field-by-field schema). See `_extract_chat_text()` below for
how that's handled defensively rather than guessed at.

The bigger thing worth understanding, not just verifying: Lyzr's model is
architecturally different from a raw chat-completion API. There is no
"pass a system prompt, get an answer" call -- an agent's persona/
instructions are configured ONCE, either in the Lyzr Studio UI or via its
Create Agent API, and you get back an `agent_id` you then send messages
to. This project's own `complete(system_prompt, user_input)` contract
predates that discovery (it was built to match OpenAI-compatible APIs,
which OpenRouter's fallback below actually is) -- so `LyzrBackend` below
receives a `system_prompt` on every call but cannot act on it; the
prompt actually driving the real Lyzr agent's behavior lives in Lyzr
Studio, on the agent `LYZR_AGENT_ID` points at. Keep that agent's
instructions in sync with `agents/synthesizer/drafter.py`'s
`_PAGE_SYSTEM_PROMPT` (the only live caller) by hand when one changes.

Why a fallback regardless: `LYZR_ENABLED=false` (no key yet) or the real
Lyzr call raising for any reason (rate limit, network, an account with no
agent configured) must never take the whole pipeline down. The fallback
goes through Langfuse tracing the same as the primary path, so
observability isn't lost either way.

The fallback deliberately calls an OPEN-WEIGHT model via OpenRouter
(https://openrouter.ai) rather than a closed-source API — this project's
own "brain" is swappable, inspectable, and not locked to one vendor,
which matters for the open-source story as much as the code being public.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agents.common.config import settings
from agents.common.langfuse_tracer import traced_llm_call
from agents.common.logging import get_logger

logger = get_logger(component="lyzr_wrapper")


@dataclass
class LLMResult:
    """What a backend actually produced, not just the text -- so the
    Langfuse trace (see langfuse_tracer.py) can report the real model name
    and token counts instead of showing a generation with 0 tokens / $0.00
    for every call, which is what happened before this existed."""

    text: str
    model: str | None = None
    usage: dict | None = None  # {"input": int, "output": int, "total": int}


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_input: str, *, run_id: str | None = None) -> LLMResult: ...


def _extract_chat_text(response) -> str:
    """Pulls the answer text out of client.inference.chat()'s return value.

    Not fully verifiable from outside a real Lyzr account (see module
    docstring) -- the public docs show the request shape but never the
    response schema, just `print(chat_response)`. The one concrete lead
    found (a community writeup) used dict-style `response["response"]`,
    consistent with the rest of this SDK's plain-dict style (the request
    itself is a plain dict, not a typed object). Tried defensively, in
    order, rather than assumed: dict key "response", then a few other
    plausible dict keys, then an attribute of the same names, then --
    rather than silently returning "" and looking like an empty answer --
    str(response) as a last resort, with a warning logged so a real call's
    actual shape shows up immediately in the logs instead of hiding a bug.
    """
    if isinstance(response, dict):
        for key in ("response", "message", "content", "text"):
            if key in response and response[key]:
                return str(response[key])
    else:
        for attr in ("response", "message", "content", "text"):
            value = getattr(response, attr, None)
            if value:
                return str(value)
    logger.warning(
        "lyzr_chat_response_shape_unrecognized",
        response_type=type(response).__name__,
        detail="none of response/message/content/text was found -- falling back to str(response); "
        "check this against a real Lyzr account and adjust _extract_chat_text if it's wrong",
    )
    return str(response)


class LyzrBackend(LLMBackend):
    """Real Lyzr Agent SDK call via the public `lyzr-python-sdk` package.

    See the module docstring for why `system_prompt` is accepted but not
    used here: Lyzr's persona lives on the pre-created agent
    (`agent_id`/`LYZR_AGENT_ID`), configured in Lyzr Studio, not passed
    per call. `run_id` is threaded through as Lyzr's `session_id` so each
    research run gets its own conversation thread on Lyzr's side too,
    rather than every call sharing one anonymous session.
    """

    def __init__(self, api_key: str, agent_role: str, agent_id: str):
        self.api_key = api_key
        self.agent_role = agent_role
        self.agent_id = agent_id

    def complete(self, system_prompt: str, user_input: str, *, run_id: str | None = None) -> LLMResult:
        if not self.agent_id:
            raise RuntimeError(
                "LYZR_ENABLED=true but LYZR_AGENT_ID is not set -- create an agent in Lyzr "
                "Studio (instructions matching drafter.py's _PAGE_SYSTEM_PROMPT) and set its "
                "agent_id in .env."
            )
        from lyzr_python_sdk import LyzrAgentAPI

        client = LyzrAgentAPI(api_key=self.api_key)
        response = client.inference.chat(
            {
                "user_id": settings.lyzr_user_id,
                "agent_id": self.agent_id,
                "message": user_input,
                "session_id": run_id or "synthetic-api-default-session",
            }
        )
        text = _extract_chat_text(response)

        # Opportunistic: use real token usage if this account/response
        # shape happens to include it, same {"input","output","total"}
        # shape OpenRouterBackend below produces -- otherwise Langfuse
        # just shows no token count for this call rather than a fabricated
        # one (see langfuse_tracer.py's _model_usage(), which already
        # treats a missing/empty usage dict as "nothing to report").
        usage = None
        raw_usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        if isinstance(raw_usage, dict) and any(k in raw_usage for k in ("input", "output", "total")):
            usage = raw_usage

        return LLMResult(text=text, model=f"lyzr:{self.agent_id}", usage=usage)


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

    def complete(self, system_prompt: str, user_input: str, *, run_id: str | None = None) -> LLMResult:
        # run_id is unused here -- OpenRouter's chat-completions call is
        # stateless per request (no server-side session concept to attach
        # it to), unlike LyzrBackend.complete() above.
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
        text = response.choices[0].message.content or ""
        usage = None
        if response.usage is not None:
            usage = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }
        return LLMResult(text=text, model=settings.openrouter_model, usage=usage)


class LyzrAgentWrapper:
    """The one object every agent module talks to for LLM calls."""

    def __init__(self, agent_role: str):
        self.agent_role = agent_role
        self._primary: LLMBackend | None = (
            LyzrBackend(settings.lyzr_api_key, agent_role, settings.lyzr_agent_id) if settings.lyzr_enabled else None
        )
        self._fallback = OpenRouterBackend()
        # Stashed by run() on every call so the @traced_llm_call decorator
        # (langfuse_tracer.py) -- which only sees this method's string
        # return value -- can still report the real model name and token
        # counts on the trace, instead of a generation showing 0 tokens /
        # $0.00 for every call regardless of what actually happened.
        self.last_model: str | None = None
        self.last_usage: dict | None = None

    @traced_llm_call(name="lyzr_agent_call")
    def run(self, system_prompt: str, user_input: str, *, run_id: str, node_id: str) -> str:
        if self._primary is not None:
            try:
                result = self._primary.complete(system_prompt, user_input, run_id=run_id)
                self.last_model, self.last_usage = result.model, result.usage
                return result.text
            except Exception as exc:  # noqa: BLE001 - deliberately broad: never let a
                # real Lyzr account/network hiccup take the whole pipeline down.
                logger.warning(
                    "lyzr_primary_call_failed_falling_back",
                    run_id=run_id,
                    node_id=node_id,
                    agent_role=self.agent_role,
                    error=str(exc),
                )
        result = self._fallback.complete(system_prompt, user_input, run_id=run_id)
        self.last_model, self.last_usage = result.model, result.usage
        return result.text
