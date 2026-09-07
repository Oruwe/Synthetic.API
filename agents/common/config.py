"""Central settings for all three agents, loaded from environment variables.

Every agent imports `settings` from here instead of reading `os.environ`
directly, so there is exactly one place that knows the env var names.

Validated eagerly at import time (`settings = Settings()` below runs at
module load, not on first use) -- pydantic's own type coercion already
did this implicitly; the validators below add real BUSINESS-RULE checks
on top (a cross-field consistency pydantic's type system alone can't
express), so a misconfiguration fails loudly and immediately on startup
instead of silently degrading or surfacing as a confusing failure deep
inside a run. Caught live, this exact class of bug: LYZR_ENABLED=true set
without a real LYZR_AGENT_ID, which the pipeline itself would have
silently absorbed (lyzr_wrapper.py's own fail-open design falls back to
OpenRouter on any Lyzr error) -- worth knowing about at startup, not
inferred later from "why does every trace show the fallback model."
"""

import warnings

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Portal / Qdrant ---
    portal_base_url: str = "http://mock_portal:5000"
    portal_username: str = "admin"
    portal_password: str = "admin123"
    qdrant_url: str = "http://qdrant:6333"
    # Blank for the self-hosted docker-compose qdrant service (no auth) --
    # required for Qdrant Cloud's free tier (see deploy/huggingface/), which
    # rejects any request with no key at all. get_client() (qdrant_store.py)
    # passes `api_key=qdrant_api_key or None` so this stays a no-op locally.
    qdrant_api_key: str = ""
    qdrant_collection: str = "delayed_orders"
    qdrant_research_collection: str = "web_knowledge"
    # The new live path's collection (chunked page text, not structured
    # order records or vision-model findings). Separate name per the pivot
    # spec, rather than mixing schemas into an existing collection.
    qdrant_pages_collection: str = "web_pages"
    # The ambient-RPA action path's collection: one point per attempted
    # ActionWorkflow (successful or not), embedded by intent text, so a
    # semantically-similar future intent can find and replay a prior
    # successful workflow instead of re-exploring from scratch.
    qdrant_action_workflows_collection: str = "action_workflows"

    # --- Ambient RPA action path (screenshot -> vision decides -> Playwright
    # acts, looped) ---
    # A hard ceiling, not a target: most simple tasks should finish well
    # under this. Exists so a confused model (or a page that never reaches
    # a recognizable "done" state) can't loop forever -- bounded the same
    # way every other external call in this codebase is (see
    # page_fetch_timeout_seconds, DAG_CIRCUIT_BREAKER_THRESHOLD, etc.).
    action_max_steps: int = 8
    # A prior workflow must score at least this well (cosine similarity)
    # against the new intent before it's trusted enough to replay blindly
    # rather than treated as just a hint. Conservative on purpose: a wrong
    # replay executes real actions on a real page, unlike a wrong semantic
    # search result in the read-only research path, which just costs one
    # bad citation.
    action_workflow_replay_min_score: float = 0.85
    # Two more independent trust gates on TOP of the similarity score
    # above -- see qdrant_store.find_workflow_memory. A single lucky
    # success is not enough to replay blind (min_success_count), and a
    # workflow that has started failing more often than it succeeds
    # (e.g. the target page got redesigned) must stop being offered even
    # if it once scored perfectly (min_trust_ratio).
    action_workflow_min_success_count: int = 1
    action_workflow_min_trust_ratio: float = 0.6
    # How long an UNTRUSTED workflow memory (never succeeded, or below
    # min_trust_ratio) can sit unused before prune_stale_workflows sweeps
    # it -- bounds the collection's growth. Deliberately much longer than
    # RUN_RETENTION_HOURS: this is durable cross-run memory, not one run's
    # transient state, and a workflow that DOES stay trustworthy is never
    # deleted by this regardless of age.
    action_workflow_retention_hours: float = 24.0 * 14
    # Cross-process lock backend for record_workflow_outcome's
    # read-merge-write (see qdrant_store.py's _distributed_lock_for_workflow).
    # Empty by default -- local dev/tests run entirely on the in-process
    # lock (agents.common.qdrant_store._lock_for_workflow), which is
    # already sufficient for a single orchestrator process/replica (the
    # deployment this system runs as today). Set this only when the
    # orchestrator is scaled to more than one process/replica, at which
    # point the in-process lock alone can no longer prevent two replicas
    # from losing an update to the same canonical workflow record.
    redis_url: str = ""
    # Generous vs. the actual critical section (one Qdrant retrieve, one
    # local embed, one Qdrant upsert) -- long enough that a slow Qdrant
    # round-trip doesn't expire the lock out from under its own holder,
    # short enough that a holder that crashed mid-critical-section doesn't
    # wedge the record for long.
    redis_lock_ttl_ms: int = 10_000
    # How long a caller will wait to acquire a contended lock before
    # giving up and proceeding WITHOUT it (logged loudly when this
    # happens) -- matches this codebase's fail-open discipline: a real
    # browser action already happened in the physical world, so refusing
    # to ever record it because a lock is contended is a worse outcome
    # than a rare, logged, unprotected write.
    redis_lock_acquire_timeout_seconds: float = 5.0

    # --- Run state ---
    run_store_dir: str = "/data/runs"
    # How long a completed run's JSON file (and its Qdrant chunks) are kept
    # before prune_old_runs()/prune_old_page_chunks() delete them -- bounds
    # otherwise-unbounded growth of both data/runs/ and the web_pages
    # collection. Swept periodically (see synthesizer_prune_every_n_polls
    # below), not on every poll.
    run_retention_hours: float = 24.0
    synthesizer_prune_every_n_polls: int = 720  # ~1 hour at the default 5s poll interval
    # 2, not a bigger number: the standard shipping-portal plan only has 3
    # nodes, so a higher default (5) could never actually trip on it -- the
    # breaker would pass its own unit tests yet be structurally inert on
    # every real plan the planner produces. Actually wired into DAGPlan by
    # orchestrator/planner.py (previously this setting was unused/dead).
    dag_circuit_breaker_threshold: int = 2

    # --- Lyzr ---
    lyzr_api_key: str = ""
    lyzr_enabled: bool = False
    # Lyzr's model is a pre-created agent (persona/instructions set ONCE in
    # Lyzr Studio, or via its Create Agent API), not a system prompt sent
    # per call -- see lyzr_wrapper.py's LyzrBackend docstring. Create one
    # agent in Lyzr Studio whose instructions match
    # agents/synthesizer/drafter.py's _PAGE_SYSTEM_PROMPT (the live
    # drafting call is the only caller this actually needs to work end to
    # end today) and paste its agent_id here.
    lyzr_agent_id: str = ""
    # Lyzr's chat API requires a user_id; this project has no per-human-user
    # concept (triggered by a transcript, not a login), so a fixed
    # identifier is fine -- override only if your Lyzr account needs a
    # specific format (e.g. an email).
    lyzr_user_id: str = "synthetic-api"

    # --- LLM fallback used by lyzr_wrapper when Lyzr is disabled/unreachable ---
    # Routed through OpenRouter to an open-weight model (not a closed API) --
    # see agents/common/lyzr_wrapper.py for why. DeepSeek V3 is the default;
    # override OPENROUTER_MODEL to point at whatever currently tops
    # https://openrouter.ai/rankings for open-weight models.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "deepseek/deepseek-chat-v3.1"
    openrouter_app_url: str = "https://github.com/Oruwe/Synthetic.API"
    openrouter_app_name: str = "Synthetic.API"

    # --- Search + fetch + chunk + semantic retrieval (the LIVE research path) ---
    tavily_api_key: str = ""
    research_max_results: int = 5
    # Fast-path HTTP GET timeout AND the Playwright fallback's timeout --
    # kept in one setting since both paths apply it per-URL the same way.
    page_fetch_timeout_seconds: float = 9.0
    research_top_k: int = 5

    # --- Retired from live routing, kept for the dormant DDG+vision pipeline
    # (agents/web_navigator/searcher.py, screenshotter.py,
    # research_handlers.py, common/vision_wrapper.py -- still present,
    # still tested, just not imported by orchestrator/main.py anymore) ---
    openrouter_vision_model: str = "qwen/qwen2.5-vl-72b-instruct"
    search_engine_url: str = "https://html.duckduckgo.com/html/"
    research_relevance_threshold: float = 0.35
    screenshot_dir: str = "/data/screenshots"

    # --- Omi (verify exact webhook contract against the hackathon starter kit) ---
    omi_webhook_secret: str = ""

    # --- Orchestrator API auth (see agents/orchestrator/auth.py) ---
    # Same fail-open-if-unset posture as omi_webhook_secret: unset (the
    # default, e.g. local `docker compose up`) leaves /trigger and
    # /runs/* open, matching this project's existing local-dev
    # convenience. MUST be set to a real value for any deployment
    # reachable from the open internet -- there is otherwise no
    # authentication at all on an API that can trigger arbitrary browser
    # actions and read/answer any run's paused state.
    orchestrator_api_key: str = ""

    # --- Langfuse ---
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_enabled: bool = True

    # --- Synthesizer polling ---
    synthesizer_poll_interval_seconds: float = 5.0
    notifier_webhook_url: str = ""

    @model_validator(mode="after")
    def _warn_if_lyzr_enabled_but_unusable(self) -> "Settings":
        """Doesn't raise -- consistent with this codebase's fail-open
        discipline everywhere else (lyzr_wrapper.py itself already falls
        back to OpenRouter on exactly this condition, per-call, silently).
        A loud startup warning is strictly more than that silent per-call
        fallback ever gave an operator, without making config.py the one
        place in this project that crashes instead of degrading."""
        if self.lyzr_enabled and not self.lyzr_agent_id:
            warnings.warn(
                "LYZR_ENABLED=true but LYZR_AGENT_ID is not set -- every "
                "call will fail Lyzr and fall back to OpenRouter (see "
                "lyzr_wrapper.py). Set LYZR_AGENT_ID, or LYZR_ENABLED=false "
                "to use the fallback intentionally instead of by accident.",
                stacklevel=2,
            )
        if self.lyzr_enabled and not self.lyzr_api_key:
            warnings.warn(
                "LYZR_ENABLED=true but LYZR_API_KEY is not set -- every "
                "call will fail Lyzr and fall back to OpenRouter.",
                stacklevel=2,
            )
        return self

    @field_validator(
        "portal_base_url", "qdrant_url", "openrouter_base_url", "langfuse_host",
    )
    @classmethod
    def _must_be_a_url_if_set(cls, value: str, info) -> str:
        """Type coercion alone (bare `str`) accepts any string, including
        an obvious typo like a bare hostname with no scheme -- pydantic
        doesn't have a built-in "URL, but allow blank" type, so this
        checks the one thing worth catching (a missing scheme) without
        pulling in a full URL-parsing dependency for it."""
        if value and not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError(f"{info.field_name} must start with http:// or https:// (got {value!r})")
        return value

    @field_validator("action_max_steps", "dag_circuit_breaker_threshold", "research_top_k", "research_max_results")
    @classmethod
    def _must_be_positive(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive (got {value})")
        return value

    @field_validator("action_workflow_replay_min_score", "action_workflow_min_trust_ratio")
    @classmethod
    def _must_be_a_valid_score(cls, value: float, info) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{info.field_name} must be between 0.0 and 1.0 (got {value})")
        return value


settings = Settings()
