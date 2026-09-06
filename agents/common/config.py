"""Central settings for all three agents, loaded from environment variables.

Every agent imports `settings` from here instead of reading `os.environ`
directly, so there is exactly one place that knows the env var names.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Portal / Qdrant ---
    portal_base_url: str = "http://mock_portal:5000"
    portal_username: str = "admin"
    portal_password: str = "admin123"
    qdrant_url: str = "http://qdrant:6333"
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

    # --- Langfuse ---
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_enabled: bool = True

    # --- Synthesizer polling ---
    synthesizer_poll_interval_seconds: float = 5.0
    notifier_webhook_url: str = ""


settings = Settings()
