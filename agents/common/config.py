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

    # --- Run state ---
    run_store_dir: str = "/data/runs"
    # 2, not a bigger number: the standard shipping-portal plan only has 3
    # nodes, so a higher default (5) could never actually trip on it -- the
    # breaker would pass its own unit tests yet be structurally inert on
    # every real plan the planner produces. Actually wired into DAGPlan by
    # orchestrator/planner.py (previously this setting was unused/dead).
    dag_circuit_breaker_threshold: int = 2

    # --- Lyzr (verify exact env var names against the hackathon starter kit) ---
    lyzr_api_key: str = ""
    lyzr_enabled: bool = False

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

    # --- Web-Researcher: keyless search, screenshots, open-weight vision model ---
    # Qwen2.5-VL is Apache-2.0 licensed and, as of writing, one of the
    # strongest open-weight vision-language models on general
    # screenshot/document understanding -- check
    # https://openrouter.ai/rankings before a demo, a better one may exist
    # by then. No code change needed to swap it, same as OPENROUTER_MODEL.
    openrouter_vision_model: str = "qwen/qwen2.5-vl-72b-instruct"
    search_engine_url: str = "https://html.duckduckgo.com/html/"
    research_max_results: int = 5
    # Cosine similarity cutoff for the curate_knowledge step: candidates
    # scoring >= this against the original query are kept permanently,
    # everything else ("majority junk") is deleted from Qdrant.
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
