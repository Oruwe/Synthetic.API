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

    # --- Run state ---
    run_store_dir: str = "/data/runs"
    dag_circuit_breaker_threshold: int = 5

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
