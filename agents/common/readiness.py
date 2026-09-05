"""Readiness checks: is this process actually able to do its job right
now, as distinct from /health's liveness check (is the process up at
all). A container can be "alive" (the FastAPI process is running and
answering HTTP) while functionally unable to do anything useful -- Qdrant
unreachable, no search/LLM key configured -- and a plain liveness check
would never catch that. This is the standard production liveness/
readiness split; kept as its own module (not inline in orchestrator/
main.py) so it's testable without spinning up FastAPI.
"""

from dataclasses import dataclass

from agents.common import qdrant_store
from agents.common.config import settings
from agents.common.logging import get_logger

logger = get_logger(component="readiness")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    # Hard-required checks flip overall readiness to False when they fail;
    # soft ones (missing API keys) are reported but the system still
    # accepts traffic -- it degrades gracefully (no search results /
    # template-drafted answers) rather than being genuinely unable to serve.
    hard_required: bool = False


def check_qdrant() -> CheckResult:
    try:
        qdrant_store.get_client().get_collections()
        return CheckResult(name="qdrant", ok=True, hard_required=True)
    except Exception as exc:  # noqa: BLE001 - this IS the check; failure is the expected path to report
        return CheckResult(name="qdrant", ok=False, detail=str(exc), hard_required=True)


def check_tavily_key_configured() -> CheckResult:
    ok = bool(settings.tavily_api_key)
    detail = "" if ok else "TAVILY_API_KEY not set -- searches will return no candidate URLs"
    return CheckResult(name="tavily_api_key", ok=ok, detail=detail)


def check_llm_key_configured() -> CheckResult:
    ok = bool(settings.lyzr_api_key or settings.openrouter_api_key)
    detail = "" if ok else "neither LYZR_API_KEY nor OPENROUTER_API_KEY set -- answers will fall back to a template"
    return CheckResult(name="llm_key", ok=ok, detail=detail)


def run_readiness_checks() -> tuple[bool, list[CheckResult]]:
    checks = [check_qdrant(), check_tavily_key_configured(), check_llm_key_configured()]
    ready = all(c.ok for c in checks if c.hard_required)
    if not ready:
        logger.warning("readiness_check_failed", failed=[c.name for c in checks if c.hard_required and not c.ok])
    return ready, checks
