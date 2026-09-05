"""Delivers a drafted message to the user. Kept intentionally simple given
the timeline: logs + stdout always, an optional webhook POST if configured
(e.g. to a Slack incoming webhook or a local stub receiver for the demo).

Lives in common/ (not synthesizer/) because it's used by two callers: the
Synthesizer (async, after a Qdrant poll) and the Orchestrator (synchronous,
for a `no_capability` plan -- see orchestrator/executor.py -- where no
Qdrant point is ever written, so the Synthesizer's watcher would never see
it and the user would otherwise get silence instead of an answer).
"""

import httpx

from agents.common.config import settings
from agents.common.logging import get_logger

logger = get_logger(component="notifier")


def notify(summary: str, run_id: str) -> None:
    logger.info("notification_ready", run_id=run_id, summary_preview=summary[:200])
    print(f"\n=== Synthetic.API summary (run {run_id}) ===\n{summary}\n{'=' * 40}\n", flush=True)

    if settings.notifier_webhook_url:
        try:
            httpx.post(
                settings.notifier_webhook_url,
                json={"run_id": run_id, "summary": summary},
                timeout=5.0,
            )
        except Exception as exc:  # noqa: BLE001 - notification delivery must not crash the caller
            logger.warning("notifier_webhook_failed", error=str(exc), run_id=run_id)
