"""Delivers the drafted summary. Kept intentionally simple given the
timeline: logs + stdout always, an optional webhook POST if configured
(e.g. to a Slack incoming webhook or a local stub receiver for the demo)."""

import httpx

from agents.common.config import settings
from agents.common.logging import get_logger

logger = get_logger(component="notifier")


def notify(summary: str, run_id: str) -> None:
    logger.info("notification_ready", run_id=run_id, summary_preview=summary[:200])
    print(f"\n=== Synthesizer summary (run {run_id}) ===\n{summary}\n{'=' * 40}\n", flush=True)

    if settings.notifier_webhook_url:
        try:
            httpx.post(
                settings.notifier_webhook_url,
                json={"run_id": run_id, "summary": summary},
                timeout=5.0,
            )
        except Exception as exc:  # noqa: BLE001 - notification delivery must not crash the watcher loop
            logger.warning("notifier_webhook_failed", error=str(exc), run_id=run_id)
