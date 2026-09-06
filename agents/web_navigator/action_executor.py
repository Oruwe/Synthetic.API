"""Ambient RPA action loop: screenshot the current page state, ask the
vision model for the single next physical action toward the intent,
execute it via Playwright, repeat -- until the model says "done", refuses
(a payment/checkout guard applies independently too, see below), gets
stuck, or a hard step limit is hit. Every attempt (successful or not) is
recorded as an ActionWorkflow (agents/common/qdrant_store.py persists it).

Real-world safety, non-negotiable, not left to the model's own compliance:
- `settings.action_max_steps` bounds the loop -- a confused model, or a
  page that never reaches a recognizable "done" state, cannot loop
  forever. Same discipline as every other external call in this codebase
  (page_fetch_timeout_seconds, DAG_CIRCUIT_BREAKER_THRESHOLD, etc.).
- `_looks_like_payment_action()` is a code-side backstop against
  submitting a payment, independent of the vision model's own (bypassable)
  system-prompt instruction not to -- checked against the model's own
  stated reasoning/typed text BEFORE an action is ever executed, not just
  relied on as a polite request to the model.
- Every step keeps its screenshot, so the full sequence of what this
  system actually did to a real page is auditable after the fact -- never
  just described after the fact with nothing to check it against.
"""

import re
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.action import ActionStep, ActionWorkflow
from agents.common.playwright_utils import PAGE_DEFAULT_TIMEOUT_MS, launched_browser
from agents.common.vision_wrapper import decide_next_action

logger = get_logger(component="action_executor")

# Fixed so normalized 0-1000 model coordinates map to a consistent,
# predictable real pixel position every call -- the model is never told
# the actual pixel size, only that it's normalized to this convention.
_VIEWPORT = {"width": 1280, "height": 800}

_PAYMENT_KEYWORDS = re.compile(
    r"\b(pay\s*now|place\s*order|complete\s*purchase|checkout|submit\s*payment|confirm\s*payment|"
    r"credit\s*card|card\s*number|cvv|billing\s*address|buy\s*now)\b",
    re.IGNORECASE,
)


def _looks_like_payment_action(step: ActionStep) -> bool:
    haystack = " ".join(filter(None, [step.reasoning, step.text]))
    return bool(_PAYMENT_KEYWORDS.search(haystack))


def execute_action_loop(intent: str, start_url: str, run_id: str, max_steps: int | None = None) -> ActionWorkflow:
    """Runs the observe/decide/act loop against a real page. Never raises:
    a browser-level failure (bad URL, crashed page, launch failure) is
    caught and recorded as a "stuck" step so the caller always gets back
    a valid ActionWorkflow to persist and report on, same fail-open
    discipline as every other node handler in this codebase."""
    max_steps = max_steps if max_steps is not None else settings.action_max_steps
    out_dir = Path(settings.screenshot_dir) / run_id / "action"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps: list[ActionStep] = []
    success = False
    refused_reason: str | None = None

    try:
        with launched_browser(PAGE_DEFAULT_TIMEOUT_MS) as browser:
            page = browser.new_page(viewport=_VIEWPORT)
            page.set_default_timeout(PAGE_DEFAULT_TIMEOUT_MS)
            page.goto(start_url, wait_until="load")

            for i in range(max_steps):
                screenshot_path = str(out_dir / f"step-{i:02d}.png")
                page.screenshot(path=screenshot_path, full_page=False)

                step = decide_next_action(screenshot_path, intent, steps, run_id=run_id, node_id=f"action-{i}")

                if step.kind not in ("done", "refused", "stuck") and _looks_like_payment_action(step):
                    logger.warning("action_refused_payment_guard", run_id=run_id, reasoning=step.reasoning)
                    step = ActionStep(
                        kind="refused",
                        reasoning="blocked by payment/checkout safety guard: " + step.reasoning,
                        screenshot_path=screenshot_path,
                    )

                steps.append(step)
                logger.info("action_step_decided", run_id=run_id, step=i, kind=step.kind, reasoning=step.reasoning)

                if step.kind == "done":
                    success = True
                    break
                if step.kind == "refused":
                    refused_reason = step.reasoning
                    break
                if step.kind == "stuck":
                    break

                _execute_step(page, step)
                time.sleep(0.3)  # let the page settle before the next screenshot
            else:
                logger.warning("action_loop_exhausted_max_steps", run_id=run_id, max_steps=max_steps)

    except Exception as exc:  # noqa: BLE001 - a browser-level failure must not crash the DAG node
        logger.warning("action_loop_failed", run_id=run_id, error=str(exc))
        steps.append(ActionStep(kind="stuck", reasoning=f"execution failed: {exc}"))

    return ActionWorkflow(
        run_id=run_id,
        intent=intent,
        start_url=start_url,
        steps=steps,
        success=success,
        refused_reason=refused_reason,
        created_at=datetime.now(timezone.utc),
    )


def _execute_step(page, step: ActionStep) -> None:
    """Maps a step's normalized 0-1000 coordinates to real viewport pixels
    and performs it. Only reached for click/type/scroll -- the loop above
    breaks on done/refused/stuck before ever calling this."""
    real_x = real_y = None
    if step.x is not None and step.y is not None:
        real_x = (step.x / 1000) * _VIEWPORT["width"]
        real_y = (step.y / 1000) * _VIEWPORT["height"]

    if step.kind == "click":
        if real_x is None:
            raise ValueError("click action missing coordinates")
        page.mouse.click(real_x, real_y)
    elif step.kind == "type":
        if real_x is not None:
            page.mouse.click(real_x, real_y)  # focus the target field first
        page.keyboard.type(step.text or "")
    elif step.kind == "scroll":
        page.mouse.wheel(0, _VIEWPORT["height"])
    else:
        raise ValueError(f"unexpected action kind reached _execute_step: {step.kind}")
