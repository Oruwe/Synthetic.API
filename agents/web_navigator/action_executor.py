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
from typing import Callable

from agents.common.config import settings
from agents.common.logging import get_logger
from agents.common.models.action import ActionStep, ActionWorkflow, WorkflowMemory
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


def execute_action_loop(
    intent: str,
    start_url: str,
    run_id: str,
    max_steps: int | None = None,
    on_success_extract: Callable[[object], str] | None = None,
) -> ActionWorkflow:
    """Runs the observe/decide/act loop against a real page. Never raises:
    a browser-level failure (bad URL, crashed page, launch failure) is
    caught and recorded as a "stuck" step so the caller always gets back
    a valid ActionWorkflow to persist and report on, same fail-open
    discipline as every other node handler in this codebase.

    `on_success_extract`, when given, is called with the live Playwright
    `Page` right after the loop reaches "done" (success=True) and BEFORE
    the browser closes -- its return value lands on the returned
    ActionWorkflow's `extracted_text`. This exists for the gated-content
    path (page_handlers.py): getting past a login/subscribe wall and
    reading what was behind it has to happen in the SAME continuous
    browser session, since a fresh plain HTTP fetch afterward wouldn't
    carry the session/cookie state that just proved "I got past the
    gate." Never called on a non-success outcome (refused/stuck/exhausted)
    -- there's nothing legitimate to extract if the gate was never passed.
    A failure inside the hook itself is caught same as any other
    browser-level failure and does not turn a successful action into a
    failed one; it just leaves `extracted_text` as None.
    """
    max_steps = max_steps if max_steps is not None else settings.action_max_steps
    out_dir = Path(settings.screenshot_dir) / run_id / "action"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps: list[ActionStep] = []
    success = False
    refused_reason: str | None = None
    extracted_text: str | None = None

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
                    if on_success_extract is not None:
                        try:
                            extracted_text = on_success_extract(page)
                        except Exception as exc:  # noqa: BLE001 - extraction failing must not undo a real success
                            logger.warning("action_extract_hook_failed", run_id=run_id, error=str(exc))
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
        extracted_text=extracted_text,
    )


def execute_login_and_extract(
    email: str | None,
    password: str | None,
    start_url: str,
    run_id: str,
    on_success_extract: Callable[[object], str] | None = None,
) -> ActionWorkflow:
    """Logs into a page's login form with human-supplied credentials, then
    (when on_success_extract is given) extracts the now-unlocked content --
    same continuous-browser-session reasoning as execute_action_loop's own
    hook, since a fresh HTTP fetch afterward wouldn't carry the session
    that just proved the login succeeded.

    SECURITY -- the entire reason this is a separate function rather than
    just calling execute_action_loop with a "log in with this password"
    intent: the vision model is NEVER told the credential value and NEVER
    asked to reproduce it. It is only ever asked WHERE a field is (a
    generic, credential-free question) via the exact same
    decide_next_action() call execute_action_loop uses for ordinary
    clicks -- so a payment-shaped submit button is still caught by the
    same _looks_like_payment_action guard. The actual keystrokes happen
    directly in Playwright code once coordinates are known; the recorded
    ActionStep for the password field always carries "[REDACTED]", never
    the real value -- that step is built here, in code, never returned by
    the model, so it never enters a future model prompt (via `history`),
    an audit log, or -- if this workflow's audit trail is ever persisted
    elsewhere -- a database.

    Never raises: same fail-open discipline as execute_action_loop.
    """
    out_dir = Path(settings.screenshot_dir) / run_id / "action"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps: list[ActionStep] = []
    success = False
    refused_reason: str | None = None
    extracted_text: str | None = None

    if not email and not password:
        # Nothing to log in with -- don't even launch a browser.
        return ActionWorkflow(
            run_id=run_id,
            intent="log in",
            start_url=start_url,
            steps=[ActionStep(kind="stuck", reasoning="no credentials were provided")],
            success=False,
            refused_reason=None,
            created_at=datetime.now(timezone.utc),
        )

    step_index = 0

    try:
        with launched_browser(PAGE_DEFAULT_TIMEOUT_MS) as browser:
            page = browser.new_page(viewport=_VIEWPORT)
            page.set_default_timeout(PAGE_DEFAULT_TIMEOUT_MS)
            page.goto(start_url, wait_until="load")

            def locate(prompt_intent: str) -> ActionStep:
                nonlocal step_index
                screenshot_path = str(out_dir / f"step-{step_index:02d}.png")
                page.screenshot(path=screenshot_path, full_page=False)
                decision = decide_next_action(
                    screenshot_path, prompt_intent, steps, run_id=run_id, node_id=f"login-{step_index}"
                )
                step_index += 1
                return decision

            for field_label, value, redact in (("email or username", email, False), ("password", password, True)):
                if not value:
                    continue
                click_step = locate(f"Click the {field_label} input field on this login form.")
                if click_step.kind == "refused" or _looks_like_payment_action(click_step):
                    refused_reason = (
                        click_step.reasoning
                        if click_step.kind == "refused"
                        else "blocked by payment/checkout safety guard: " + click_step.reasoning
                    )
                    steps.append(
                        ActionStep(kind="refused", reasoning=refused_reason, screenshot_path=click_step.screenshot_path)
                    )
                    break
                if click_step.kind != "click" or click_step.x is None or click_step.y is None:
                    steps.append(
                        ActionStep(
                            kind="stuck",
                            reasoning=f"could not locate the {field_label} field",
                            screenshot_path=click_step.screenshot_path,
                        )
                    )
                    break
                _execute_step(page, click_step)
                page.keyboard.type(value)
                logger.info("login_field_filled", run_id=run_id, field=field_label)  # never the value itself
                steps.append(
                    ActionStep(
                        kind="type",
                        x=click_step.x,
                        y=click_step.y,
                        text="[REDACTED]" if redact else value,
                        reasoning=f"entered the provided {field_label}",
                        screenshot_path=click_step.screenshot_path,
                    )
                )
                time.sleep(0.2)
            else:
                # Reached only if the loop above completed without a
                # break (i.e. every requested field was located and filled).
                submit_step = locate("Click the login/submit button to submit this form.")
                if submit_step.kind == "refused" or _looks_like_payment_action(submit_step):
                    refused_reason = (
                        submit_step.reasoning
                        if submit_step.kind == "refused"
                        else "blocked by payment/checkout safety guard: " + submit_step.reasoning
                    )
                    steps.append(
                        ActionStep(kind="refused", reasoning=refused_reason, screenshot_path=submit_step.screenshot_path)
                    )
                elif submit_step.kind == "click" and submit_step.x is not None and submit_step.y is not None:
                    _execute_step(page, submit_step)
                    steps.append(submit_step)
                    time.sleep(0.5)  # let the page navigate/settle after submit
                    confirm_step = locate(
                        "Has the login succeeded (you now see account/member-only content), or does this still "
                        'look like a login form or an error message? Respond kind="done" if it succeeded, '
                        'kind="stuck" otherwise.'
                    )
                    steps.append(confirm_step)
                    if confirm_step.kind == "done":
                        success = True
                        if on_success_extract is not None:
                            try:
                                extracted_text = on_success_extract(page)
                            except Exception as exc:  # noqa: BLE001 - extraction failing must not undo a real login
                                logger.warning("action_extract_hook_failed", run_id=run_id, error=str(exc))
                else:
                    steps.append(
                        ActionStep(
                            kind="stuck",
                            reasoning="could not locate the login/submit button",
                            screenshot_path=submit_step.screenshot_path,
                        )
                    )

    except Exception as exc:  # noqa: BLE001 - a browser-level failure must not crash the DAG node
        logger.warning("login_flow_failed", run_id=run_id, error=str(exc))
        steps.append(ActionStep(kind="stuck", reasoning=f"execution failed: {exc}"))

    return ActionWorkflow(
        run_id=run_id,
        intent="log in" + (" and extract content" if on_success_extract is not None else ""),
        start_url=start_url,
        steps=steps,
        success=success,
        refused_reason=refused_reason,
        created_at=datetime.now(timezone.utc),
        extracted_text=extracted_text,
    )


def replay_workflow(prior: WorkflowMemory, run_id: str) -> ActionWorkflow:
    """Deterministically re-executes a previously-successful workflow's
    recorded steps against a fresh page load -- no vision-model calls, so
    it's fast, costs no LLM budget, and leaves no room for a newly
    hallucinated action. This is the "Qdrant supplies the workflow" half
    of the ambient RPA pitch: a semantically-matched past success is
    replayed outright rather than re-explored from scratch.

    A stored (x, y) sequence is only as good as the page layout it was
    recorded against, though -- if the target moved or the flow changed,
    blindly continuing would click the wrong thing. So ANY failure partway
    through (a raised exception from Playwright, or the payment guard
    tripping on a step the original run never needed to guard because it
    took a different path) falls back to a fresh, live
    execute_action_loop() rather than returning a partially-executed,
    unverified workflow.
    """
    out_dir = Path(settings.screenshot_dir) / run_id / "action-replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    executed: list[ActionStep] = []

    try:
        with launched_browser(PAGE_DEFAULT_TIMEOUT_MS) as browser:
            page = browser.new_page(viewport=_VIEWPORT)
            page.set_default_timeout(PAGE_DEFAULT_TIMEOUT_MS)
            page.goto(prior.start_url, wait_until="load")

            for i, step in enumerate(prior.steps):
                if step.kind in ("done", "refused", "stuck"):
                    executed.append(step)
                    continue

                if _looks_like_payment_action(step):
                    logger.warning("action_replay_refused_payment_guard", run_id=run_id, reasoning=step.reasoning)
                    executed.append(
                        ActionStep(
                            kind="refused",
                            reasoning="blocked by payment/checkout safety guard during replay: " + step.reasoning,
                        )
                    )
                    return ActionWorkflow(
                        run_id=run_id,
                        intent=prior.representative_intent,
                        start_url=prior.start_url,
                        steps=executed,
                        success=False,
                        refused_reason=executed[-1].reasoning,
                        created_at=datetime.now(timezone.utc),
                    )

                page.screenshot(path=str(out_dir / f"step-{i:02d}.png"), full_page=False)
                _execute_step(page, step)
                executed.append(step)
                time.sleep(0.3)

    except Exception as exc:  # noqa: BLE001 - a stale/broken replay must fall back to live exploration, not fail
        logger.warning("action_replay_failed_falling_back_to_live", run_id=run_id, error=str(exc))
        return execute_action_loop(prior.representative_intent, prior.start_url, run_id=run_id)

    logger.info("action_replay_succeeded", run_id=run_id, canonical_key=prior.canonical_key, step_count=len(executed))
    return ActionWorkflow(
        run_id=run_id,
        intent=prior.representative_intent,
        start_url=prior.start_url,
        steps=executed,
        success=True,
        refused_reason=None,
        created_at=datetime.now(timezone.utc),
    )


def extract_visible_text(page) -> str:
    """Default `on_success_extract` hook (see execute_action_loop above):
    the same trafilatura-based extraction page_fetcher.py's own Playwright
    fallback uses, so a gated page read this way and a normal page read
    over plain HTTP produce comparably-shaped text for the drafter to
    work with -- not a second, different extraction quality standard.
    Falls back to raw visible body text if trafilatura finds nothing
    structured (a real possibility right after a form submission, where
    the "success" state might be a sparse confirmation banner rather than
    an article-shaped page) rather than returning empty and discarding
    content that's visibly right there on the screen.
    """
    import trafilatura

    html = page.content()
    document = trafilatura.bare_extraction(html, with_metadata=False)
    text = (document.text if document and document.text else "").strip()
    if text:
        return text
    return (page.inner_text("body") or "").strip()[:20000]


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
