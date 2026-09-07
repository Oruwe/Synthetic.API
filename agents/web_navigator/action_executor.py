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

import hashlib
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


# --- Stall detection + tier-2 (whole-page, geometry-free) fallback grounding ---
#
# _snap_to_clickable (below) only ever nudges a click within a small local
# radius -- it can rescue "the model's estimate was a few pixels off," but
# it CANNOT rescue "the model's spatial estimate for this control is off
# by hundreds of pixels," which is a real, observed failure mode of a
# general (non-specialized) vision model doing pixel-precise UI grounding.
# No radius is the right radius for that: too small and it doesn't help,
# too large and it starts grabbing unrelated controls by pure geometric
# luck. The fix isn't a bigger number, it's a different STRATEGY, used
# only once the loop has PROOF (not a guess) that its last action had no
# effect: reground by matching the model's own reasoning text against
# every real clickable element on the page, not just nearby ones. This
# reduces "estimate exact pixels" (hard for a general VLM) to "does this
# element's own label appear in what the model already said it meant to
# click" (a plain substring check against real DOM text) -- using a
# signal the model already produced, not a new model call, and without
# ever depending on a pre-known selector (still true "ambient RPA": this
# generalizes to any page, not just demo_target).


def _page_signature(page) -> str | None:
    """A cheap fingerprint of the page's current visible HTML -- lets the
    loop PROVE an executed action had no effect (same signature before
    and after) instead of inferring it from the model repeating similar
    reasoning, which is a weaker, later, more expensive signal to notice.
    Best-effort: returns None on any failure (a fake Page in tests with
    no .content(), a detached frame mid-navigation, anything), and a None
    signature is never treated as equal to another None -- so stall
    detection is simply skipped for that step rather than ever false-
    triggering on two failures to observe, matching this module's
    fail-open discipline everywhere else."""
    try:
        html = page.content()
    except Exception:
        return None
    return hashlib.sha256(html.encode("utf-8", errors="ignore")).hexdigest()


_STALL_CORRECTION_HINT = (
    "Your last action did not visibly change the page -- it likely missed its target. "
    "Look very carefully at the exact pixel boundaries of the element before choosing "
    "coordinates again; consider that your previous estimate may have been off by a "
    "significant margin, not just a few pixels."
)

_FIND_BY_REASONING_ANYWHERE_JS = """
(reasoning) => {
    const ownText = (el) => (
        el.innerText || el.value || el.getAttribute("aria-label") || el.getAttribute("placeholder") || ""
    ).trim();
    const reasoningLower = (reasoning || "").toLowerCase();
    if (!reasoningLower) return null;

    const candidates = document.querySelectorAll(
        "button, a, input, textarea, select, [role=button], [role=link], [onclick]"
    );
    let best = null;
    let bestLen = 0;
    for (const el of candidates) {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) continue;
        const t = ownText(el).toLowerCase();
        // >= 3 chars, same anchor as the local snap's text-match, so a
        // stray one-letter overlap can't trigger a click somewhere
        // unrelated on the page purely by coincidence.
        if (t.length >= 3 && reasoningLower.includes(t) && t.length > bestLen) {
            best = {cx: rect.left + rect.width / 2, cy: rect.top + rect.height / 2};
            bestLen = t.length;
        }
    }
    return best ? [best.cx, best.cy] : null;
}
"""


def _click_anywhere_by_reasoning(page, reasoning: str, run_id: str | None = None) -> bool:
    """Tier-2 fallback: searches the ENTIRE page (no radius) for a real
    clickable element whose own visible text/label the model's reasoning
    names, and clicks it directly if found. Only ever invoked after the
    stall detector has PROVEN (via _page_signature, not a heuristic) that
    the model's own pixel-coordinate attempt had no effect -- this is a
    last-resort re-grounding, not a replacement for normal execution.
    Returns True if a click was executed, False if no confident text
    match exists anywhere on the page (the caller falls back to a
    corrective hint for the model instead). Never raises: a page that
    can't run .evaluate at all just means this fallback isn't available,
    same fail-open posture as _snap_to_clickable."""
    try:
        target = page.evaluate(_FIND_BY_REASONING_ANYWHERE_JS, reasoning)
    except Exception as exc:  # noqa: BLE001 - best-effort only, see docstring
        logger.warning("click_fallback_evaluate_failed", run_id=run_id, error=str(exc))
        return False
    if not target or len(target) != 2:
        logger.info("click_fallback_no_match", run_id=run_id, reasoning=reasoning)
        return False

    x, y = float(target[0]), float(target[1])
    page.mouse.click(x, y)
    try:
        page.wait_for_load_state("load", timeout=5000)
    except Exception:  # noqa: BLE001 - not every click navigates
        pass
    logger.warning("click_fallback_executed", run_id=run_id, x=round(x), y=round(y), reasoning=reasoning)
    return True


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

            # Stall detection: page_signature is PROOF an executed action had
            # no effect (not an inference from repeated-looking reasoning),
            # and hint escalates what the model is told once that's
            # confirmed twice in a row -- see the module-level comment above
            # _page_signature for the full reasoning.
            prev_signature = _page_signature(page)
            stall_count = 0
            hint: str | None = None

            for i in range(max_steps):
                screenshot_path = str(out_dir / f"step-{i:02d}.png")
                page.screenshot(path=screenshot_path, full_page=False)

                step = decide_next_action(
                    screenshot_path, intent, steps, run_id=run_id, node_id=f"action-{i}", hint=hint
                )

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

                _execute_step(page, step, run_id)
                time.sleep(0.3)  # let the page settle before the next screenshot

                new_signature = _page_signature(page)
                unchanged = (
                    prev_signature is not None and new_signature is not None and new_signature == prev_signature
                )
                if unchanged:
                    stall_count += 1
                    logger.warning("action_step_no_visible_effect", run_id=run_id, step=i, stall_count=stall_count)
                else:
                    stall_count = 0
                    hint = None

                if stall_count >= 2:
                    # Two proven-no-effect actions in a row: the model's
                    # pixel estimate isn't just a little off, it's not
                    # working at all. Re-ground against the real DOM using
                    # what the model already told us it meant to click,
                    # rather than asking the identical geometric question
                    # a third time and hoping for a different answer.
                    if _click_anywhere_by_reasoning(page, step.reasoning, run_id):
                        steps.append(
                            ActionStep(
                                kind="click",
                                reasoning=f"[stall recovery: whole-page text match] {step.reasoning}",
                            )
                        )
                        stall_count = 0
                        hint = None
                        new_signature = _page_signature(page)
                    else:
                        hint = _STALL_CORRECTION_HINT

                prev_signature = new_signature
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
                _execute_step(page, click_step, run_id)
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
                    pre_submit_signature = _page_signature(page)
                    _execute_step(page, submit_step, run_id)
                    steps.append(submit_step)
                    time.sleep(0.5)  # let the page navigate/settle after submit

                    # One geometry-independent retry if the submit click
                    # provably had no effect -- same stall-recovery logic
                    # as execute_action_loop's main loop, applied to the
                    # one highest-risk click in this flow.
                    post_submit_signature = _page_signature(page)
                    if (
                        pre_submit_signature is not None
                        and post_submit_signature is not None
                        and post_submit_signature == pre_submit_signature
                    ):
                        logger.warning("login_submit_no_visible_effect", run_id=run_id)
                        if _click_anywhere_by_reasoning(page, submit_step.reasoning, run_id):
                            steps.append(
                                ActionStep(
                                    kind="click",
                                    reasoning=f"[stall recovery: whole-page text match] {submit_step.reasoning}",
                                )
                            )
                            time.sleep(0.5)

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
                _execute_step(page, step, run_id)
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


# How far (in real screenshot pixels) to search for a nearby clickable
# element when the model's own coordinate guess misses one -- generous
# enough to forgive a modest vision-grounding miss, tight enough not to
# grab an unrelated control several UI elements away.
_CLICK_SNAP_RADIUS_PX = 60

_SNAP_TO_CLICKABLE_JS = """
([x, y, radius, reasoning]) => {
    const isClickable = (el) => {
        if (!el) return false;
        if (["BUTTON", "A", "INPUT", "TEXTAREA", "SELECT", "LABEL"].includes(el.tagName)) return true;
        const role = el.getAttribute && el.getAttribute("role");
        if (role && ["button", "link", "checkbox", "radio", "tab"].includes(role)) return true;
        return window.getComputedStyle(el).cursor === "pointer";
    };
    const ownText = (el) => (
        el.innerText || el.value || el.getAttribute("aria-label") || el.getAttribute("placeholder") || ""
    ).trim();
    const describe = (el) => el ? (el.tagName + (el.id ? "#" + el.id : "") + (ownText(el) ? " " + JSON.stringify(ownText(el).slice(0, 40)) : "")) : null;

    const direct = document.elementFromPoint(x, y);
    // Distances computed against EVERY clickable element on the page, not
    // just ones within radius -- this is what lets the caller tell "the
    // model's guess just missed by a few px" apart from "the model's
    // guess is nowhere near any real control", which only tracking
    // in-radius candidates could never distinguish.
    const all = Array.from(document.querySelectorAll(
        "button, a, input, textarea, select, [role=button], [role=link], [onclick]"
    )).map(el => {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return null;
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        return {el, cx, cy, dist: Math.hypot(cx - x, cy - y)};
    }).filter(Boolean);

    const nearest = all.reduce((best, c) => (!best || c.dist < best.dist) ? c : best, null);
    const inRadius = all.filter(c => c.dist <= radius);

    let chosen = null;
    if (inRadius.length > 0) {
        // Prefer whichever nearby candidate's own visible text/label the
        // model's reasoning actually quotes -- the model very often
        // names exactly what it means to click (e.g. "Clicking the
        // 'Subscribe to continue reading' button"). Raw pixel proximity
        // alone can't disambiguate two clickable elements only a few
        // pixels apart (e.g. an input immediately above a submit
        // button) -- this can, since it uses a signal the model already
        // gave us, not just geometry.
        const reasoningLower = (reasoning || "").toLowerCase();
        if (reasoningLower) {
            let bestLen = 0;
            for (const c of inRadius) {
                const t = ownText(c.el).toLowerCase();
                if (t.length >= 3 && reasoningLower.includes(t) && t.length > bestLen) {
                    chosen = c;
                    bestLen = t.length;
                }
            }
        }
        if (!chosen) {
            chosen = inRadius.reduce((best, c) => (!best || c.dist < best.dist) ? c : best, null);
        }
    }

    const moved = (chosen && chosen.el !== direct) ? [chosen.cx, chosen.cy] : null;
    return {
        moved,
        direct: describe(direct),
        candidatesInRadius: inRadius.length,
        nearestClickable: nearest ? describe(nearest.el) : null,
        nearestClickableDist: nearest ? Math.round(nearest.dist) : null,
    };
}
"""


def _snap_to_clickable(
    page, x: float, y: float, reasoning: str | None = None, run_id: str | None = None
) -> tuple[float, float]:
    """Nudges a click/focus target onto the nearest real clickable element
    when the model's coordinate guess landed on the wrong spot.

    Found live, in three rounds: a manual browser test proved
    demo_target's gate forms work fine end to end, which isolated a live
    run's repeated failed "submit" clicks (same reasoning, same
    non-progress, every attempt) to the model's own coordinate guess --
    not a page bug, and not the navigation-timing race an earlier fix
    addressed. The FIRST version of this function (raw nearest-clickable-
    within-radius, only triggered when the exact point wasn't already on
    something clickable) still failed the exact same way: demo_target's
    email input and submit button sit only ~8px apart, so a guess
    landing a few pixels high inside the email input's own box was
    *already* "on something clickable" by that check -- just the WRONG
    clickable element -- and the old logic left it there. The SECOND
    version added reasoning-text matching to disambiguate nearby
    candidates and was verified, live, to correctly resolve that exact
    adversarial case against a local replica -- but a THIRD live round
    still failed identically, logging click_snap_unchanged on every
    attempt with no way to tell whether that meant "already correctly on
    target" (should have worked) or "nothing clickable found nearby at
    all" (a miss far outside any reasonable snap radius). This version
    closes THAT blind spot: it always reports what's directly under the
    model's coordinate, how many real candidates are within the snap
    radius, and -- critically -- how far away and what the single
    nearest real clickable element on the whole page actually is, even
    when that's outside the radius. That last figure is what finally
    tells the caller (from logs alone, no screenshot needed) whether a
    miss is "just needs a bigger radius" or "the model's spatial
    estimate for this control is off by a wide margin, which no
    execution-side nudge can paper over."

    The model still does 100% of the visual reasoning (what to click and
    roughly where, and implicitly its label too, via reasoning it
    already produces for its own purposes) -- this only makes EXECUTION
    forgiving of a modest miss, the same way a real mouse click a few
    pixels off a button's edge still usually "counts" for a human. An
    already-correct click is never moved: only a miss (or a click that
    landed on the wrong nearby element) triggers a relocation.

    Falls back to the original coordinates on ANY failure (a fake Page in
    tests with no .evaluate, a cross-origin frame, anything else) -- this
    is a best-effort nudge, never a hard requirement for a click to
    proceed. Every outcome is logged, never swallowed silently -- see the
    version history above for why that discipline exists.
    """
    try:
        result = page.evaluate(_SNAP_TO_CLICKABLE_JS, [x, y, _CLICK_SNAP_RADIUS_PX, reasoning])
    except Exception as exc:  # noqa: BLE001 - best-effort only, see docstring
        logger.warning("click_snap_evaluate_failed", run_id=run_id, error=str(exc))
        return x, y

    result = result or {}
    moved = result.get("moved")
    logger.info(
        "click_snap_result", run_id=run_id, x=round(x), y=round(y), moved=moved, direct=result.get("direct"),
        candidates_in_radius=result.get("candidatesInRadius"), nearest_clickable=result.get("nearestClickable"),
        nearest_clickable_dist_px=result.get("nearestClickableDist"),
    )
    if moved and len(moved) == 2:
        return float(moved[0]), float(moved[1])
    return x, y


def _execute_step(page, step: ActionStep, run_id: str | None = None) -> None:
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
        real_x, real_y = _snap_to_clickable(page, real_x, real_y, step.reasoning, run_id)
        page.mouse.click(real_x, real_y)
    elif step.kind == "type":
        if real_x is not None:
            real_x, real_y = _snap_to_clickable(page, real_x, real_y, step.reasoning, run_id)
            page.mouse.click(real_x, real_y)  # focus the target field first
        page.keyboard.type(step.text or "")
    elif step.kind == "scroll":
        page.mouse.wheel(0, _VIEWPORT["height"])
    else:
        raise ValueError(f"unexpected action kind reached _execute_step: {step.kind}")

    # A click can trigger a full page navigation -- a plain HTML
    # <form method="post"> submit, not an AJAX one, exactly what
    # demo_target's gate forms use. page.mouse.click() only dispatches the
    # synthetic mouse event and returns immediately; it does NOT wait for
    # any resulting navigation. Without this, the next screenshot can be
    # taken before the new page has rendered, showing the model the SAME
    # pre-submission state -- which then reasons it needs to click
    # "submit" again, and again, until the step ceiling is hit. Caught
    # live against demo_target's /article gate (real Docker networking
    # latency exposed the race), not by the mocked test suite, where a
    # fake Page has nothing to navigate. wait_for_load_state resolves
    # near-instantly when the click did NOT cause a navigation -- the
    # page is already at the "load" state -- so this costs nothing on an
    # ordinary same-page click/type/scroll.
    try:
        page.wait_for_load_state("load", timeout=5000)
    except Exception:  # noqa: BLE001 - no navigation happened, or "load" wasn't reached in time; the caller's own post-step sleep still gives the page one more chance to settle
        pass
