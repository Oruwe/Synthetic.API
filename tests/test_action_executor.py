"""Tests for the ambient RPA action loop: observe (screenshot) -> decide
(vision model) -> act (Playwright) -> repeat, until done/refused/stuck/
max-steps. Playwright and the vision decision are both mocked -- offline,
deterministic, consistent with the rest of the suite.

Safety behavior gets the most coverage here on purpose: the payment/
checkout guard, the hard step ceiling, and "never raise, always return a
valid ActionWorkflow" are the properties that matter most once this loop
is allowed to click/type on a real page.
"""

from contextlib import contextmanager

from agents.common.models.action import ActionStep
from agents.web_navigator import action_executor


class _FakeMouse:
    def __init__(self):
        self.clicks = []
        self.wheels = []

    def click(self, x, y):
        self.clicks.append((x, y))

    def wheel(self, dx, dy):
        self.wheels.append((dx, dy))


class _FakeKeyboard:
    def __init__(self):
        self.typed = []

    def type(self, text):
        self.typed.append(text)


class _FakePage:
    def __init__(self):
        self.mouse = _FakeMouse()
        self.keyboard = _FakeKeyboard()
        self.screenshots = 0
        self.goto_calls = []
        self.wait_for_load_state_calls = []

    def set_default_timeout(self, ms):
        pass

    def goto(self, url, wait_until="load"):
        self.goto_calls.append(url)

    def screenshot(self, path, full_page=False):
        self.screenshots += 1
        with open(path, "wb") as f:
            f.write(b"fake-png-bytes")

    def wait_for_load_state(self, state="load", timeout=None):
        # A fake page never navigates, so this is a no-op -- it exists so
        # _execute_step's real post-action wait (see its own comment) has
        # something to call without raising AttributeError. Recorded so a
        # test can confirm it's actually invoked after every step.
        self.wait_for_load_state_calls.append(state)


class _FakeContentPage(_FakePage):
    """A _FakePage that also implements .content(), returning successive
    values from `content_sequence` (repeating the last one once
    exhausted) -- lets a test control exactly what _page_signature sees
    on each call, to deterministically drive the stall detector."""

    def __init__(self, content_sequence: list[str]):
        super().__init__()
        self._content_sequence = list(content_sequence)
        self._content_index = 0

    def content(self):
        i = min(self._content_index, len(self._content_sequence) - 1)
        self._content_index += 1
        return self._content_sequence[i]


class _FakeBrowser:
    def __init__(self, page: _FakePage):
        self._page = page

    def new_page(self, viewport=None):
        return self._page


def _patch_browser(monkeypatch, page: _FakePage):
    @contextmanager
    def fake_launched_browser(timeout_ms=None):
        yield _FakeBrowser(page)

    monkeypatch.setattr(action_executor, "launched_browser", fake_launched_browser)


def _steps_queue(monkeypatch, steps: list[ActionStep]):
    """Returns decisions from `steps` in order, one per call, regardless
    of the actual screenshot/intent/history/hint arguments passed in."""
    queue = iter(steps)

    def fake_decide(screenshot_path, intent, history, *, run_id, node_id, hint=None):
        return next(queue)

    monkeypatch.setattr(action_executor, "decide_next_action", fake_decide)


def test_loop_stops_and_succeeds_when_model_says_done(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click the button"),
            ActionStep(kind="done", reasoning="goal accomplished"),
        ],
    )

    workflow = action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r1")

    assert workflow.success is True
    assert workflow.refused_reason is None
    assert [s.kind for s in workflow.steps] == ["click", "done"]
    assert page.mouse.clicks == [(640.0, 400.0)]  # (500/1000)*1280, (500/1000)*800
    assert page.goto_calls == ["https://example.test"]


def test_execute_step_waits_for_load_state_after_a_click_that_might_navigate(tmp_path, monkeypatch):
    """A form submit is a real page navigation; page.mouse.click() doesn't
    wait for it on its own. Caught live: without this wait, a screenshot
    taken right after a submit click can still show the pre-submission
    page, driving the model to click "submit" again and again until the
    step ceiling is hit (demo_target's /article gate, over real Docker
    networking latency). Proven here at the unit level -- the mocked page
    can't reproduce the race itself, but it CAN prove the wait is actually
    called after every executed step, not just hoped for."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="submit the form"),
            ActionStep(kind="done", reasoning="goal accomplished"),
        ],
    )

    action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r1")

    # Once for the click; "done" breaks the loop before ever reaching
    # _execute_step, so exactly one call, not two.
    assert page.wait_for_load_state_calls == ["load"]


def test_execute_step_survives_wait_for_load_state_timing_out(tmp_path, monkeypatch):
    """A click that DOESN'T cause navigation (e.g. one that opens a
    JS-driven dropdown) can legitimately leave wait_for_load_state timing
    out -- that must never surface as a loop failure, only as a missed
    opportunity to shortcut the fixed post-step sleep."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))

    class _SlowPage(_FakePage):
        def wait_for_load_state(self, state="load", timeout=None):
            raise TimeoutError("Timeout 5000ms exceeded.")

    page = _SlowPage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click something inert"),
            ActionStep(kind="done", reasoning="goal accomplished"),
        ],
    )

    workflow = action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r1")

    assert workflow.success is True  # the timeout was swallowed, not propagated


def test_on_success_extract_hook_runs_before_the_browser_closes(tmp_path, monkeypatch):
    """The gated-content path's whole reason for existing: getting past a
    login/subscribe wall and reading what's now visible has to happen in
    ONE continuous browser session (a fresh HTTP fetch afterward wouldn't
    carry the session state that just proved the gate was passed). Prove
    the hook actually receives the SAME live page the loop was using --
    not a fresh/closed one -- by having it read something only visible in
    that live session, and prove it's stored on the returned workflow."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(monkeypatch, [ActionStep(kind="done", reasoning="gate passed")])

    captured_pages = []

    def extract(live_page):
        captured_pages.append(live_page)
        return "the real article text, now visible"

    workflow = action_executor.execute_action_loop(
        "get past the gate", "https://example.test", run_id="r10", on_success_extract=extract
    )

    assert workflow.success is True
    assert workflow.extracted_text == "the real article text, now visible"
    assert captured_pages == [page]  # the hook saw the actual live page, not something else


def test_on_success_extract_hook_is_not_called_on_a_non_success_outcome(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(monkeypatch, [ActionStep(kind="stuck", reasoning="couldn't find the field")])

    called = {"value": False}

    def extract(live_page):
        called["value"] = True
        return "should never be reached"

    workflow = action_executor.execute_action_loop(
        "get past the gate", "https://example.test", run_id="r11", on_success_extract=extract
    )

    assert workflow.success is False
    assert workflow.extracted_text is None
    assert called["value"] is False


def test_on_success_extract_hook_failure_does_not_undo_a_real_success(tmp_path, monkeypatch):
    """A broken extraction (e.g. trafilatura chokes on odd markup) must
    not turn an actual, successfully-completed gate-pass into a failure
    -- the physical action already happened."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(monkeypatch, [ActionStep(kind="done", reasoning="gate passed")])

    def broken_extract(live_page):
        raise RuntimeError("extraction blew up")

    workflow = action_executor.execute_action_loop(
        "get past the gate", "https://example.test", run_id="r12", on_success_extract=broken_extract
    )

    assert workflow.success is True  # the real success is preserved
    assert workflow.extracted_text is None  # just no content came out of the broken hook


def test_loop_stops_when_model_refuses(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(monkeypatch, [ActionStep(kind="refused", reasoning="this requires entering a credit card")])

    workflow = action_executor.execute_action_loop("buy the item", "https://example.test", run_id="r2")

    assert workflow.success is False
    assert workflow.refused_reason == "this requires entering a credit card"
    assert page.mouse.clicks == []  # never executed anything


def test_payment_guard_overrides_a_click_the_model_itself_did_not_refuse(tmp_path, monkeypatch):
    """The code-side backstop: even if the model doesn't self-refuse,
    a click whose own stated reasoning mentions checkout/payment must be
    blocked before it's ever executed."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [ActionStep(kind="click", x=500, y=500, reasoning="Click 'Place Order' to complete purchase")],
    )

    workflow = action_executor.execute_action_loop("buy the item", "https://example.test", run_id="r3")

    assert workflow.success is False
    assert workflow.refused_reason is not None
    assert "payment/checkout safety guard" in workflow.refused_reason
    assert page.mouse.clicks == []  # blocked before execution


def test_payment_guard_checks_typed_text_too(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [ActionStep(kind="type", x=500, y=500, text="4111 1111 1111 1111", reasoning="enter card number")],
    )

    workflow = action_executor.execute_action_loop("checkout", "https://example.test", run_id="r4")

    assert workflow.success is False
    assert workflow.refused_reason is not None
    assert page.keyboard.typed == []  # blocked before execution


def test_loop_stops_when_model_gets_stuck(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(monkeypatch, [ActionStep(kind="stuck", reasoning="model response was unparseable")])

    workflow = action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r5")

    assert workflow.success is False
    assert workflow.refused_reason is None
    assert workflow.steps[-1].kind == "stuck"


def test_loop_respects_the_max_steps_ceiling(tmp_path, monkeypatch):
    """A model that never says done/refused/stuck must not loop forever."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)

    def always_scroll(screenshot_path, intent, history, *, run_id, node_id, hint=None):
        return ActionStep(kind="scroll", reasoning="keep looking")

    monkeypatch.setattr(action_executor, "decide_next_action", always_scroll)

    workflow = action_executor.execute_action_loop("find it", "https://example.test", run_id="r6", max_steps=3)

    assert len(workflow.steps) == 3
    assert workflow.success is False
    assert page.mouse.wheels == [(0, 800)] * 3


def test_execute_step_maps_normalized_coordinates_to_viewport_pixels(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=0, y=0, reasoning="top-left"),
            ActionStep(kind="click", x=1000, y=1000, reasoning="bottom-right"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )

    action_executor.execute_action_loop("test coords", "https://example.test", run_id="r7")

    assert page.mouse.clicks == [(0.0, 0.0), (1280.0, 800.0)]


class _FakeLogger:
    """Captures (event, kwargs) pairs instead of rendering them, so a test
    can assert on the actual logged fields directly -- same pattern as
    test_lyzr_wrapper.py's own _FakeLogger. Deliberately NOT asserting on
    rendered log text (via capsys/caplog) for these: this codebase's
    structlog is configured once, globally, by whichever test module
    happens to import something that calls configure_logging() first
    (agents/common/logging.py) -- so the actual rendering (JSON vs
    key=value) depends on test execution order and differs between
    running this file alone vs. the full suite. Learned the hard way:
    the first version of these tests asserted on rendered substrings like
    "candidates_in_radius=0" and passed in isolation, then failed in the
    full suite once an earlier-run test's configure_logging() call had
    already switched the global renderer to JSON, where the same field
    renders as "candidates_in_radius": 0 instead."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def info(self, event, **kwargs):
        self.calls.append((event, kwargs))

    def warning(self, event, **kwargs):
        self.calls.append((event, kwargs))


class _FakeSnapPage(_FakePage):
    """A _FakePage that also implements .evaluate, returning whatever
    `snap_result` is set to -- standing in for _SNAP_TO_CLICKABLE_JS's
    real return value: a dict with a `moved` key (an [x, y] pair, or
    None/absent when the model's own coordinate already landed on the
    chosen target) plus diagnostic fields (`direct`, `candidatesInRadius`,
    `nearestClickable`, `nearestClickableDist`) that _snap_to_clickable
    logs but doesn't act on."""

    def __init__(self, snap_result):
        super().__init__()
        self.snap_result = snap_result
        self.evaluate_calls = []

    def evaluate(self, script, arg):
        self.evaluate_calls.append(arg)
        return self.snap_result


def test_execute_step_snaps_a_click_onto_a_nearby_clickable_element(tmp_path, monkeypatch):
    """Found live: a manual browser test proved demo_target's gate forms
    work fine, isolating a run's repeated failed "submit" clicks to the
    model's own coordinate guess landing on dead space next to the real
    button. When the page-side snap logic finds a real clickable element
    near the model's guess, the click must land THERE, not at the raw
    model coordinates."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    # the button's real center, a bit off from the model's guess
    page = _FakeSnapPage(snap_result={"moved": [650.0, 410.0], "direct": "DIV", "candidatesInRadius": 1})
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click the button"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )

    action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r1")

    assert page.mouse.clicks == [(650.0, 410.0)]
    assert page.evaluate_calls == [[640.0, 400.0, action_executor._CLICK_SNAP_RADIUS_PX, "click the button"]]


def test_execute_step_does_not_move_a_click_already_on_a_clickable_element(tmp_path, monkeypatch):
    """The snap JS reports `moved: null` when the model's own coordinate
    is already on the chosen target -- an already-correct click must be
    executed exactly where the model aimed it, never relocated onto some
    other nearby control."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakeSnapPage(snap_result={"moved": None, "direct": "BUTTON#gate-submit-btn", "candidatesInRadius": 1})
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click the button"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )

    action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r1")

    assert page.mouse.clicks == [(640.0, 400.0)]  # unchanged: (500/1000)*1280, (500/1000)*800


def test_execute_step_logs_distinguish_no_candidates_nearby_from_already_correct(tmp_path, monkeypatch):
    """A THIRD live round still failed after the reasoning-aware snap
    fix, logging click_snap_unchanged on every attempt with no way to
    tell, from that alone, whether it meant "already correctly on
    target" (should have worked) or "nothing clickable found nearby at
    all" (a miss too far for any snap radius to rescue). This is the fix
    for THAT blind spot: the diagnostic fields (direct element,
    candidate count, and the single nearest real clickable element's
    identity/distance even when it's outside the radius) must be present
    in the log line so a live run is self-diagnosing without needing a
    screenshot handed back and forth."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakeSnapPage(
        snap_result={
            "moved": None,
            "direct": "DIV",
            "candidatesInRadius": 0,
            "nearestClickable": 'BUTTON#gate-submit-btn "Subscribe to continue reading"',
            "nearestClickableDist": 187,
        }
    )
    _patch_browser(monkeypatch, page)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(action_executor, "logger", fake_logger)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click subscribe"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )

    action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r-diag")

    snap_calls = [kwargs for event, kwargs in fake_logger.calls if event == "click_snap_result"]
    assert len(snap_calls) == 1
    assert snap_calls[0]["candidates_in_radius"] == 0
    assert snap_calls[0]["nearest_clickable_dist_px"] == 187
    assert "gate-submit-btn" in snap_calls[0]["nearest_clickable"]


def test_execute_step_click_survives_evaluate_raising(tmp_path, monkeypatch):
    """A page that can't run .evaluate at all (a fake without it, a
    cross-origin frame, anything) must fall back to the model's raw
    coordinates rather than fail the step."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))

    class _BoomOnEvaluatePage(_FakePage):
        def evaluate(self, script, arg):
            raise RuntimeError("execution context was destroyed")

    page = _BoomOnEvaluatePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click the button"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )

    workflow = action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r1")

    assert workflow.success is True
    assert page.mouse.clicks == [(640.0, 400.0)]  # fell back to the raw coordinates


def test_execute_step_logs_when_the_snap_evaluate_call_fails(tmp_path, monkeypatch):
    """Two live rounds of "the fix should work but the exact same failure
    kept recurring" had no way to tell, from the run's OWN logs, whether
    _snap_to_clickable was even reaching page.evaluate successfully on
    the user's real environment -- it silently swallowed every exception.
    This is the fix for that blind spot: a failure must be visible in the
    run's logs, not just invisible inside a try/except."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))

    class _BoomOnEvaluatePage(_FakePage):
        def evaluate(self, script, arg):
            raise RuntimeError("execution context was destroyed")

    page = _BoomOnEvaluatePage()
    _patch_browser(monkeypatch, page)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(action_executor, "logger", fake_logger)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click the button"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )

    action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r-log-fail")

    failures = [kwargs for event, kwargs in fake_logger.calls if event == "click_snap_evaluate_failed"]
    assert len(failures) == 1
    assert failures[0]["error"] == "execution context was destroyed"


def test_execute_step_logs_when_the_snap_moves_a_click(tmp_path, monkeypatch):
    """The success path is logged too -- so a live run's logs show
    whether a click actually got relocated, not just whether the
    mechanism ran without raising."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakeSnapPage(snap_result={"moved": [650.0, 410.0], "direct": "DIV", "candidatesInRadius": 1})
    _patch_browser(monkeypatch, page)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(action_executor, "logger", fake_logger)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click the button"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )

    action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r-log-moved")

    snap_calls = [kwargs for event, kwargs in fake_logger.calls if event == "click_snap_result"]
    assert len(snap_calls) == 1
    assert snap_calls[0]["moved"] == [650.0, 410.0]


def test_loop_never_raises_on_a_browser_launch_failure(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))

    @contextmanager
    def broken_launch(timeout_ms=None):
        raise RuntimeError("no chromium binary")
        yield  # pragma: no cover - unreachable, makes this a generator

    monkeypatch.setattr(action_executor, "launched_browser", broken_launch)

    workflow = action_executor.execute_action_loop("do the thing", "https://example.test", run_id="r8")

    assert workflow.success is False
    assert workflow.steps[-1].kind == "stuck"
    assert "no chromium binary" in workflow.steps[-1].reasoning


# --- execute_login_and_extract -----------------------------------------
#
# Security is the actual subject under test here, not just behavior: the
# whole reason this function exists separately from execute_action_loop
# is that a password must never reach the vision model or the audit
# trail. Every test below either proves the feature works (real
# keystrokes happen) or proves the guarantee holds (the real value never
# appears anywhere it shouldn't) -- several do both at once.


def test_execute_login_and_extract_succeeds_and_redacts_the_password_in_the_audit_trail(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=200, y=200, reasoning="the email field"),
            ActionStep(kind="click", x=200, y=300, reasoning="the password field"),
            ActionStep(kind="click", x=200, y=400, reasoning="the login button"),
            ActionStep(kind="done", reasoning="member content is now visible"),
        ],
    )

    workflow = action_executor.execute_login_and_extract(
        email="judge@example.com", password="hunter2", start_url="https://example.test", run_id="r1"
    )

    assert workflow.success is True
    # the REAL keystrokes happened -- the feature actually works
    assert "judge@example.com" in page.keyboard.typed
    assert "hunter2" in page.keyboard.typed
    # but the AUDIT TRAIL (what gets persisted/logged/could reach a
    # future model prompt) never contains the real password
    type_steps = [s for s in workflow.steps if s.kind == "type"]
    assert any(s.text == "judge@example.com" for s in type_steps)  # email is fine to keep
    assert any(s.text == "[REDACTED]" for s in type_steps)
    assert not any(s.text == "hunter2" for s in workflow.steps)


def test_execute_login_and_extract_password_never_appears_in_the_serialized_workflow(tmp_path, monkeypatch):
    """The property that actually matters: not just 'the field says
    REDACTED' but 'the real string is not present ANYWHERE in the
    object that gets persisted to disk / returned over the API.'"""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=200, y=200, reasoning="the email field"),
            ActionStep(kind="click", x=200, y=300, reasoning="the password field"),
            ActionStep(kind="click", x=200, y=400, reasoning="the login button"),
            ActionStep(kind="done", reasoning="member content is now visible"),
        ],
    )

    workflow = action_executor.execute_login_and_extract(
        email="judge@example.com", password="correct-horse-battery-staple", start_url="https://example.test", run_id="r2"
    )

    assert "correct-horse-battery-staple" not in workflow.model_dump_json()


def test_execute_login_and_extract_never_sends_the_password_to_the_vision_model(tmp_path, monkeypatch):
    """Direct proof of the core security property: capture every argument
    passed to decide_next_action across the whole flow (prompt/intent
    text AND the history it builds from prior steps) and assert the real
    password is in none of it."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)

    captured_intents = []
    queue = iter(
        [
            ActionStep(kind="click", x=200, y=200, reasoning="the email field"),
            ActionStep(kind="click", x=200, y=300, reasoning="the password field"),
            ActionStep(kind="click", x=200, y=400, reasoning="the login button"),
            ActionStep(kind="done", reasoning="member content is now visible"),
        ]
    )

    def fake_decide(screenshot_path, intent, history, *, run_id, node_id, hint=None):
        captured_intents.append(intent)
        for h in history:
            captured_intents.append(str(h.text))
        return next(queue)

    monkeypatch.setattr(action_executor, "decide_next_action", fake_decide)

    action_executor.execute_login_and_extract(
        email="judge@example.com", password="hunter2", start_url="https://example.test", run_id="r3"
    )

    assert not any("hunter2" in text for text in captured_intents)


def test_execute_login_and_extract_with_email_only_skips_the_password_field(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=200, y=200, reasoning="the email field"),
            ActionStep(kind="click", x=200, y=400, reasoning="the login button"),
            ActionStep(kind="done", reasoning="member content is now visible"),
        ],
    )

    workflow = action_executor.execute_login_and_extract(
        email="judge@example.com", password=None, start_url="https://example.test", run_id="r4"
    )

    assert workflow.success is True
    assert page.keyboard.typed == ["judge@example.com"]


def test_execute_login_and_extract_refuses_on_a_payment_shaped_submit_button(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=200, y=200, reasoning="the email field"),
            ActionStep(kind="click", x=200, y=300, reasoning="the password field"),
            ActionStep(kind="click", x=200, y=400, reasoning="click 'Confirm Payment' to submit"),
        ],
    )

    workflow = action_executor.execute_login_and_extract(
        email="judge@example.com", password="hunter2", start_url="https://example.test", run_id="r5"
    )

    assert workflow.success is False
    assert workflow.refused_reason is not None
    assert "payment" in workflow.refused_reason.lower()


def test_execute_login_and_extract_returns_stuck_without_launching_a_browser_when_no_credentials_given(monkeypatch):
    launched = {"value": False}

    @contextmanager
    def fake_launched_browser(timeout_ms=None):
        launched["value"] = True
        yield None

    monkeypatch.setattr(action_executor, "launched_browser", fake_launched_browser)

    workflow = action_executor.execute_login_and_extract(
        email=None, password=None, start_url="https://example.test", run_id="r6"
    )

    assert workflow.success is False
    assert workflow.steps[0].kind == "stuck"
    assert launched["value"] is False


def test_execute_login_and_extract_calls_the_extract_hook_on_success(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=200, y=200, reasoning="the email field"),
            ActionStep(kind="click", x=200, y=400, reasoning="the login button"),
            ActionStep(kind="done", reasoning="member content is now visible"),
        ],
    )

    workflow = action_executor.execute_login_and_extract(
        email="judge@example.com",
        password=None,
        start_url="https://example.test",
        run_id="r7",
        on_success_extract=lambda live_page: "the unlocked article text",
    )

    assert workflow.extracted_text == "the unlocked article text"


# --- extract_visible_text --------------------------------------------


class _FakeExtractPage:
    def __init__(self, html: str, body_text: str = ""):
        self._html = html
        self._body_text = body_text

    def content(self):
        return self._html

    def inner_text(self, selector):
        assert selector == "body"
        return self._body_text


def test_extract_visible_text_uses_trafilatura_on_article_shaped_html():
    html = "<html><body><article><p>" + "This is real article content. " * 20 + "</p></article></body></html>"
    page = _FakeExtractPage(html)

    text = action_executor.extract_visible_text(page)

    assert "real article content" in text


def test_extract_visible_text_falls_back_to_raw_body_text_when_trafilatura_finds_nothing():
    """Right after a form submission, the "success" state can be a sparse
    confirmation banner, not an article-shaped page -- trafilatura may
    reasonably find nothing structured there. Content visibly on screen
    must still come back, not silently turn into an empty string."""
    page = _FakeExtractPage(html="<html></html>", body_text="Subscribed: judge@example.com")

    text = action_executor.extract_visible_text(page)

    assert text == "Subscribed: judge@example.com"


# --- replay_workflow ------------------------------------------------


def _prior_workflow(steps, start_url="https://example.test"):
    from datetime import datetime, timezone

    from agents.common.models.action import WorkflowMemory

    now = datetime.now(timezone.utc)
    return WorkflowMemory(
        canonical_key="example.test:book a table",
        domain="example.test",
        representative_intent="book a table",
        start_url=start_url,
        steps=steps,
        success_count=3,
        failure_count=0,
        created_at=now,
        last_used_at=now,
        last_success_at=now,
    )


def test_replay_workflow_reexecutes_recorded_steps_without_calling_the_model(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)

    def fail_if_called(*a, **k):
        raise AssertionError("decide_next_action must not be called during a replay")

    monkeypatch.setattr(action_executor, "decide_next_action", fail_if_called)

    prior = _prior_workflow(
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click search"),
            ActionStep(kind="type", x=500, y=600, text="table for two", reasoning="type query"),
            ActionStep(kind="done", reasoning="done"),
        ]
    )

    workflow = action_executor.replay_workflow(prior, run_id="r10")

    assert workflow.success is True
    assert page.mouse.clicks == [(640.0, 400.0), (640.0, 480.0)]  # click step + type's focus-click
    assert page.keyboard.typed == ["table for two"]
    assert page.goto_calls == ["https://example.test"]


def test_replay_workflow_applies_the_payment_guard_before_executing(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    prior = _prior_workflow([ActionStep(kind="click", x=500, y=500, reasoning="click 'Place Order' to checkout")])

    workflow = action_executor.replay_workflow(prior, run_id="r11")

    assert workflow.success is False
    assert workflow.refused_reason is not None
    assert page.mouse.clicks == []


def test_replay_workflow_falls_back_to_live_loop_on_failure(tmp_path, monkeypatch):
    """A stored (x, y) sequence is only as good as the page it was recorded
    against -- if replay execution raises partway (e.g. the page structure
    changed), it must fall back to a fresh live loop rather than returning
    a partial, unverified workflow."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))

    class _BrokenPage(_FakePage):
        def screenshot(self, path, full_page=False):
            raise RuntimeError("page crashed mid-replay")

    page = _BrokenPage()
    _patch_browser(monkeypatch, page)
    live_loop_calls = []

    def fake_live_loop(intent, start_url, run_id, max_steps=None):
        live_loop_calls.append((intent, start_url, run_id))
        return action_executor.ActionWorkflow(
            run_id=run_id, intent=intent, start_url=start_url, steps=[], success=True, refused_reason=None,
            created_at=action_executor.datetime.now(action_executor.timezone.utc),
        )

    monkeypatch.setattr(action_executor, "execute_action_loop", fake_live_loop)
    prior = _prior_workflow([ActionStep(kind="click", x=500, y=500, reasoning="click search")])

    workflow = action_executor.replay_workflow(prior, run_id="r12")

    assert live_loop_calls == [("book a table", "https://example.test", "r12")]
    assert workflow.success is True


def test_execute_step_type_focuses_the_field_before_typing(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakePage()
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="type", x=500, y=500, text="hello", reasoning="type into search box"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )

    action_executor.execute_action_loop("search for hello", "https://example.test", run_id="r9")

    assert page.mouse.clicks == [(640.0, 400.0)]
    assert page.keyboard.typed == ["hello"]


# --- _page_signature ---------------------------------------------------


def test_page_signature_reflects_content_changes():
    page = _FakeContentPage(content_sequence=["<html>A</html>", "<html>B</html>"])
    sig_a = action_executor._page_signature(page)
    sig_b = action_executor._page_signature(page)
    assert sig_a != sig_b


def test_page_signature_is_stable_for_identical_content():
    page = _FakeContentPage(content_sequence=["<html>A</html>"])
    sig1 = action_executor._page_signature(page)
    sig2 = action_executor._page_signature(page)
    assert sig1 == sig2


def test_page_signature_returns_none_on_failure():
    class _BrokenContentPage(_FakePage):
        def content(self):
            raise RuntimeError("page crashed")

    assert action_executor._page_signature(_BrokenContentPage()) is None


# --- _click_anywhere_by_reasoning (tier-2 fallback grounding) ----------


class _FakeEvaluatePage(_FakePage):
    def __init__(self, evaluate_result):
        super().__init__()
        self.evaluate_result = evaluate_result
        self.evaluate_calls = []

    def evaluate(self, script, arg):
        self.evaluate_calls.append(arg)
        return self.evaluate_result


def test_click_anywhere_by_reasoning_clicks_the_matched_element():
    page = _FakeEvaluatePage(evaluate_result=[820.0, 60.0])

    found = action_executor._click_anywhere_by_reasoning(page, "Clicking the 'Sign up' link")

    assert found is True
    assert page.mouse.clicks == [(820.0, 60.0)]
    assert page.evaluate_calls == ["Clicking the 'Sign up' link"]


def test_click_anywhere_by_reasoning_returns_false_when_nothing_matches():
    page = _FakeEvaluatePage(evaluate_result=None)

    found = action_executor._click_anywhere_by_reasoning(page, "some vague reasoning")

    assert found is False
    assert page.mouse.clicks == []


def test_click_anywhere_by_reasoning_survives_evaluate_raising():
    class _BoomPage(_FakePage):
        def evaluate(self, script, arg):
            raise RuntimeError("execution context was destroyed")

    found = action_executor._click_anywhere_by_reasoning(_BoomPage(), "click subscribe")

    assert found is False


# --- Stall detection + tier-2 recovery, wired into execute_action_loop -


def test_loop_recovers_from_a_proven_stall_via_the_whole_page_fallback(tmp_path, monkeypatch):
    """Two consecutive executed clicks that provably don't change the
    page (proven via _page_signature, not inferred from reasoning text)
    must trigger the whole-page text-match fallback -- and when THAT
    finds a real target, the loop must actually click it and keep going,
    not just report the same failure a third time."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    # 4 .content() calls expected: initial signature, after click 1
    # (unchanged), after click 2 (unchanged -> triggers fallback), and
    # once more after a successful fallback click (changed, simulating
    # the fallback actually landing on the real target).
    page = _FakeContentPage(content_sequence=["A", "A", "A", "B"])
    _patch_browser(monkeypatch, page)
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click subscribe"),
            ActionStep(kind="click", x=500, y=500, reasoning="click subscribe"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )
    fallback_calls = []

    def fake_fallback(page_arg, reasoning, run_id=None):
        fallback_calls.append(reasoning)
        return True

    monkeypatch.setattr(action_executor, "_click_anywhere_by_reasoning", fake_fallback)

    workflow = action_executor.execute_action_loop("subscribe", "https://example.test", run_id="r-stall")

    assert fallback_calls == ["click subscribe"]
    assert workflow.success is True
    assert any("stall recovery" in s.reasoning for s in workflow.steps)


def test_loop_escalates_to_a_correction_hint_when_the_fallback_also_finds_nothing(tmp_path, monkeypatch):
    """When even the whole-page fallback can't find a match, the loop
    must not just silently repeat the identical question a third time --
    the model must be told explicitly that its last estimate didn't
    land."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakeContentPage(content_sequence=["A", "A", "A"])
    _patch_browser(monkeypatch, page)
    monkeypatch.setattr(action_executor, "_click_anywhere_by_reasoning", lambda page_arg, reasoning, run_id=None: False)

    hints_seen = []
    queue = iter(
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click subscribe"),
            ActionStep(kind="click", x=500, y=500, reasoning="click subscribe"),
            ActionStep(kind="done", reasoning="done"),
        ]
    )

    def fake_decide(screenshot_path, intent, history, *, run_id, node_id, hint=None):
        hints_seen.append(hint)
        return next(queue)

    monkeypatch.setattr(action_executor, "decide_next_action", fake_decide)

    action_executor.execute_action_loop("subscribe", "https://example.test", run_id="r-hint")

    assert hints_seen == [None, None, action_executor._STALL_CORRECTION_HINT]


def test_loop_does_not_stall_when_the_page_keeps_changing(tmp_path, monkeypatch):
    """The stall detector must never fire on a normal, successfully
    progressing run -- a different content signature after every step is
    exactly what real progress looks like, not a stall."""
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakeContentPage(content_sequence=["A", "B", "C"])
    _patch_browser(monkeypatch, page)
    fallback_calls = []
    monkeypatch.setattr(
        action_executor, "_click_anywhere_by_reasoning", lambda page_arg, reasoning, run_id=None: fallback_calls.append(1)
    )
    _steps_queue(
        monkeypatch,
        [
            ActionStep(kind="click", x=500, y=500, reasoning="click email field"),
            ActionStep(kind="click", x=500, y=500, reasoning="click subscribe"),
            ActionStep(kind="done", reasoning="done"),
        ],
    )

    workflow = action_executor.execute_action_loop("subscribe", "https://example.test", run_id="r-normal")

    assert fallback_calls == []  # the fallback must never have been invoked
    assert workflow.success is True


# --- Stall recovery in execute_login_and_extract's submit click -------


def test_login_submit_recovers_from_a_stall_via_the_whole_page_fallback(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "screenshot_dir", str(tmp_path))
    page = _FakeContentPage(content_sequence=["A", "A"])  # pre-submit, post-submit: unchanged
    _patch_browser(monkeypatch, page)
    fallback_calls = []

    def fake_fallback(page_arg, reasoning, run_id=None):
        fallback_calls.append(reasoning)
        return True

    monkeypatch.setattr(action_executor, "_click_anywhere_by_reasoning", fake_fallback)
    queue = iter(
        [
            ActionStep(kind="click", x=200, y=200, reasoning="the email field"),
            ActionStep(kind="click", x=200, y=400, reasoning="the login button"),
            ActionStep(kind="done", reasoning="member content is now visible"),
        ]
    )
    monkeypatch.setattr(
        action_executor,
        "decide_next_action",
        lambda screenshot_path, intent, history, *, run_id, node_id, hint=None: next(queue),
    )

    workflow = action_executor.execute_login_and_extract(
        email="judge@example.com", password=None, start_url="https://example.test", run_id="r-login-stall"
    )

    assert fallback_calls == ["the login button"]
    assert workflow.success is True
    assert any("stall recovery" in s.reasoning for s in workflow.steps)
