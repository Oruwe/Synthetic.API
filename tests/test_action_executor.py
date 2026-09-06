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

    def set_default_timeout(self, ms):
        pass

    def goto(self, url, wait_until="load"):
        self.goto_calls.append(url)

    def screenshot(self, path, full_page=False):
        self.screenshots += 1
        with open(path, "wb") as f:
            f.write(b"fake-png-bytes")


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
    of the actual screenshot/intent/history arguments passed in."""
    queue = iter(steps)

    def fake_decide(screenshot_path, intent, history, *, run_id, node_id):
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

    def always_scroll(screenshot_path, intent, history, *, run_id, node_id):
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
