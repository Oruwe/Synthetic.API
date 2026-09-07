"""Tests for agents/common/playwright_utils.py's shared browser launcher.

Real bug this guards against: no launch args at all meant Chromium used
Docker's default 64MB /dev/shm, a classic crash cause in a container --
this codebase's own docker-compose.yml doesn't work around it via
shm_size either, so it was a latent risk on every deployment, not just a
theoretical one. --disable-dev-shm-usage must always be passed; a
regression back to omitting it wouldn't show up in any of the mocked
action_executor tests (which monkeypatch launched_browser wholesale), so
this needs its own direct test.
"""

from unittest.mock import MagicMock, patch

from agents.common import playwright_utils


def test_launched_browser_always_passes_disable_dev_shm_usage():
    fake_browser = MagicMock()
    fake_chromium = MagicMock()
    fake_chromium.launch.return_value = fake_browser
    fake_playwright = MagicMock()
    fake_playwright.chromium = fake_chromium

    with patch.object(playwright_utils, "sync_playwright") as mock_sync_playwright:
        mock_sync_playwright.return_value.__enter__.return_value = fake_playwright

        with playwright_utils.launched_browser() as browser:
            assert browser is fake_browser

    launch_kwargs = fake_chromium.launch.call_args.kwargs
    assert launch_kwargs["args"] == ["--disable-dev-shm-usage"]
    assert launch_kwargs["headless"] is True


def test_launched_browser_closes_the_browser_even_if_the_caller_raises():
    fake_browser = MagicMock()
    fake_chromium = MagicMock()
    fake_chromium.launch.return_value = fake_browser
    fake_playwright = MagicMock()
    fake_playwright.chromium = fake_chromium

    with patch.object(playwright_utils, "sync_playwright") as mock_sync_playwright:
        mock_sync_playwright.return_value.__enter__.return_value = fake_playwright

        try:
            with playwright_utils.launched_browser():
                raise ValueError("boom")
        except ValueError:
            pass

    fake_browser.close.assert_called_once()


def test_launched_browser_passes_the_executable_path_override_alongside_the_launch_args(monkeypatch):
    """The override (for a sandbox with a pre-installed Chromium whose
    revision doesn't match this Playwright version) must not get dropped
    now that launch_kwargs also carries `args`."""
    monkeypatch.setattr(playwright_utils, "CHROMIUM_EXECUTABLE_OVERRIDE", "/opt/pw-browsers/chromium/chrome")
    fake_browser = MagicMock()
    fake_chromium = MagicMock()
    fake_chromium.launch.return_value = fake_browser
    fake_playwright = MagicMock()
    fake_playwright.chromium = fake_chromium

    with patch.object(playwright_utils, "sync_playwright") as mock_sync_playwright:
        mock_sync_playwright.return_value.__enter__.return_value = fake_playwright

        with playwright_utils.launched_browser():
            pass

    launch_kwargs = fake_chromium.launch.call_args.kwargs
    assert launch_kwargs["executable_path"] == "/opt/pw-browsers/chromium/chrome"
    assert launch_kwargs["args"] == ["--disable-dev-shm-usage"]
