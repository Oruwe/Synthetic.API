"""Tests for deploy/huggingface/merged_app.py -- the single-port merge of
the Orchestrator's FastAPI app and the Gradio demo UI, built for
single-port hosts (Hugging Face Spaces) where docker-compose's separate
agents-orchestrator/demo-ui ports aren't available.

The real risk this guards against: gr.mount_gradio_app's mounted path
("/") could in principle shadow one of the Orchestrator's own routes, or a
future Gradio/FastAPI version bump could change routing precedence.
Exercised here against the REAL merged `app` object (a Starlette
TestClient, not a mock), the same way test_orchestrator_auth.py already
tests the Orchestrator's own app -- so a regression here is a real
regression, not a mocked-away one.
"""

from starlette.testclient import TestClient

from agents.common.config import settings
from deploy.huggingface.merged_app import app

client = TestClient(app)


def test_orchestrator_routes_are_not_shadowed_by_the_gradio_mount():
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_serves_the_real_gradio_ui_not_a_404_or_empty_page():
    resp = client.get("/")

    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    body = resp.text.lower()
    # Confirms this is genuinely ui/app.py's own Gradio Blocks (its title,
    # set via `gr.Blocks(title="Synthetic.API", ...)`) rendering through the
    # mount, not some other/blank page that happens to return 200.
    assert "synthetic.api" in body


def test_trigger_route_is_still_live_through_the_mount(monkeypatch):
    """A 400 on an empty transcript proves the ROUTE itself is reachable
    and running real Orchestrator logic (planner.build_plan's own
    validation) -- not swallowed by the "/" mount or returning Gradio's
    catch-all page instead."""
    monkeypatch.setattr(settings, "orchestrator_api_key", "")

    resp = client.post("/trigger", json={"transcript": ""})

    assert resp.status_code == 400
    assert "transcript is empty" in resp.json()["detail"]


def test_trigger_route_still_enforces_auth_through_the_mount(monkeypatch):
    """The merge must not accidentally bypass require_api_key -- confirms
    the SAME dependency-wired route object is being served, not a
    stripped-down copy."""
    monkeypatch.setattr(settings, "orchestrator_api_key", "secret-key")

    resp = client.post("/trigger", json={"transcript": "test"})

    assert resp.status_code == 401
