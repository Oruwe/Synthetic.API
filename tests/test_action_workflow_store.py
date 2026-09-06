"""Tests for the ambient RPA action path's Qdrant functions:
upsert_action_workflow (store every attempt, success or not) and
find_similar_workflow (semantic lookup of a past SUCCESSFUL workflow to
replay). Fake client, stubbed embedder -- offline, deterministic.

Regression coverage: find_similar_workflow's stored payload carries a
"point_key" field (the same dedup convention every other collection in
this module uses) that isn't part of the ActionWorkflow schema, and the
model is extra="forbid" -- validating the raw payload back into the model
without dropping that key raises on every real point and silently looks
identical to "no similar workflow found."
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import agents.common.qdrant_store as qdrant_store
from agents.common.models.action import ActionStep, ActionWorkflow


class _FakeCollections:
    def __init__(self, names):
        self.collections = [SimpleNamespace(name=n) for n in names]


class FakeClient:
    def __init__(self, query_points_result=None):
        self._collection_names = {"action_workflows"}
        self.upsert_calls = []
        self.query_points_calls = []
        self._query_points_result = query_points_result or SimpleNamespace(points=[])

    def get_collections(self):
        return _FakeCollections(self._collection_names)

    def create_collection(self, **kwargs):
        self._collection_names.add(kwargs["collection_name"])

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)

    def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return self._query_points_result


def _workflow(success=True, intent="find the cheapest flight to Goa"):
    return ActionWorkflow(
        run_id="run-1",
        intent=intent,
        start_url="https://example.test",
        steps=[ActionStep(kind="click", x=500, y=500, reasoning="click search"), ActionStep(kind="done", reasoning="done")],
        success=success,
        refused_reason=None,
        created_at=datetime.now(timezone.utc),
    )


def _stored_payload(workflow: ActionWorkflow, point_key: str = "run-1:find the cheapest flight to Goa") -> dict:
    """What upsert_action_workflow actually writes -- including the
    point_key field that isn't part of the ActionWorkflow schema."""
    return {
        "run_id": workflow.run_id,
        "intent": workflow.intent,
        "start_url": workflow.start_url,
        "steps": [s.model_dump(mode="json") for s in workflow.steps],
        "success": workflow.success,
        "refused_reason": workflow.refused_reason,
        "created_at": workflow.created_at.isoformat(),
        "point_key": point_key,
    }


# --- upsert_action_workflow --------------------------------------------


def test_upsert_action_workflow_stores_the_full_step_sequence(monkeypatch):
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeClient()
    monkeypatch.setattr(qdrant_store, "_client", client)
    workflow = _workflow()

    point_id = qdrant_store.upsert_action_workflow(workflow, client=client)

    assert point_id == "run-1:find the cheapest flight to Goa"
    assert len(client.upsert_calls) == 1
    payload = client.upsert_calls[0]["points"][0].payload
    assert payload["intent"] == workflow.intent
    assert payload["success"] is True
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["kind"] == "click"


def test_upsert_action_workflow_persists_a_failed_attempt_too(monkeypatch):
    """A refused/stuck attempt is still recorded -- it's audit signal even
    though find_similar_workflow will never replay it."""
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeClient()
    monkeypatch.setattr(qdrant_store, "_client", client)
    workflow = _workflow(success=False)

    qdrant_store.upsert_action_workflow(workflow, client=client)

    payload = client.upsert_calls[0]["points"][0].payload
    assert payload["success"] is False


def test_upsert_action_workflow_never_raises_on_qdrant_failure(monkeypatch):
    class BrokenClient(FakeClient):
        def upsert(self, **kwargs):
            raise RuntimeError("qdrant is down")

    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = BrokenClient()
    monkeypatch.setattr(qdrant_store, "_client", client)

    point_id = qdrant_store.upsert_action_workflow(_workflow(), client=client)

    assert point_id  # still returns a point id even though the write failed


# --- find_similar_workflow ----------------------------------------------


def test_find_similar_workflow_returns_a_validated_workflow_above_threshold(monkeypatch):
    """Regression test for the point_key/extra=forbid bug: a real stored
    payload (point_key included) must still validate successfully."""
    workflow = _workflow()
    fake_point = SimpleNamespace(score=0.92, payload=_stored_payload(workflow))
    client = FakeClient(query_points_result=SimpleNamespace(points=[fake_point]))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_similar_workflow("find the cheapest flight to Goa")

    assert result is not None
    assert isinstance(result, ActionWorkflow)
    assert result.intent == workflow.intent
    assert len(result.steps) == 2
    # only successful workflows are queryable -- the filter must be applied
    assert client.query_points_calls[0]["query_filter"] is not None


def test_find_similar_workflow_returns_none_below_the_score_threshold(monkeypatch):
    workflow = _workflow()
    fake_point = SimpleNamespace(score=0.5, payload=_stored_payload(workflow))
    client = FakeClient(query_points_result=SimpleNamespace(points=[fake_point]))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_similar_workflow("something unrelated", min_score=0.85)

    assert result is None


def test_find_similar_workflow_returns_none_when_nothing_matches(monkeypatch):
    client = FakeClient(query_points_result=SimpleNamespace(points=[]))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_similar_workflow("anything")

    assert result is None


def test_find_similar_workflow_never_raises_on_qdrant_failure(monkeypatch):
    class BrokenClient(FakeClient):
        def query_points(self, **kwargs):
            raise RuntimeError("qdrant is down")

    monkeypatch.setattr(qdrant_store, "_client", BrokenClient())
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_similar_workflow("anything")

    assert result is None
