"""Tests for the ambient RPA action path's production memory layer:
record_workflow_outcome (fold one execution attempt into a durable,
trust-weighted WorkflowMemory record) and find_workflow_memory (semantic
lookup gated on similarity AND accumulated trust, not similarity alone),
plus prune_stale_workflows for bounded growth. Fake client, stubbed
embedder -- offline, deterministic.

This supersedes the earlier one-point-per-run design: the whole point of
this rewrite is that repeated attempts against the SAME (domain, intent)
pair update ONE record in place, building or eroding trust, rather than
piling up near-duplicate points. See WorkflowMemory's own docstring
(agents/common/models/action.py) for the full design rationale.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import agents.common.qdrant_store as qdrant_store
from agents.common.models.action import ActionStep, ActionWorkflow, WorkflowMemory


class _FakeCollections:
    def __init__(self, names):
        self.collections = [SimpleNamespace(name=n) for n in names]


class FakeClient:
    def __init__(self, query_points_result=None, existing_points=None):
        self._collection_names = {"action_workflows"}
        self.upsert_calls = []
        self.query_points_calls = []
        self.delete_calls = []
        self._query_points_result = query_points_result or SimpleNamespace(points=[])
        # point_id -> payload dict, simulating what's already stored
        self._points: dict[str, dict] = dict(existing_points or {})

    def get_collections(self):
        return _FakeCollections(self._collection_names)

    def create_collection(self, **kwargs):
        self._collection_names.add(kwargs["collection_name"])

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)
        for point in kwargs["points"]:
            self._points[point.id] = point.payload

    def retrieve(self, **kwargs):
        return [SimpleNamespace(payload=self._points[pid]) for pid in kwargs["ids"] if pid in self._points]

    def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return self._query_points_result

    def scroll(self, **kwargs):
        records = [SimpleNamespace(id=pid, payload=payload) for pid, payload in self._points.items()]
        return records, None

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)


def _workflow(success=True, intent="find the cheapest flight to Goa", start_url="https://flights.test/search"):
    return ActionWorkflow(
        run_id="run-1",
        intent=intent,
        start_url=start_url,
        steps=[ActionStep(kind="click", x=500, y=500, reasoning="click search"), ActionStep(kind="done", reasoning="done")],
        success=success,
        refused_reason=None,
        created_at=datetime.now(timezone.utc),
    )


def _memory_payload(**overrides) -> dict:
    base = WorkflowMemory(
        canonical_key="flights.test:find the cheapest flight to goa",
        domain="flights.test",
        representative_intent="find the cheapest flight to Goa",
        start_url="https://flights.test/search",
        steps=[ActionStep(kind="done", reasoning="done")],
        success_count=3,
        failure_count=1,
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
        last_used_at=datetime.now(timezone.utc) - timedelta(hours=1),
        last_success_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    return base.model_copy(update=overrides).model_dump(mode="json")


# --- record_workflow_outcome ---------------------------------------------


def test_record_workflow_outcome_creates_a_fresh_memory_on_first_attempt(monkeypatch):
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeClient()
    monkeypatch.setattr(qdrant_store, "_client", client)

    memory = qdrant_store.record_workflow_outcome(_workflow(success=True), client=client)

    assert memory.success_count == 1
    assert memory.failure_count == 0
    assert memory.domain == "flights.test"
    assert len(memory.steps) == 2
    assert len(client.upsert_calls) == 1


def test_record_workflow_outcome_reinforces_the_same_record_on_repeated_success(monkeypatch):
    """The whole point of the rewrite: N successful attempts of the SAME
    (domain, intent) pair must update ONE point, not create N points."""
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeClient()
    monkeypatch.setattr(qdrant_store, "_client", client)

    qdrant_store.record_workflow_outcome(_workflow(success=True), client=client)
    qdrant_store.record_workflow_outcome(_workflow(success=True), client=client)
    memory = qdrant_store.record_workflow_outcome(_workflow(success=True), client=client)

    assert memory.success_count == 3
    assert memory.failure_count == 0
    assert len(client._points) == 1  # one canonical record, not three


def test_record_workflow_outcome_erodes_trust_on_failure_without_losing_the_replay_target(monkeypatch):
    """A failed attempt must update the trust counters but must NEVER
    overwrite a previously-verified good step sequence with an empty/
    unverified one -- that would destroy the thing that made the record
    trustworthy in the first place."""
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeClient()
    monkeypatch.setattr(qdrant_store, "_client", client)

    qdrant_store.record_workflow_outcome(_workflow(success=True), client=client)
    memory = qdrant_store.record_workflow_outcome(_workflow(success=False), client=client)

    assert memory.success_count == 1
    assert memory.failure_count == 1
    assert len(memory.steps) == 2  # still the good sequence from the successful attempt


def test_record_workflow_outcome_never_raises_on_qdrant_failure(monkeypatch):
    class BrokenClient(FakeClient):
        def upsert(self, **kwargs):
            raise RuntimeError("qdrant is down")

    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = BrokenClient()
    monkeypatch.setattr(qdrant_store, "_client", client)

    memory = qdrant_store.record_workflow_outcome(_workflow(), client=client)

    assert memory is not None  # still returns a usable in-memory record
    assert memory.success_count == 1


# --- find_workflow_memory --------------------------------------------------


def test_find_workflow_memory_returns_a_trusted_match_above_all_three_gates(monkeypatch):
    fake_point = SimpleNamespace(score=0.92, payload=_memory_payload())
    client = FakeClient(query_points_result=SimpleNamespace(points=[fake_point]))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_workflow_memory("find the cheapest flight to Goa", start_url="https://flights.test/x")

    assert result is not None
    assert isinstance(result, WorkflowMemory)
    assert result.domain == "flights.test"
    # domain filter must actually be applied when start_url is known
    assert client.query_points_calls[0]["query_filter"] is not None


def test_find_workflow_memory_returns_none_below_the_score_threshold(monkeypatch):
    fake_point = SimpleNamespace(score=0.5, payload=_memory_payload())
    client = FakeClient(query_points_result=SimpleNamespace(points=[fake_point]))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_workflow_memory("something unrelated", min_score=0.85)

    assert result is None


def test_find_workflow_memory_returns_none_when_success_count_too_low(monkeypatch):
    """A single lucky success is not enough trust to replay blind."""
    fake_point = SimpleNamespace(score=0.95, payload=_memory_payload(success_count=0, failure_count=0))
    client = FakeClient(query_points_result=SimpleNamespace(points=[fake_point]))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_workflow_memory("find the cheapest flight to Goa")

    assert result is None


def test_find_workflow_memory_returns_none_when_trust_ratio_too_low(monkeypatch):
    """A workflow that has started failing more than it succeeds (e.g. the
    page got redesigned) must stop being offered, even with high
    similarity and a healthy raw success_count."""
    fake_point = SimpleNamespace(score=0.95, payload=_memory_payload(success_count=2, failure_count=8))
    client = FakeClient(query_points_result=SimpleNamespace(points=[fake_point]))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_workflow_memory("find the cheapest flight to Goa")

    assert result is None


def test_find_workflow_memory_returns_none_when_nothing_matches(monkeypatch):
    client = FakeClient(query_points_result=SimpleNamespace(points=[]))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_workflow_memory("anything")

    assert result is None


def test_find_workflow_memory_never_raises_on_qdrant_failure(monkeypatch):
    class BrokenClient(FakeClient):
        def query_points(self, **kwargs):
            raise RuntimeError("qdrant is down")

    monkeypatch.setattr(qdrant_store, "_client", BrokenClient())
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_workflow_memory("anything")

    assert result is None


def test_find_workflow_memory_without_start_url_skips_the_domain_filter(monkeypatch):
    fake_point = SimpleNamespace(score=0.92, payload=_memory_payload())
    client = FakeClient(query_points_result=SimpleNamespace(points=[fake_point]))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    result = qdrant_store.find_workflow_memory("find the cheapest flight to Goa")

    assert result is not None
    assert client.query_points_calls[0]["query_filter"] is None


# --- _canonical_key / _domain_of -------------------------------------------


def test_canonical_key_is_stable_across_trivial_phrasing_differences():
    key_a = qdrant_store._canonical_key("example.test", "Book a table for two!")
    key_b = qdrant_store._canonical_key("example.test", "book a table for two")

    assert key_a == key_b


def test_canonical_key_differs_across_domains_for_the_same_intent():
    key_a = qdrant_store._canonical_key("a.test", "sign up for the newsletter")
    key_b = qdrant_store._canonical_key("b.test", "sign up for the newsletter")

    assert key_a != key_b


def test_domain_of_strips_www_and_never_raises_on_garbage():
    assert qdrant_store._domain_of("https://www.example.test/path") == "example.test"
    assert qdrant_store._domain_of("not a url at all") == ""


# --- prune_stale_workflows --------------------------------------------------


def test_prune_stale_workflows_deletes_old_untrusted_records(monkeypatch):
    old_untrusted = _memory_payload(
        canonical_key="a", success_count=0, failure_count=3, last_used_at=datetime.now(timezone.utc) - timedelta(days=30)
    )
    client = FakeClient(existing_points={"point-a": old_untrusted})
    monkeypatch.setattr(qdrant_store, "_client", client)

    deleted = qdrant_store.prune_stale_workflows(max_age_hours=24, client=client)

    assert deleted == 1
    assert client.delete_calls[0]["points_selector"].points == ["point-a"]


def test_prune_stale_workflows_never_deletes_a_trusted_record_regardless_of_age(monkeypatch):
    old_trusted = _memory_payload(
        canonical_key="b", success_count=10, failure_count=1, last_used_at=datetime.now(timezone.utc) - timedelta(days=365)
    )
    client = FakeClient(existing_points={"point-b": old_trusted})
    monkeypatch.setattr(qdrant_store, "_client", client)

    deleted = qdrant_store.prune_stale_workflows(max_age_hours=24, client=client)

    assert deleted == 0
    assert client.delete_calls == []


def test_prune_stale_workflows_never_raises_on_qdrant_failure(monkeypatch):
    class BrokenClient(FakeClient):
        def scroll(self, **kwargs):
            raise RuntimeError("qdrant is down")

    client = BrokenClient()
    monkeypatch.setattr(qdrant_store, "_client", client)

    deleted = qdrant_store.prune_stale_workflows(client=client)

    assert deleted == 0
