"""Tests for the curate_candidates lifecycle: promote relevant findings to
`status=permanent`, hard-delete the rest ("majority junk"). Uses a fake
Qdrant client (no real Qdrant needed) plus a stubbed embedder, consistent
with the offline/dependency-free test suite.
"""

from types import SimpleNamespace

import agents.common.qdrant_store as qdrant_store
from agents.common.qdrant_store import cosine_similarity, should_retain


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_does_not_divide_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_should_retain_uses_configured_default_threshold():
    from agents.common.config import settings

    assert should_retain(settings.research_relevance_threshold + 0.01) is True
    assert should_retain(settings.research_relevance_threshold - 0.01) is False


def test_should_retain_respects_explicit_threshold_override():
    assert should_retain(0.5, threshold=0.6) is False
    assert should_retain(0.7, threshold=0.6) is True


class _FakeCollections:
    def __init__(self, names):
        self.collections = [SimpleNamespace(name=n) for n in names]


class FakeCurationClient:
    """Serves one page of candidate records and records what
    curate_candidates does with them (set_payload / delete calls)."""

    def __init__(self, records):
        self._records = records
        self._collection_names = {"web_knowledge"}
        self.set_payload_calls = []
        self.delete_calls = []

    def get_collections(self):
        return _FakeCollections(self._collection_names)

    def create_collection(self, **kwargs):
        self._collection_names.add(kwargs["collection_name"])

    def scroll(self, **kwargs):
        return (self._records, None)

    def set_payload(self, **kwargs):
        self.set_payload_calls.append(kwargs)

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)


def _candidate(point_id, vector, url):
    return SimpleNamespace(id=point_id, vector=vector, payload={"url": url, "status": "candidate"})


def test_curate_candidates_promotes_relevant_and_deletes_junk(monkeypatch):
    # Query embeds to [1, 0]; "relevant" candidate is aligned with it,
    # "junk" candidate is orthogonal.
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeCurationClient(
        records=[
            _candidate("relevant-id", [1.0, 0.0], "https://relevant.example"),
            _candidate("junk-id", [0.0, 1.0], "https://junk.example"),
        ]
    )
    monkeypatch.setattr(qdrant_store, "_client", client)

    result = qdrant_store.curate_candidates(run_id="run-1", query="anything", threshold=0.5)

    assert result == {"promoted": 1, "deleted": 1}
    assert client.set_payload_calls[0]["payload"] == {"status": "permanent"}
    assert client.set_payload_calls[0]["points"] == ["relevant-id"]
    assert client.delete_calls[0]["points_selector"].points == ["junk-id"]


def test_curate_candidates_promotes_nothing_when_all_below_threshold(monkeypatch):
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeCurationClient(records=[_candidate("junk-id", [0.0, 1.0], "https://junk.example")])
    monkeypatch.setattr(qdrant_store, "_client", client)

    result = qdrant_store.curate_candidates(run_id="run-1", query="anything", threshold=0.5)

    assert result == {"promoted": 0, "deleted": 1}
    assert client.set_payload_calls == []


def test_curate_candidates_no_candidates_is_a_noop(monkeypatch):
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeCurationClient(records=[])
    monkeypatch.setattr(qdrant_store, "_client", client)

    result = qdrant_store.curate_candidates(run_id="run-1", query="anything")

    assert result == {"promoted": 0, "deleted": 0}
    assert client.set_payload_calls == []
    assert client.delete_calls == []
