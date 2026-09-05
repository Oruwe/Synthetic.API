"""Tests for the live path's Qdrant functions: upsert_page_chunks (chunk +
embed a fetched page) and semantic_search_pages (top-k vector retrieval,
never raises). Fake client, stubbed embedder -- offline, deterministic.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import agents.common.qdrant_store as qdrant_store
from agents.common.models.page import FetchedPage


class _FakeCollections:
    def __init__(self, names):
        self.collections = [SimpleNamespace(name=n) for n in names]


class FakeClient:
    def __init__(self, query_points_result=None):
        self._collection_names = {"web_pages"}
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


def _page(text="x" * 2000, error=None):
    return FetchedPage(
        url="https://example.test", title="Example", text=text if error is None else "",
        timestamp=datetime.now(timezone.utc), fetch_method="http", error=error,
    )


def test_upsert_page_chunks_embeds_and_upserts_each_chunk(monkeypatch):
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeClient()
    monkeypatch.setattr(qdrant_store, "_client", client)

    point_ids = qdrant_store.upsert_page_chunks(_page(), question="q", run_id="run-1", client=client)

    assert len(point_ids) > 1  # 2000 chars at 800/chunk with overlap -> multiple chunks
    assert len(client.upsert_calls) == len(point_ids)
    first_payload = client.upsert_calls[0]["points"][0].payload
    assert first_payload["run_id"] == "run-1"
    assert first_payload["question"] == "q"
    assert first_payload["url"] == "https://example.test"
    assert first_payload["chunk_index"] == 0


def test_upsert_page_chunks_is_a_noop_for_a_failed_fetch(monkeypatch):
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeClient()
    monkeypatch.setattr(qdrant_store, "_client", client)

    point_ids = qdrant_store.upsert_page_chunks(_page(error="fetch failed"), question="q", run_id="run-1", client=client)

    assert point_ids == []
    assert client.upsert_calls == []


def test_upsert_page_chunks_point_ids_are_idempotent_across_calls(monkeypatch):
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])
    client = FakeClient()
    monkeypatch.setattr(qdrant_store, "_client", client)

    ids_first = qdrant_store.upsert_page_chunks(_page(), question="q", run_id="run-1", client=client)
    ids_second = qdrant_store.upsert_page_chunks(_page(), question="q", run_id="run-1", client=client)

    assert ids_first == ids_second  # same run_id + url + chunk_index -> same point_key


def test_semantic_search_pages_returns_scored_points(monkeypatch):
    fake_points = [SimpleNamespace(id="p1", score=0.9, payload={"url": "https://a.test"})]
    client = FakeClient(query_points_result=SimpleNamespace(points=fake_points))
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    results = qdrant_store.semantic_search_pages(run_id="run-1", question="q", top_k=5)

    assert results == fake_points
    assert client.query_points_calls[0]["limit"] == 5


def test_semantic_search_pages_never_raises_on_qdrant_failure(monkeypatch):
    class BrokenClient(FakeClient):
        def query_points(self, **kwargs):
            raise RuntimeError("qdrant is down")

    monkeypatch.setattr(qdrant_store, "_client", BrokenClient())
    monkeypatch.setattr(qdrant_store, "embed_text", lambda text: [1.0, 0.0])

    results = qdrant_store.semantic_search_pages(run_id="run-1", question="q")

    assert results == []
