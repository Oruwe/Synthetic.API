"""Tests for qdrant_store.scroll_new_delayed's pagination.

Uses a fake client (no real Qdrant needed) so this stays in the offline,
dependency-free test suite. Regression test: the previous implementation
scrolled a single unpaginated page, so a new point could be silently
missed once the collection held more delayed points than that page size.
"""

from types import SimpleNamespace

import agents.common.qdrant_store as qdrant_store


class _FakeCollections:
    def __init__(self, names):
        self.collections = [SimpleNamespace(name=n) for n in names]


class FakeClient:
    """Serves `pages` (a list of (records, next_offset) tuples) one per
    call to `.scroll()`, in order -- simulates a multi-page Qdrant scroll."""

    def __init__(self, pages):
        self._pages = list(pages)
        self._collection_names = {"delayed_orders"}
        self.scroll_calls = []

    def get_collections(self):
        return _FakeCollections(self._collection_names)

    def create_collection(self, **kwargs):
        self._collection_names.add(kwargs["collection_name"])

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return self._pages.pop(0)


def _record(point_key, order_id="ORD-X"):
    return SimpleNamespace(payload={"point_key": point_key, "order_id": order_id, "status": "delayed"})


def test_scroll_new_delayed_single_page(monkeypatch):
    client = FakeClient(pages=[([_record("run1:ORD-1"), _record("run1:ORD-2")], None)])
    monkeypatch.setattr(qdrant_store, "_client", client)

    new_records = qdrant_store.scroll_new_delayed(seen_point_ids=set())

    assert len(new_records) == 2
    assert len(client.scroll_calls) == 1


def test_scroll_new_delayed_follows_pagination_cursor(monkeypatch):
    """Regression test: past the first page, new points must still be found."""
    page1 = ([_record(f"run1:ORD-{i}") for i in range(100)], "cursor-1")
    page2 = ([_record("run1:ORD-100")], None)  # the "new" point, on page 2
    client = FakeClient(pages=[page1, page2])
    monkeypatch.setattr(qdrant_store, "_client", client)

    seen = {f"run1:ORD-{i}" for i in range(100)}  # everything on page 1 already seen
    new_records = qdrant_store.scroll_new_delayed(seen_point_ids=seen)

    assert len(client.scroll_calls) == 2
    assert [r.payload["point_key"] for r in new_records] == ["run1:ORD-100"]
    # second call must have used the cursor returned by the first
    assert client.scroll_calls[1]["offset"] == "cursor-1"


def test_scroll_new_delayed_stops_when_no_next_offset(monkeypatch):
    client = FakeClient(pages=[([], None)])
    monkeypatch.setattr(qdrant_store, "_client", client)

    new_records = qdrant_store.scroll_new_delayed(seen_point_ids=set())

    assert new_records == []
    assert len(client.scroll_calls) == 1
