"""Tests for the Synthesizer's dual-kind watcher: it must dispatch shipping
("delayed") and web-research ("research") points separately, each tracked
against its own seen-set."""

from types import SimpleNamespace

from agents.synthesizer import watcher


def test_poll_once_tags_delayed_and_research_records_by_kind(monkeypatch):
    delayed_record = SimpleNamespace(payload={"point_key": "run1:ORD-1", "status": "delayed"})
    research_record = SimpleNamespace(payload={"point_key": "run2:https://x.test", "status": "permanent"})

    monkeypatch.setattr(
        "agents.common.qdrant_store.scroll_new_delayed", lambda seen: [delayed_record]
    )
    monkeypatch.setattr(
        "agents.common.qdrant_store.scroll_new_permanent_research", lambda seen: [research_record]
    )

    seen = {"delayed": set(), "research": set()}
    found = watcher.poll_once(seen)

    kinds = {kind for kind, _ in found}
    assert kinds == {"delayed", "research"}
    assert "run1:ORD-1" in seen["delayed"]
    assert "run2:https://x.test" in seen["research"]


def test_poll_once_does_not_re_notify_already_seen_points(monkeypatch):
    monkeypatch.setattr("agents.common.qdrant_store.scroll_new_delayed", lambda seen: [])
    monkeypatch.setattr("agents.common.qdrant_store.scroll_new_permanent_research", lambda seen: [])

    seen = {"delayed": {"run1:ORD-1"}, "research": set()}
    found = watcher.poll_once(seen)

    assert found == []


def test_load_and_save_seen_round_trip(tmp_path, monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "run_store_dir", str(tmp_path))
    watcher._save_seen({"delayed": {"a", "b"}, "research": {"c"}})

    reloaded = watcher._load_seen()
    assert reloaded == {"delayed": {"a", "b"}, "research": {"c"}}
