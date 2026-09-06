"""Tests for the cross-process half of the ambient RPA memory layer's
concurrency story: _distributed_lock_for_workflow (agents/common/
qdrant_store.py). The in-process lock it falls back to (_lock_for_workflow)
already has its own concurrency proof in test_action_workflow_store.py --
this file covers what's new here: Redis-backed mutual exclusion across
real threads, and every degrade-instead-of-block path (unset, unreachable,
contended-and-timed-out).

A fake Redis client stands in for the real one -- no real Redis needed to
develop against or verify this offline, matching every other external
dependency in this codebase's test suite. The fake's `.eval` encodes the
SAME check-token-then-delete logic as `_REDIS_RELEASE_LOCK_SCRIPT`
(verified by inspection against that script's text, not by running real
Lua -- there's no Lua interpreter in this test environment, same honest
scope limit as this repo's own "docker compose has never run inside this
sandbox" note elsewhere).
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import agents.common.qdrant_store as qdrant_store


class _FakeRedis:
    """Minimal double for the subset of the redis-py sync client
    _distributed_lock_for_workflow actually uses. Its own internal lock
    makes `.set` atomic under real thread contention -- the fake needs to
    be correct for the concurrency proof below to mean anything."""

    def __init__(self):
        self._lock = threading.Lock()
        self._store: dict[str, tuple[str, float]] = {}  # key -> (token, expires_at_monotonic)
        self.set_calls = 0
        self.eval_calls = 0

    def _is_expired(self, key: str) -> bool:
        entry = self._store.get(key)
        return entry is None or time.monotonic() >= entry[1]

    def set(self, key, value, nx=False, px=None):
        self.set_calls += 1
        with self._lock:
            if nx and key in self._store and not self._is_expired(key):
                return None
            self._store[key] = (value, time.monotonic() + (px or 10_000) / 1000)
            return True

    def eval(self, script, numkeys, key, token):
        """Mirrors _REDIS_RELEASE_LOCK_SCRIPT: only deletes if the stored
        token still matches this caller's -- a caller whose lock already
        expired and was re-acquired by someone else must not delete it."""
        self.eval_calls += 1
        with self._lock:
            entry = self._store.get(key)
            if entry is not None and entry[0] == token:
                del self._store[key]
                return 1
            return 0


class _UnreachableFakeRedis(_FakeRedis):
    def set(self, *a, **k):
        raise ConnectionError("redis unreachable")


def _use_redis(monkeypatch, client):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "redis_url", "redis://fake-for-tests")
    monkeypatch.setattr(qdrant_store, "_get_redis_client", lambda: client)


def test_redis_url_unset_falls_through_to_the_in_process_lock(monkeypatch):
    from agents.common.config import settings

    monkeypatch.setattr(settings, "redis_url", "")
    entered = {"value": False}

    with qdrant_store._distributed_lock_for_workflow("point-1"):
        entered["value"] = True

    assert entered["value"] is True  # the in-process fallback still lets the caller through


def test_acquires_and_releases_with_a_matching_token(monkeypatch):
    fake = _FakeRedis()
    _use_redis(monkeypatch, fake)

    with qdrant_store._distributed_lock_for_workflow("point-2"):
        assert "workflow_lock:point-2" in fake._store  # held during the critical section

    assert "workflow_lock:point-2" not in fake._store  # released on exit
    assert fake.eval_calls == 1


def test_falls_back_to_in_process_lock_when_redis_is_unreachable(monkeypatch):
    _use_redis(monkeypatch, _UnreachableFakeRedis())
    entered = {"value": False}

    with qdrant_store._distributed_lock_for_workflow("point-3"):
        entered["value"] = True  # must still get here -- a Redis outage must not block the write

    assert entered["value"] is True


def test_proceeds_without_a_lock_when_acquisition_times_out(monkeypatch):
    """A real browser action already happened; refusing to ever record it
    because a lock is contended is worse than a rare, logged, unprotected
    write -- so this must proceed, not raise or hang."""
    from agents.common.config import settings

    fake = _FakeRedis()
    fake._store["workflow_lock:point-4"] = ("someone-elses-token", time.monotonic() + 60)  # held, not expiring soon
    _use_redis(monkeypatch, fake)
    monkeypatch.setattr(settings, "redis_lock_acquire_timeout_seconds", 0.05)

    entered = {"value": False}
    with qdrant_store._distributed_lock_for_workflow("point-4"):
        entered["value"] = True

    assert entered["value"] is True


def test_release_never_deletes_a_lock_it_no_longer_holds(monkeypatch):
    """The safety property the Lua script exists for: if this holder's
    lock already expired and someone else acquired the key, releasing
    must be a no-op, never a delete of the new holder's lock."""
    fake = _FakeRedis()
    fake._store["workflow_lock:point-5"] = ("a-different-holders-token", time.monotonic() + 60)

    result = fake.eval(qdrant_store._REDIS_RELEASE_LOCK_SCRIPT, 1, "workflow_lock:point-5", "my-token")

    assert result == 0
    assert fake._store["workflow_lock:point-5"][0] == "a-different-holders-token"  # untouched


def test_distributed_lock_serializes_real_concurrent_holders(monkeypatch):
    """The concurrency proof: N real threads each do a non-atomic
    read-sleep-write on a shared counter while holding the lock. Without
    correct mutual exclusion this reliably loses increments (two threads
    both read the same value, both write back the same +1); with it, the
    final count is exactly N every run -- the same proof style as
    test_action_workflow_store.py's in-process-lock test, one level down
    the stack."""
    fake = _FakeRedis()
    _use_redis(monkeypatch, fake)
    counter = {"value": 0}
    n = 20

    def critical_section(_):
        with qdrant_store._distributed_lock_for_workflow("shared-point"):
            current = counter["value"]
            time.sleep(0.005)
            counter["value"] = current + 1

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(critical_section, range(n)))

    assert counter["value"] == n
    assert fake.set_calls >= n  # at least one SET attempt per holder (more if any had to retry)
