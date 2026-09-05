"""Simple per-domain courtesy rate limit for the fetch pipeline: don't
hammer the same site with back-to-back requests just because several
search results happened to land on it. Process-lifetime, in-memory,
thread-safe (the DAG executor runs handlers in a thread pool, so multiple
runs' fetches can be in flight concurrently).
"""

import threading
import time
from urllib.parse import urlparse

from agents.common.logging import get_logger

logger = get_logger(component="rate_limiter")

_last_request_at: dict[str, float] = {}
_lock = threading.Lock()
_MIN_GAP_SECONDS = 1.0


def throttle(url: str) -> None:
    domain = urlparse(url).netloc
    if not domain:
        return

    with _lock:
        last = _last_request_at.get(domain)
        now = time.monotonic()
        wait = (_MIN_GAP_SECONDS - (now - last)) if last is not None else 0.0
        _last_request_at[domain] = now + max(wait, 0.0)

    if wait > 0:
        logger.info("rate_limit_throttling", domain=domain, wait_seconds=round(wait, 2))
        time.sleep(wait)
