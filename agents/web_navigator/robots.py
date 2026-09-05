"""robots.txt respect for the fetch pipeline (page_fetcher.py).

Fetched once per domain and cached for this process's lifetime via our
OWN httpx call, rather than letting stdlib's RobotFileParser.read() do it
-- that method has no configurable timeout and could hang indefinitely on
a slow or dead robots.txt endpoint, which would defeat the whole point of
the timeout discipline everywhere else in this module.

Fails open (allow) on any fetch/parse problem: a robots.txt outage or a
malformed file is the conventional "no robots.txt present" case, not a
reason to block an otherwise-legitimate fetch.
"""

import threading
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from agents.common.logging import get_logger

logger = get_logger(component="robots")

_cache: dict[str, RobotFileParser] = {}
_cache_lock = threading.Lock()
_USER_AGENT = "SyntheticAPI-Researcher/1.0"
_FETCH_TIMEOUT_SECONDS = 5.0


def _domain_root(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _get_parser(domain_root: str) -> RobotFileParser:
    with _cache_lock:
        cached = _cache.get(domain_root)
        if cached is not None:
            return cached

    parser = RobotFileParser()
    parser.set_url(f"{domain_root}/robots.txt")
    try:
        response = httpx.get(f"{domain_root}/robots.txt", timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True)
        if response.status_code == 200:
            parser.parse(response.text.splitlines())
        else:
            parser.parse([])  # no robots.txt (404 etc.) -- conventional "allow all"
    except Exception as exc:  # noqa: BLE001 - a robots.txt hiccup must not block the real fetch
        logger.info("robots_txt_fetch_failed_allowing_by_default", domain=domain_root, error=str(exc))
        parser.parse([])

    with _cache_lock:
        _cache[domain_root] = parser
    return parser


def is_allowed(url: str) -> bool:
    try:
        parser = _get_parser(_domain_root(url))
        return parser.can_fetch(_USER_AGENT, url)
    except Exception as exc:  # noqa: BLE001 - never let a parsing quirk block a legitimate fetch
        logger.info("robots_txt_check_failed_allowing_by_default", url=url, error=str(exc))
        return True
