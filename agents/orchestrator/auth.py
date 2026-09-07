"""Shared-secret auth for the Orchestrator's own API (/trigger, /runs/*).

Deliberately the same shape as omi_webhook.verify_webhook_secret --
constant-time comparison via hmac.compare_digest (a naive `==` on a
secret leaks timing information an attacker can use to guess it one byte
at a time), and the SAME fail-open-if-unconfigured posture: unset
ORCHESTRATOR_API_KEY (the default, e.g. local `docker compose up`) means
this simply doesn't gate anything, preserving the local-dev convenience
every other script/curl example in this repo already relies on.

This exists because /trigger, GET /runs/{run_id}, and POST
/runs/{run_id}/resume have NO authentication at all otherwise -- fine on
localhost, not fine on a VM with a public IP: anyone who can reach the
port can trigger arbitrary browser actions (the ambient RPA action path
included), read any run's full state, or answer any paused run's gate
prompt. See README's deployment section for why this MUST be set before
exposing the Orchestrator to the open internet, and why /health and
/ready deliberately stay ungated (a load balancer/uptime check needs to
reach those without a credential).
"""

import hmac

from fastapi import Header, HTTPException

from agents.common.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency -- wire with `Depends(require_api_key)` on any
    route that must not be open to the public internet unauthenticated.
    No-ops if ORCHESTRATOR_API_KEY isn't configured."""
    if not settings.orchestrator_api_key:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.orchestrator_api_key):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
