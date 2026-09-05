"""Omi voice transcript ingestion.

TODO(verify against hackathon starter kit): Omi's real webhook payload
shape and auth header are NOT confirmed here from first-hand docs. This
parser is intentionally liberal — it accepts a couple of plausible shapes
(a flat `transcript` string, or a list of `segments` with `text` fields,
which is how several voice-capture webhook APIs, Omi included in publicly
documented examples, deliver a session's utterances) so the Orchestrator
keeps working once real payloads start arriving; adjust `parse_omi_payload`
to match the confirmed schema and nothing else needs to change.
"""

import hmac

from fastapi import HTTPException

from agents.common.config import settings


class OmiPayloadError(ValueError):
    pass


def verify_webhook_secret(provided_secret: str | None) -> None:
    """Best-effort shared-secret check. No-ops if OMI_WEBHOOK_SECRET isn't
    configured (e.g. local demo) rather than locking the operator out."""
    if not settings.omi_webhook_secret:
        return
    if not provided_secret or not hmac.compare_digest(provided_secret, settings.omi_webhook_secret):
        raise HTTPException(status_code=401, detail="invalid or missing webhook secret")


def parse_omi_payload(payload: dict) -> str:
    if "transcript" in payload and isinstance(payload["transcript"], str):
        return payload["transcript"].strip()

    segments = payload.get("segments") or payload.get("transcript_segments")
    if isinstance(segments, list) and segments:
        texts = [seg.get("text", "") for seg in segments if isinstance(seg, dict)]
        joined = " ".join(t.strip() for t in texts if t.strip())
        if joined:
            return joined

    memory = payload.get("memory") or payload.get("structured")
    if isinstance(memory, dict):
        text = memory.get("transcript") or memory.get("overview")
        if isinstance(text, str) and text.strip():
            return text.strip()

    raise OmiPayloadError(
        "could not extract a transcript from the Omi payload — "
        "verify the payload shape against the starter kit and update parse_omi_payload()"
    )
