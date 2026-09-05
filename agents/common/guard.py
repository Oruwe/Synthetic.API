"""Prompt-injection guard: scans untrusted text extracted from the portal.

This is the SECOND layer of defense. The FIRST and more important one is
architectural: web_navigator/extractor.py pulls fields via specific
Playwright DOM selectors into a strict pydantic schema, so an LLM never
reads raw scraped HTML/free text in the first place. This module exists
for the fields that *are* free text by nature (delay_reason, raw_notes,
customer_name) — an attacker who controls portal content could still put
adversarial instructions inside those.

Hits are recorded, never silently stripped: `scan_for_injection` returns
what it found; callers decide what to do (extractor.py logs+flags,
synthesizer/drafter.py redacts flagged fields before they reach a prompt).
Silently sanitizing here would hide evidence from logs/judges and give a
false sense of security if a pattern is incomplete.
"""

import re

from pydantic import BaseModel


class GuardHit(BaseModel):
    pattern_name: str
    matched_text: str
    field_name: str


# Deliberately simple, explainable patterns over an ML classifier: for a
# hackathon threat-model demo, "here is exactly what we check for and why"
# is more defensible than an opaque model, and these are fast/dependency-free.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore_instructions", re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions", re.I)),
    ("disregard_prompt", re.compile(r"disregard\s+(the\s+)?(system|previous)\s+(prompt|instructions)", re.I)),
    ("role_marker", re.compile(r"(^|\n)\s*(system|assistant|user)\s*:", re.I)),
    ("chat_control_token", re.compile(r"(<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\])", re.I)),
    ("persona_override", re.compile(r"\byou are now\b|\back as\b", re.I)),
    ("prompt_leak_probe", re.compile(r"(reveal|print|show)\s+(your|the)\s+(system prompt|instructions)", re.I)),
]

_MATCH_PREVIEW_LEN = 80


def scan_for_injection(text: str | None, field_name: str) -> list[GuardHit]:
    if not text:
        return []
    hits: list[GuardHit] = []
    for pattern_name, pattern in PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(
                GuardHit(
                    pattern_name=pattern_name,
                    matched_text=match.group(0)[:_MATCH_PREVIEW_LEN],
                    field_name=field_name,
                )
            )
    return hits
