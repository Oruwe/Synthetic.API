"""Tests for agents/common/config.py's startup business-rule validators --
added on top of pydantic's own type coercion (which already ran eagerly at
import time via `settings = Settings()`) to catch real, likely
misconfigurations rather than just type mismatches.

Constructs a fresh `Settings()` per test (never mutates the shared
`settings` singleton) so these are fully isolated from each other and from
the rest of the suite, which does monkeypatch fields on that singleton.
"""

import warnings

import pytest
from pydantic import ValidationError

from agents.common.config import Settings

# --- Lyzr consistency (warns, does not raise -- fail-open, matching
# lyzr_wrapper.py's own per-call fallback behavior) ------------------------


def test_warns_when_lyzr_enabled_without_an_agent_id():
    with pytest.warns(UserWarning, match="LYZR_AGENT_ID"):
        Settings(lyzr_enabled=True, lyzr_agent_id="", lyzr_api_key="real-key")


def test_warns_when_lyzr_enabled_without_an_api_key():
    with pytest.warns(UserWarning, match="LYZR_API_KEY"):
        Settings(lyzr_enabled=True, lyzr_agent_id="real-agent", lyzr_api_key="")


def test_no_warning_when_lyzr_enabled_and_fully_configured():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        Settings(lyzr_enabled=True, lyzr_agent_id="real-agent", lyzr_api_key="real-key")


def test_no_warning_when_lyzr_disabled_regardless_of_missing_fields():
    """The default, unconfigured state -- must stay silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Settings(lyzr_enabled=False, lyzr_agent_id="", lyzr_api_key="")


# --- URL fields -------------------------------------------------------------


def test_rejects_a_url_missing_its_scheme():
    with pytest.raises(ValidationError, match="qdrant_url"):
        Settings(qdrant_url="qdrant:6333")  # missing http://


def test_accepts_a_well_formed_https_url():
    s = Settings(qdrant_url="https://my-cluster.cloud.qdrant.io:6333")
    assert s.qdrant_url == "https://my-cluster.cloud.qdrant.io:6333"


def test_blank_url_field_is_allowed_not_rejected():
    """A blank value is never what this validator is trying to catch --
    only an obviously-wrong non-empty one (see the validator's own
    docstring: these fields all have non-empty defaults, so this only
    matters if something explicitly overrides one to blank)."""
    s = Settings(langfuse_host="")
    assert s.langfuse_host == ""


# --- Positive-int fields -----------------------------------------------------


def test_rejects_a_zero_action_max_steps():
    with pytest.raises(ValidationError, match="action_max_steps"):
        Settings(action_max_steps=0)


def test_rejects_a_negative_research_top_k():
    with pytest.raises(ValidationError, match="research_top_k"):
        Settings(research_top_k=-1)


def test_accepts_a_positive_dag_circuit_breaker_threshold():
    s = Settings(dag_circuit_breaker_threshold=3)
    assert s.dag_circuit_breaker_threshold == 3


# --- Score fields (must be a valid [0, 1] similarity/ratio) -----------------


def test_rejects_a_replay_min_score_above_one():
    with pytest.raises(ValidationError, match="action_workflow_replay_min_score"):
        Settings(action_workflow_replay_min_score=1.5)


def test_rejects_a_negative_trust_ratio():
    with pytest.raises(ValidationError, match="action_workflow_min_trust_ratio"):
        Settings(action_workflow_min_trust_ratio=-0.1)


def test_accepts_boundary_score_values():
    s = Settings(action_workflow_replay_min_score=0.0, action_workflow_min_trust_ratio=1.0)
    assert s.action_workflow_replay_min_score == 0.0
    assert s.action_workflow_min_trust_ratio == 1.0
