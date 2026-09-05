"""Re-exports the shared notifier. Moved to agents/common/notifier.py so
the Orchestrator can also use it for `no_capability` plans (see
orchestrator/executor.py) without a synthesizer -> orchestrator import
cycle. Kept here so `from agents.synthesizer import notifier` still works
for anything not yet updated."""

from agents.common.notifier import notify

__all__ = ["notify"]
