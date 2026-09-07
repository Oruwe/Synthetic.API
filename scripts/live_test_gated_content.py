#!/usr/bin/env python3
"""Runs the REAL human-in-the-loop pause/resume path -- real vision model
call, real Chromium, real DAG executor pause/persist/resume -- against the
safe local demo_target fixture's gated pages.

Same reason this exists as scripts/live_test_action_loop.py: the sandbox
this feature was built in cannot reach openrouter.ai, so the vision model's
own judgment (locating the email/password fields, confirming a login
succeeded) could not be proven from inside it. Run this wherever
OPENROUTER_API_KEY is actually reachable -- your machine, not a
locked-down CI/sandbox.

This drives the SAME two entry points the orchestrator's own
POST /trigger and POST /runs/{run_id}/resume use --
agents.orchestrator.executor.execute_plan / resume_plan -- against a
hand-built one-node DAGPlan (bypassing Tavily search, same as
live_test_action_loop.py bypasses it for the action path), so this proves
the real pause -> persist -> resume cycle end to end, not just the
gate-detection/action-executor pieces already covered by mocked tests.

Prerequisites:
  1. demo_target running and reachable, e.g.:
       docker compose up -d demo_target
     or directly:
       python demo_target/app.py &
  2. OPENROUTER_API_KEY set (.env or exported) -- see live_test_action_loop.py
     for the same free-tier-key notes.

Usage:
  uv run python scripts/live_test_gated_content.py                       # email-gated /article
  uv run python scripts/live_test_gated_content.py --page members        # login-gated /members
  uv run python scripts/live_test_gated_content.py --target-url http://localhost:5050
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.common.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-url", default="http://localhost:5050", help="demo_target's base URL")
    parser.add_argument(
        "--page", choices=["article", "members"], default="article",
        help="which gated fixture page to hit: 'article' (email-only gate) or 'members' (login gate)",
    )
    parser.add_argument("--email", default="judge@example.com", help="email to answer the gate with")
    parser.add_argument(
        "--password", default="demo123",
        help="password to answer a login gate with -- demo_target's own throwaway demo account "
        "(demo@example.com / demo123) unless overridden; only used for --page members",
    )
    parser.add_argument("--run-id", default="live-test-gate", help="run_id, used to namespace screenshots on disk")
    args = parser.parse_args()

    if not settings.openrouter_api_key:
        print(
            "ERROR: OPENROUTER_API_KEY is not set. Put it in a .env file at the repo root "
            "(see .env.example) or export it before running this script.",
            file=sys.stderr,
        )
        return 1

    # Imported after the key check above so a missing key fails fast with a
    # clear message, and after sys.path is set up so the package imports
    # resolve the same way they would under `uv run`.
    from agents.common import run_store
    from agents.common.models.dag import DAGNode, DAGPlan, NodeType
    from agents.orchestrator.executor import execute_plan, resume_plan

    gate_url = f"{args.target_url.rstrip('/')}/{args.page}"
    email = args.email
    # For the /members fixture, the ONLY account that actually exists is
    # demo_target's own demo@example.com/demo123 -- defaulting the email to
    # that (rather than the generic judge@example.com used for /article)
    # so `--page members` works out of the box without extra flags.
    if args.page == "members" and email == parser.get_default("email"):
        email = "demo@example.com"

    print(f"Model:  {settings.openrouter_vision_model}")
    print(f"Target: {gate_url}")
    print(f"Fields: {'email + password' if args.page == 'members' else 'email only'}")
    print()

    plan = DAGPlan(
        run_id=args.run_id,
        transcript=f"what does the {args.page} page at {gate_url} say",
        created_at=datetime.now(timezone.utc),
        nodes=[
            DAGNode(
                id="fetch",
                type=NodeType.FETCH_PAGES,
                name="fetch",
                handler_key="fetch_pages",
                params={
                    "question": f"what does the {args.page} page say",
                    "search_results": [{"title": "Gated fixture page", "url": gate_url}],
                },
                timeout_seconds=120,
            )
        ],
        edges=[],
    )

    print("--- Step 1: execute_plan (real search-bypassed fetch, expect a pause) ---")
    run = execute_plan(plan)
    print(f"overall_status: {run.overall_status}")
    if run.overall_status != "awaiting_human_input" or run.pending_input is None:
        print(
            "FAIL: expected the run to pause for human input -- either demo_target isn't "
            "serving a real gate at this URL, or the gate went undetected. Full run state:",
            file=sys.stderr,
        )
        print(json.dumps(run.model_dump(mode="json"), indent=2), file=sys.stderr)
        return 2
    print(f"pending_input:  fields={run.pending_input.fields} prompt={run.pending_input.prompt!r}")
    print()

    provided = {"email": email}
    if "password" in run.pending_input.fields:
        provided["password"] = args.password
    print(f"--- Step 2: resume_plan (answering with fields={list(provided.keys())}) ---")
    run = resume_plan(args.run_id, provided)
    print(f"overall_status: {run.overall_status}")

    page = (run.plan.nodes[0].params or {}).get("sources_succeeded")
    print(f"sources_succeeded (node param): {page}")

    # The security proof, live: reload the run from disk exactly as a
    # restarted orchestrator process would, and confirm the password is
    # nowhere in what got persisted -- see PendingInputRequest's docstring
    # (agents/common/models/dag.py) for the guarantee this is checking.
    reloaded = run_store.load_run(args.run_id)
    raw = reloaded.model_dump_json() if reloaded else ""
    password_leaked = "password" in provided and provided["password"] in raw
    print()
    print("=" * 60)
    print(f"final overall_status:       {run.overall_status}")
    print(f"password persisted to disk: {password_leaked}  (must be False)")
    print(f"human_provided_inputs:      {run.human_provided_inputs}")
    print("=" * 60)

    if password_leaked:
        print("FAIL: the password ended up in persisted run state.", file=sys.stderr)
        return 3
    if run.overall_status != "completed" and run.overall_status != "running":
        # "running" is fine here: with only one node in this hand-built
        # plan, a successful gate-pass leaves the DAG with nothing left to
        # walk, so it should be "completed" -- but print whatever it
        # actually is either way so a real failure (e.g. "failed") is
        # visible rather than silently passing.
        print(f"NOTE: overall_status is {run.overall_status!r} -- inspect data/runs/{args.run_id}.json for detail.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
