#!/usr/bin/env python3
"""Runs the REAL ambient RPA action loop -- real vision model call, real
Chromium, real clicks -- against the safe local demo_target fixture.

This is the one thing that could NOT be validated from within the sandbox
this feature was built in: that sandbox's network policy blocks
openrouter.ai entirely (confirmed via a direct connectivity check, not
assumed), so every other piece of this system was proven for real except
the vision model's own judgment. Run this wherever OPENROUTER_API_KEY is
actually reachable -- your machine, not a locked-down CI/sandbox.

Prerequisites:
  1. demo_target running and reachable, e.g.:
       docker compose up -d demo_target
     or directly:
       python demo_target/app.py &
  2. OPENROUTER_API_KEY set (.env or exported) -- a free-tier key is fine;
     override OPENROUTER_VISION_MODEL if the default isn't available on
     the free tier for your account (check https://openrouter.ai/models,
     filter to vision-capable, sort by price -- many have a `:free`
     variant, though not always the same one over time).

Usage:
  uv run python scripts/live_test_action_loop.py
  uv run python scripts/live_test_action_loop.py --target-url http://localhost:5050
  uv run python scripts/live_test_action_loop.py --intent "unlock Northwind Weekly Pro"  # exercises the payment guard
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.common.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-url", default="http://localhost:5050", help="demo_target's base URL")
    parser.add_argument(
        "--intent",
        default="subscribe to the newsletter",
        help='the task to attempt. Try "unlock Northwind Weekly Pro" to exercise the payment guard instead.',
    )
    parser.add_argument("--run-id", default="live-test", help="run_id, used to namespace screenshots on disk")
    args = parser.parse_args()

    if not settings.openrouter_api_key:
        print(
            "ERROR: OPENROUTER_API_KEY is not set. Put it in a .env file at the repo root "
            "(see .env.example) or export it before running this script.",
            file=sys.stderr,
        )
        return 1

    # Imported after the key check above so a missing key fails fast with
    # a clear message instead of a confusing error from deep inside the
    # OpenAI client construction.
    from agents.web_navigator import action_executor

    print(f"Model: {settings.openrouter_vision_model}")
    print(f"Target: {args.target_url}")
    print(f"Intent: {args.intent!r}")
    print(f"Screenshots: {settings.screenshot_dir}/{args.run_id}/action/")
    print()

    workflow = action_executor.execute_action_loop(
        intent=args.intent,
        start_url=args.target_url,
        run_id=args.run_id,
    )

    print()
    print("=" * 60)
    print(f"success:        {workflow.success}")
    print(f"refused_reason: {workflow.refused_reason}")
    print(f"steps taken:    {len(workflow.steps)}")
    for i, step in enumerate(workflow.steps):
        coords = f"({step.x},{step.y})" if step.x is not None else ""
        text = f' text="{step.text}"' if step.text else ""
        print(f"  {i}. {step.kind}{coords}{text} -- {step.reasoning}")
    print("=" * 60)
    print()
    print("Full workflow JSON:")
    print(json.dumps(workflow.model_dump(mode="json"), indent=2))

    return 0 if workflow.success or workflow.refused_reason else 2


if __name__ == "__main__":
    raise SystemExit(main())
