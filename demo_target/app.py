"""Safe, self-controlled demo/test target for the ambient RPA action path
AND the human-in-the-loop gated-content path.

This exists so the action executor (agents/web_navigator/action_executor.py)
always has a reversible, zero-risk site to act on -- for the hackathon demo
itself, and for validating the real vision model's decisions (as opposed to
the mechanical click/type/screenshot plumbing, which is tested against this
same fixture in tests/test_page_fetcher_live.py's style but for the action
path -- see the live-test scripts this app ships alongside).

Four flows on this fixture, deliberately:
- `/` -- a newsletter signup (email + Subscribe). The "should complete"
  case for the ambient RPA action path on its own.
- `/` -- a "Complete Purchase" button, deliberately payment-shaped so it
  exercises BOTH lines of defense live: the vision model's own
  self-refusal instruction, and action_executor.py's independent regex
  backstop, which must block the click before it ever reaches this
  page's backend (there's intentionally nothing behind this button for a
  real purchase to complete -- refusing to click it IS the test).
- `/article` -- an EMAIL-gated content page: a short teaser plus
  "Subscribe to continue reading" until an email is provided, then the
  real article text. This is what page_fetcher._detect_gate_phrase is
  meant to recognize and what page_handlers.handle_fetch_pages pauses a
  run for.
- `/members` -- a LOGIN-gated page (email + password, a throwaway demo
  account: see _DEMO_ACCOUNT below -- not connected to anything real,
  same as every other piece of state here). Exercises
  execute_login_and_extract's credential-safe flow: this fixture will
  happily show real member content when logged in, but it's the ONE
  place this whole project ever asks for a password, and even here
  nothing about the credential is remotely sensitive.

State is in-memory only, module-level, cleared via POST /reset -- never
a database, never persisted: this fixture's entire point is having
nothing real to corrupt. Same Flask/server-rendered-HTML convention as
mock_portal/app.py.
"""

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

_subscribers: list[str] = []
_is_logged_in: bool = False

# A throwaway demo account -- not a real credential for anything,
# printed here in the open on purpose. This is the one place in the
# entire project that solicits a password at all, and it's this: a
# fixture account with no real value behind it whatsoever.
_DEMO_ACCOUNT = {"email": "demo@example.com", "password": "demo123"}

# Deliberately short (well under page_fetcher._MIN_ACCEPTABLE_WORD_COUNT)
# and phrased to match page_fetcher._GATE_PHRASES -- this is the fixture
# half of that detector's contract, exercised for real by
# tests/test_page_fetcher_live.py-style scripts and the live-test scripts
# this app ships alongside.
_ARTICLE_TEASER = "Scientists announced a breakthrough today. Subscribe to continue reading the full story."
_ARTICLE_FULL = (
    "Scientists announced a breakthrough today: a new synthetic API bridge lets voice-driven "
    "agents operate legacy web portals that were never given a real API. The key insight -- "
    "coordinating multiple agents entirely through shared memory rather than direct calls -- "
    "turns out to generalize far beyond its original shipping-portal use case."
)
_MEMBERS_TEASER = "This briefing is for members only. Sign in to continue reading."
_MEMBERS_CONTENT = "Member briefing: quarterly roadmap review is scheduled for next week. Full agenda attached."


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.get("/")
def index():
    return render_template("index.html", subscribers=_subscribers)


@app.post("/subscribe")
def subscribe():
    email = (request.form.get("email") or "").strip()
    if email:
        _subscribers.append(email)
    return render_template("index.html", subscribers=_subscribers, just_subscribed=email or None)


@app.get("/article")
def article():
    unlocked = len(_subscribers) > 0
    return render_template(
        "article.html", unlocked=unlocked, teaser=_ARTICLE_TEASER, full_text=_ARTICLE_FULL
    )


@app.post("/article/subscribe")
def article_subscribe():
    email = (request.form.get("email") or "").strip()
    if email:
        _subscribers.append(email)
    return render_template("article.html", unlocked=len(_subscribers) > 0, teaser=_ARTICLE_TEASER, full_text=_ARTICLE_FULL)


@app.get("/members")
def members():
    return render_template(
        "members.html", logged_in=_is_logged_in, teaser=_MEMBERS_TEASER, content=_MEMBERS_CONTENT, error=None
    )


@app.post("/members/login")
def members_login():
    global _is_logged_in
    email = (request.form.get("email") or "").strip()
    password = request.form.get("password") or ""
    if email == _DEMO_ACCOUNT["email"] and password == _DEMO_ACCOUNT["password"]:
        _is_logged_in = True
        return render_template("members.html", logged_in=True, teaser=_MEMBERS_TEASER, content=_MEMBERS_CONTENT, error=None)
    return render_template(
        "members.html", logged_in=False, teaser=_MEMBERS_TEASER, content=_MEMBERS_CONTENT, error="Invalid email or password"
    ), 401


@app.post("/reset")
def reset():
    """Clears all state between demo/test runs -- the reversible half of
    "safe, reversible default demo target" (see README)."""
    global _is_logged_in
    _subscribers.clear()
    _is_logged_in = False
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
