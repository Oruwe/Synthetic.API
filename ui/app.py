"""Demo UI for Synthetic.API.

Purely additive: this only calls the Orchestrator's existing HTTP API
(/trigger, /runs/{id}, /runs/{id}/resume) over the network -- no direct
access to Qdrant, run_store, or anything else -- so it carries zero risk to
the pipeline that's already proven to work. It exists because curl/bash
scripts are fine for development but not a great surface for a judge or a
live demo audience.

Deliberately NOT Gradio's default theme/layout -- a custom dark theme
(SyntheticTheme below) plus hand-written CSS keyed on elem_id/elem_classes
(never on Gradio's own internal implementation class names, which differ
across Gradio versions -- this file only ever styles hooks it declared
itself, so it can't silently break on a Gradio upgrade). Verified against
a real Gradio 5.50.0 install (matching ui/requirements.txt's `<6` pin)
before shipping: the Blocks graph constructs, the app launches, and the
root page actually serves valid HTML containing this file's own CSS/JS,
not just "the Python didn't raise an exception."

Voice, honestly:
- STT (speech-to-text) isn't something this UI needs to do. In the real
  deployment, Omi's own wearable/app does the transcription and POSTs the
  resulting transcript straight to /webhook/omi. This UI's text box is
  the same "already-transcribed question" input, just typed instead of
  spoken -- exactly like scripts/send_sample_transcript.sh already stands
  in for it on the CLI.
- TTS (voice out): rather than guess at an unconfirmed Omi voice-response
  API this close to a deadline, "read aloud" here uses the browser's own
  native SpeechSynthesis API client-side -- zero backend, zero new
  dependency, and it actually works today. See README's "Voice & UI"
  section for the reasoning.

Human-in-the-loop (feature/ambient-rpa-action-bridge): a run can come back
with overall_status "awaiting_human_input" when the Web-Researcher hits a
login/subscribe wall on the only source that answers the question (see
agents/common/models/dag.py's PendingInputRequest docstring for the full
security story). This UI surfaces that as an inline email/password prompt
and POSTs the answer to /runs/{run_id}/resume -- it never stores, logs, or
does anything with the password beyond that one POST; Gradio's own state
(`run_id_state`) carries only the run_id across the pause, never the
credential.
"""

import os
import time

import gradio as gr
import requests

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
# Sent as X-API-Key on every Orchestrator request when set -- matches
# ORCHESTRATOR_API_KEY there (see agents/orchestrator/auth.py). Both
# unset by default (local dev, unauthenticated); a real deployment sets
# both to the same value, or every request here gets a 401.
_API_KEY = os.environ.get("ORCHESTRATOR_API_KEY", "")
_AUTH_HEADERS = {"X-API-Key": _API_KEY} if _API_KEY else {}
_POLL_INTERVAL_SECONDS = 3.0
# Generous ceiling: embed_pages' own node timeout is 180s (see
# orchestrator/planner.py), so this needs enough headroom above that plus
# fetch + the drafting LLM call itself for a real, non-trivial question.
_MAX_WAIT_SECONDS = 300

# Reused wherever a yield needs to hide the human-input group and leave its
# fields untouched -- most polling ticks do exactly this, so spelling it out
# every time would bury the one branch that actually matters (the gate
# itself) in repetition.
_GATE_HIDDEN = gr.update(visible=False)


def _format_sources_markdown(sources: list[dict], sources_attempted, sources_succeeded) -> str:
    """Renders the structured `sources` list (url/title/snippet/score) as
    clickable markdown links instead of the old flattened "Sources used:
    url1, url2" string -- see RunState's field comments (dag.py) for why
    both still exist side by side."""
    lines = []
    if sources:
        lines.append("**Sources**")
        for s in sources:
            title = s.get("title") or s.get("url") or ""
            url = s.get("url") or ""
            score = s.get("score")
            score_str = f" _(relevance {score:.2f})_" if isinstance(score, (int, float)) else ""
            # Escape `[`/`]` in the title and wrap the URL in `<...>` (a
            # CommonMark "explicit" link destination): titles or URLs with
            # literal parens/brackets are common in the wild (e.g. Wikipedia
            # disambiguation pages like ".../Gaganyaan_(spacecraft)") and
            # would otherwise close the markdown link early, truncating it.
            safe_title = title.replace("[", "\\[").replace("]", "\\]")
            lines.append(f"- [{safe_title}](<{url}>){score_str}")
    if sources_attempted and sources_succeeded is not None and sources_succeeded < sources_attempted:
        lines.append(f"\n_Partial results: {sources_succeeded}/{sources_attempted} candidate sources were retrievable._")
    return "\n".join(lines)


def _poll_until_done_or_gate(run_id: str, waited: float = 0.0):
    """Shared polling loop used by both ask() and resume_gate() -- picking
    up an in-flight run and following it to one of three outcomes: a real
    answer, a timeout, or a pause for human input. Each yield is the full
    8-output tuple the Blocks wiring below expects (see its outputs= list):
    (status_md, answer, sources_md, gate_group_visible, gate_prompt,
    gate_email_update, gate_password_update, run_id).

    Split out from ask() once resume_gate() needed to keep polling the SAME
    run past a pause -- duplicating this loop would have meant two places
    to keep in sync on every future change to how a run's status is read.
    """
    dag_finished_status: str | None = None
    while waited < _MAX_WAIT_SECONDS:
        time.sleep(_POLL_INTERVAL_SECONDS)
        waited += _POLL_INTERVAL_SECONDS
        try:
            run_resp = requests.get(f"{ORCHESTRATOR_URL}/runs/{run_id}", headers=_AUTH_HEADERS, timeout=10)
            run_resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - show the real error, don't crash the UI
            yield f"⚠️ Lost contact polling the run: {exc}", "", "", _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, run_id
            return

        run = run_resp.json()
        status = run.get("overall_status")
        # answer_text (no footer) is what should be displayed/read aloud;
        # fall back to the older flattened `answer` field for runs
        # persisted before this field existed.
        answer_text = run.get("answer_text") or run.get("answer")
        if answer_text:
            sources_md = _format_sources_markdown(
                run.get("sources") or [], run.get("sources_attempted"), run.get("sources_succeeded")
            )
            yield (
                f"● **Done** · `{status}` · {waited:.0f}s elapsed", answer_text, sources_md,
                _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, run_id,
            )
            return

        if status == "awaiting_human_input":
            # The run has stopped itself and persisted -- there is no live
            # thread to keep polling here, so this generator simply ends;
            # gate_continue_btn.click(resume_gate, ...) below is what wakes
            # the run back up when the human answers.
            pending = run.get("pending_input") or {}
            fields = pending.get("fields") or []
            prompt = pending.get("prompt") or "This source needs more information to continue."
            yield (
                f"● **Paused** — run `{run_id}` needs your input to get past a login/subscribe wall.",
                "", "",
                gr.update(visible=True), prompt,
                gr.update(visible=True), gr.update(visible="password" in fields, value=""),
                run_id,
            )
            return

        if dag_finished_status is None and status in ("completed", "failed", "circuit_broken", "no_capability"):
            dag_finished_status = status

        if dag_finished_status is not None:
            yield (
                f"● Search/fetch/embed finished · `{dag_finished_status}` · waiting for the "
                f"Synthesizer to draft the answer... ({waited:.0f}s elapsed)",
                "", "", _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, run_id,
            )
        else:
            yield f"● Working... `{status}` · {waited:.0f}s elapsed", "", "", _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, run_id

    yield (
        f"⚠️ Timed out after {_MAX_WAIT_SECONDS}s waiting for run `{run_id}`'s answer — "
        f"check `docker compose logs agents-synthesizer` or poll `/runs/{run_id}` directly.",
        "", "", _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, run_id,
    )


def ask(question: str):
    question = (question or "").strip()
    if not question:
        yield "Type a question first.", "", "", _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, None
        return

    yield "● Sending your question to the Orchestrator...", "", "", _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, None

    try:
        resp = requests.post(f"{ORCHESTRATOR_URL}/trigger", json={"transcript": question}, headers=_AUTH_HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - show the real error, don't crash the UI
        yield f"⚠️ Could not reach the Orchestrator at {ORCHESTRATOR_URL}: {exc}", "", "", _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, None
        return

    body = resp.json()
    run_id = body.get("run_id")
    if not run_id:
        yield f"⚠️ Unexpected response from Orchestrator: {body}", "", "", _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, None
        return

    yield (
        f"● Run `{run_id}` started — searching the web, fetching pages, and embedding...",
        "", "", _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, run_id,
    )

    # The DAG (fetch -> embed) finishing and the Synthesizer actually
    # drafting an answer are two SEPARATE, asynchronous steps: the
    # Orchestrator marks overall_status "completed" the instant the DAG
    # itself is done, but the Synthesizer only notices and starts drafting
    # on its own next poll cycle (every ~5s, see watcher.py) and then the
    # LLM call itself still takes real time. Caught live: this loop used
    # to stop the moment overall_status went terminal and report "no
    # answer was recorded" if the Synthesizer simply hadn't caught up yet
    # -- not a real failure, just polling for the wrong signal. It now
    # keeps polling for the answer specifically (or a pause) via the
    # shared helper above, using the DAG's terminal status only to change
    # the status message, not to stop early.
    yield from _poll_until_done_or_gate(run_id)


def resume_gate(run_id: str | None, email: str, password: str):
    """Wired to gate_continue_btn.click(). Sends whatever the human typed
    straight to POST /runs/{run_id}/resume and nowhere else -- this
    function's own local `password` variable goes out of scope the moment
    it returns, and the field is cleared in every yield below so a
    plaintext password never lingers in the page's rendered state past the
    one request that needed it."""
    if not run_id:
        yield (
            "⚠️ No paused run to resume — ask a question first.", "", "",
            _GATE_HIDDEN, "", _GATE_HIDDEN, _GATE_HIDDEN, run_id,
        )
        return

    payload = {}
    if email and email.strip():
        payload["email"] = email.strip()
    if password:
        payload["password"] = password

    try:
        resp = requests.post(f"{ORCHESTRATOR_URL}/runs/{run_id}/resume", json=payload, headers=_AUTH_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
        yield (
            f"⚠️ Could not resume run `{run_id}`: {detail}", "", "",
            gr.update(visible=True), "", gr.update(visible=True), gr.update(value=""), run_id,
        )
        return
    except Exception as exc:  # noqa: BLE001 - show the real error, don't crash the UI
        yield (
            f"⚠️ Could not reach the Orchestrator at {ORCHESTRATOR_URL}: {exc}", "", "",
            gr.update(visible=True), "", gr.update(visible=True), gr.update(value=""), run_id,
        )
        return

    yield f"● Continuing run `{run_id}`...", "", "", _GATE_HIDDEN, "", gr.update(value=""), gr.update(value=""), run_id
    yield from _poll_until_done_or_gate(run_id)


_READ_ALOUD_JS = """
(text) => {
    if (!text) { return; }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
}
"""

# --- Theme + CSS -------------------------------------------------------
#
# Built on gr.themes.Base (the blank slate, not Soft/Default) using only
# the stable, documented top-level constructor kwargs (hue names, fonts) --
# never the deeper .set(variable_name=...) overrides, whose exact variable
# names have shifted between Gradio major versions. The actual dark,
# "synthetic circuit" look comes entirely from plain CSS below, keyed on
# elem_id/elem_classes this file assigns itself -- a hook this file
# controls end to end, not a guess about Gradio's internal DOM/class
# names, so a future Gradio upgrade can't silently break the look.


class SyntheticTheme(gr.themes.Base):
    def __init__(self):
        super().__init__(
            primary_hue="teal",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        )


_CSS = """
:root {
    --synth-bg: #0a0e14;
    --synth-panel: #11161f;
    --synth-panel-2: #161c27;
    --synth-border: #232b38;
    --synth-text: #e6edf3;
    --synth-text-dim: #8b96a5;
    --synth-accent: #2dd4bf;
    --synth-accent-dim: #0f766e;
    --synth-warn: #f2b84b;
}

.gradio-container {
    background: var(--synth-bg) !important;
    color: var(--synth-text) !important;
    max-width: 880px !important;
    margin: 0 auto !important;
}

#synth-header {
    padding: 28px 4px 4px 4px;
    border-bottom: none;
}
#synth-header h1 {
    font-family: var(--font-mono, monospace);
    letter-spacing: 0.04em;
    font-size: 1.6rem;
    margin: 0 0 6px 0;
    background: linear-gradient(90deg, var(--synth-accent), #7dd3fc);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
#synth-header p {
    color: var(--synth-text-dim);
    font-size: 0.92rem;
    margin: 0 0 4px 0;
}
#synth-badges {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
}
#synth-badges span {
    font-family: var(--font-mono, monospace);
    font-size: 0.72rem;
    color: var(--synth-accent);
    border: 1px solid var(--synth-accent-dim);
    background: rgba(45, 212, 191, 0.08);
    border-radius: 999px;
    padding: 3px 10px;
}

.synth-card {
    background: var(--synth-panel) !important;
    border: 1px solid var(--synth-border) !important;
    border-radius: 12px !important;
    padding: 18px !important;
}

#synth-ask-btn {
    background: var(--synth-accent) !important;
    color: #06110f !important;
    font-weight: 600 !important;
    border: none !important;
}
#synth-ask-btn:hover {
    background: #5eead4 !important;
}

#synth-status {
    font-family: var(--font-mono, monospace) !important;
    font-size: 0.86rem !important;
    color: var(--synth-text-dim) !important;
    background: var(--synth-panel-2) !important;
    border-left: 3px solid var(--synth-accent) !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    min-height: 1.4em;
}
#synth-status p { margin: 0 !important; color: var(--synth-text-dim) !important; }
#synth-status strong { color: var(--synth-text) !important; }

#synth-gate {
    border: 1px solid var(--synth-warn) !important;
    background: rgba(242, 184, 75, 0.06) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
#synth-gate-prompt p { color: var(--synth-text) !important; font-weight: 500; }

#synth-answer-label label span {
    color: var(--synth-text-dim) !important;
    font-family: var(--font-mono, monospace) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
#synth-answer textarea {
    background: var(--synth-panel-2) !important;
    color: var(--synth-text) !important;
    border: 1px solid var(--synth-border) !important;
    font-size: 0.98rem !important;
    line-height: 1.55 !important;
}

#synth-sources { font-size: 0.88rem !important; color: var(--synth-text-dim) !important; }
#synth-sources a { color: var(--synth-accent) !important; }

#synth-footer {
    text-align: center;
    color: var(--synth-text-dim);
    font-size: 0.78rem;
    padding: 18px 0 8px 0;
}
"""

with gr.Blocks(title="Synthetic.API", theme=SyntheticTheme(), css=_CSS) as demo:
    with gr.Column(elem_id="synth-header"):
        gr.Markdown("# SYNTHETIC.API")
        gr.Markdown(
            "The API-less bridge — a voice transcript in, a multi-agent research or ambient-RPA "
            "workflow out. Type a question as if it were a transcript Omi already produced from "
            "your voice; in the real deployment, Omi's wearable does the speech-to-text itself and "
            "POSTs the transcript to `/webhook/omi` directly."
        )
        gr.HTML(
            '<div id="synth-badges"><span>Lyzr</span><span>Qdrant</span><span>Omi</span>'
            '<span>Multi-Agent Swarm</span></div>'
        )

    with gr.Column(elem_classes=["synth-card"]):
        question_box = gr.Textbox(
            label="Your question",
            placeholder="What is the current status of ISRO's Gaganyaan mission?",
            show_label=True,
        )
        ask_button = gr.Button("Ask", variant="primary", elem_id="synth-ask-btn")
        status_box = gr.Markdown(elem_id="synth-status")

        # Hidden until a run actually pauses on a gated source. See this
        # file's module docstring and PendingInputRequest (dag.py) for why
        # the password field only ever leaves the browser in the one POST
        # below.
        with gr.Group(visible=False, elem_id="synth-gate") as human_input_group:
            gate_prompt_md = gr.Markdown(elem_id="synth-gate-prompt")
            gate_email_box = gr.Textbox(label="Email", placeholder="you@example.com")
            gate_password_box = gr.Textbox(label="Password", type="password", visible=False)
            gate_continue_btn = gr.Button("Continue", variant="primary")

        run_id_state = gr.State(value=None)

        # Only the clean answer text lives here now -- no "Sources used:
        # ..." footer mixed in, so "Read answer aloud" below doesn't
        # recite URLs.
        with gr.Column(elem_id="synth-answer-label"):
            answer_box = gr.Textbox(
                label="Answer", lines=8, interactive=False, show_label=True, elem_id="synth-answer"
            )
        sources_box = gr.Markdown(elem_id="synth-sources")
        read_aloud_button = gr.Button("🔊 Read answer aloud")

    gr.Markdown(
        "Synthetic.API — *The Dawn of the Autonomous AI Builder* hackathon (Lyzr × Qdrant × Omi).",
        elem_id="synth-footer",
    )

    _ask_outputs = [
        status_box, answer_box, sources_box,
        human_input_group, gate_prompt_md, gate_email_box, gate_password_box,
        run_id_state,
    ]
    ask_button.click(ask, inputs=question_box, outputs=_ask_outputs)
    question_box.submit(ask, inputs=question_box, outputs=_ask_outputs)
    gate_continue_btn.click(
        resume_gate, inputs=[run_id_state, gate_email_box, gate_password_box], outputs=_ask_outputs
    )
    read_aloud_button.click(None, inputs=answer_box, outputs=None, js=_READ_ALOUD_JS)

if __name__ == "__main__":
    # ask() blocks its own request for up to _MAX_WAIT_SECONDS while polling.
    # Gradio's queue defaults to a concurrency limit of 1, which would
    # serialize every question behind that wait -- a second person's click
    # would just sit frozen for up to 5 minutes during a live demo. Raise it
    # so a handful of people can ask questions at the same time.
    demo.queue(default_concurrency_limit=10).launch(server_name="0.0.0.0", server_port=7860)
