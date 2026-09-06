"""Minimal demo UI for Synthetic.API.

Purely additive: this only calls the Orchestrator's existing HTTP API
(/trigger, /runs/{id}) over the network -- no direct access to Qdrant,
run_store, or anything else -- so it carries zero risk to the pipeline
that's already proven to work. It exists because curl/bash scripts are
fine for development but not a great surface for a judge or a live demo
audience.

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
"""

import os
import time

import gradio as gr
import requests

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
_POLL_INTERVAL_SECONDS = 3.0
# Generous ceiling: embed_pages' own node timeout is 180s (see
# orchestrator/planner.py), so this needs enough headroom above that plus
# fetch + the drafting LLM call itself for a real, non-trivial question.
_MAX_WAIT_SECONDS = 300


def ask(question: str):
    question = (question or "").strip()
    if not question:
        yield "Type a question first.", ""
        return

    yield "🔎 Sending your question to the Orchestrator...", ""

    try:
        resp = requests.post(f"{ORCHESTRATOR_URL}/trigger", json={"transcript": question}, timeout=10)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - show the real error, don't crash the UI
        yield f"⚠️ Could not reach the Orchestrator at {ORCHESTRATOR_URL}: {exc}", ""
        return

    body = resp.json()
    run_id = body.get("run_id")
    if not run_id:
        yield f"⚠️ Unexpected response from Orchestrator: {body}", ""
        return

    yield f"🛰️ Run `{run_id}` started — searching the web, fetching pages, and embedding...", ""

    waited = 0.0
    while waited < _MAX_WAIT_SECONDS:
        time.sleep(_POLL_INTERVAL_SECONDS)
        waited += _POLL_INTERVAL_SECONDS
        try:
            run_resp = requests.get(f"{ORCHESTRATOR_URL}/runs/{run_id}", timeout=10)
            run_resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            yield f"⚠️ Lost contact polling the run: {exc}", ""
            return

        run = run_resp.json()
        status = run.get("overall_status")
        if status in ("completed", "failed", "circuit_broken", "no_capability"):
            answer = run.get("answer")
            if answer:
                yield f"✅ Done (`{status}`, {waited:.0f}s elapsed).", answer
            else:
                yield f"⚠️ Run finished with status `{status}` but no answer was recorded.", ""
            return

        yield f"⏳ Still working... (`{status}`, {waited:.0f}s elapsed)", ""

    yield (
        f"⚠️ Timed out after {_MAX_WAIT_SECONDS}s waiting for run `{run_id}` — "
        f"check `docker compose logs` or poll `/runs/{run_id}` directly.",
        "",
    )


_READ_ALOUD_JS = """
(text) => {
    if (!text) { return; }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
}
"""

with gr.Blocks(title="Synthetic.API") as demo:
    gr.Markdown(
        "# Synthetic.API — the API-Less Bridge\n"
        "Type a question as if it were a transcript Omi already produced from your voice. "
        "In the real deployment, Omi's wearable does the speech-to-text itself and POSTs the "
        "transcript to `/webhook/omi` directly — this box is the same input, just typed for "
        "demo convenience."
    )
    question_box = gr.Textbox(
        label="Your question",
        placeholder="What is the current status of ISRO's Gaganyaan mission?",
    )
    ask_button = gr.Button("Ask", variant="primary")
    status_box = gr.Markdown()
    answer_box = gr.Textbox(label="Answer", lines=10, interactive=False)
    read_aloud_button = gr.Button("🔊 Read answer aloud")

    ask_button.click(ask, inputs=question_box, outputs=[status_box, answer_box])
    question_box.submit(ask, inputs=question_box, outputs=[status_box, answer_box])
    read_aloud_button.click(None, inputs=answer_box, outputs=None, js=_READ_ALOUD_JS)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
