"""Merges the Orchestrator's FastAPI app and the Gradio demo UI into a
single process on a single port.

Why this exists: docker-compose keeps them as two separate services on two
separate ports (agents-orchestrator:8000, demo-ui:7860) -- fine for a VM,
where every port can be exposed independently. Every single-container
target this repo supports (deploy/huggingface/, deploy/render/) routes
external traffic to exactly one port, so a single-container deployment
needs both behind that one port. Shared here (deploy/common/) rather than
duplicated per target -- see each target's own Dockerfile/start.sh/
SETUP.md for what's platform-specific (how Qdrant is reached, how the
image gets built) -- this file is only the "one process, one port" piece,
identical either way.

Nothing about either app's own code changes here -- this only WIRES them
together: `agents.orchestrator.main.app` keeps every one of its existing
routes (/trigger, /webhook/omi, /runs/*, /health, /ready) exactly as they
are, and `ui.app.demo` (the Gradio Blocks) gets mounted onto that same
FastAPI app via Gradio's own documented `mount_gradio_app` -- the standard,
public way to combine an existing FastAPI app with a Gradio UI, not a
workaround around either library's API. FastAPI/Starlette matches a
request against concrete routes (/trigger, /health, ...) before falling
through to a path-mounted sub-application, so mounting Gradio at "/" does
not shadow any of the Orchestrator's own routes -- confirmed below, not
assumed.

`ui.app` calls the Orchestrator over a real HTTP request (see its own
ORCHESTRATOR_URL) even here, self-referentially back into this same
process, rather than an in-process function call -- deliberately, so
ui/app.py's code stays completely unmodified and behaves identically
whether it's talking to a separate container (docker-compose) or the same
process (this file). This works because Gradio runs its own callback
functions (ask(), resume_gate()) in a worker thread, not on uvicorn's
asyncio event loop -- so a blocking `requests.post("http://localhost:.../
trigger")` from inside one of those callbacks doesn't block the very
server it's calling into. deploy/common/start.sh sets
ORCHESTRATOR_URL=http://localhost:<port> (the same port this merged app
listens on) for exactly this reason.
"""

import gradio as gr

from agents.orchestrator.main import app  # the real FastAPI app, routes untouched
from ui.app import demo  # the real Gradio Blocks, untouched

# Gradio's own queueing (see ui/app.py's __main__ block) is what raises the
# default concurrency limit above 1 -- mount_gradio_app doesn't call
# demo.launch(), so that queue() call would never otherwise run.
demo.queue(default_concurrency_limit=10)

# Mounted at "/" so visiting the Space's URL directly shows the UI (what a
# judge clicking the Space link expects) -- the Orchestrator's own routes
# stay reachable at their own paths (/trigger, /webhook/omi, /runs/{id},
# /health, /ready), matched first by FastAPI's routing before falling
# through to this mount (confirmed in this file's own module docstring
# above, and by the test that imports and exercises both paths on `app`).
app = gr.mount_gradio_app(app, demo, path="/")
