"""Orchestrator FastAPI app.

Entry points into the whole system:
  POST /webhook/omi   - real Omi voice-transcript webhook
  POST /trigger        - manual trigger for local dev/demo (same payload shape as a parsed Omi transcript)
  GET  /runs/{run_id}  - inspect a run's live state (also readable at data/runs/<run_id>.json)
"""

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel

from agents.common import run_store
from agents.common.logging import configure_logging, get_logger
from agents.orchestrator import omi_webhook, planner
from agents.orchestrator.executor import execute_plan
from agents.web_navigator import page_handlers  # noqa: F401 - registers handlers
# NOTE: agents.orchestrator.handlers, agents.web_navigator.handlers, and
# agents.web_navigator.research_handlers registered the shipping-portal and
# DDG+vision pipelines' DAG node handlers. Both pipelines are retired from
# live routing (see planner.py) -- their modules are still present and
# still tested, just no longer imported here, so their handler_keys are
# absent from HANDLER_REGISTRY in the live app.

configure_logging("orchestrator")
logger = get_logger(component="orchestrator.main")

app = FastAPI(title="Synthetic.API Orchestrator", version="0.1.0")


class TriggerRequest(BaseModel):
    transcript: str


class TriggerResponse(BaseModel):
    run_id: str
    status: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/trigger", response_model=TriggerResponse)
def trigger(req: TriggerRequest, background_tasks: BackgroundTasks) -> TriggerResponse:
    try:
        plan = planner.build_plan(req.transcript)
    except planner.PlannerInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("plan_built", run_id=plan.run_id, status=plan.status, node_count=len(plan.nodes))
    background_tasks.add_task(execute_plan, plan)
    return TriggerResponse(run_id=plan.run_id, status=plan.status)


@app.post("/webhook/omi", response_model=TriggerResponse)
def webhook_omi(
    payload: dict,
    background_tasks: BackgroundTasks,
    x_omi_signature: str | None = Header(default=None),
) -> TriggerResponse:
    omi_webhook.verify_webhook_secret(x_omi_signature)
    try:
        transcript = omi_webhook.parse_omi_payload(payload)
        plan = planner.build_plan(transcript)
    except (omi_webhook.OmiPayloadError, planner.PlannerInputError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("omi_plan_built", run_id=plan.run_id, status=plan.status)
    background_tasks.add_task(execute_plan, plan)
    return TriggerResponse(run_id=plan.run_id, status=plan.status)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    run = run_store.load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run.model_dump(mode="json")
