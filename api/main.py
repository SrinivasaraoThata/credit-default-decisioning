"""FastAPI wrapper around the LangGraph agent pipeline."""

import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

from agents.graph import run_pipeline
from agents.schemas import ApplicationInput, RiskFactor

app = FastAPI(title="Credit Default Decisioning API")

# In-memory store, fine for a portfolio demo. A real deployment would use a
# database so decisions survive a restart and can be audited later.
_applications: dict[str, dict] = {}


def _serialize_risk_factors(risk_factors) -> list[dict]:
    return [rf.model_dump() if isinstance(rf, RiskFactor) else rf for rf in risk_factors or []]


def _to_response(state: dict) -> dict:
    return {
        "application_id": state["application_id"],
        "decision": state.get("decision"),
        "risk_band": state.get("risk_band"),
        "risk_probability": state.get("risk_probability"),
        "risk_factors": _serialize_risk_factors(state.get("risk_factors")),
        "policy_explanation": state.get("policy_explanation"),
        "policy_sources": state.get("policy_sources", []),
        "intake_errors": state.get("intake_errors", []),
        "trace": state.get("trace", []),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/applications/submit")
def submit_application(application: ApplicationInput):
    application_id = str(uuid.uuid4())
    state = run_pipeline(application.model_dump(), application_id=application_id)
    response = _to_response(state)
    _applications[application_id] = response
    return response


@app.get("/applications/{application_id}/status")
def get_application_status(application_id: str):
    result = _applications.get(application_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return result
