"""Intake agent: validates a raw application against the expected schema
before anything downstream sees it."""

from pydantic import ValidationError

from agents.schemas import ApplicationInput, PipelineState


def run_intake(state: PipelineState) -> PipelineState:
    raw = state["raw_application"]
    trace = list(state.get("trace", []))

    try:
        application = ApplicationInput(**raw)
    except ValidationError as exc:
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        trace.append({
            "agent": "intake",
            "summary": f"Rejected malformed application ({len(errors)} field error(s)).",
            "detail": errors,
        })
        return {
            **state,
            "intake_valid": False,
            "intake_errors": errors,
            "trace": trace,
        }

    trace.append({
        "agent": "intake",
        "summary": "Application fields validated, all required fields present and in range.",
        "detail": [],
    })
    return {
        **state,
        "application": application,
        "intake_valid": True,
        "intake_errors": [],
        "trace": trace,
    }
