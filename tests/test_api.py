from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app

client = TestClient(app)


def _fake_run_pipeline(raw_application, application_id, llm=None, retriever=None):
    return {
        "application_id": application_id,
        "decision": "manual_review",
        "risk_band": "medium",
        "risk_probability": 0.45,
        "risk_factors": [],
        "policy_explanation": "Borderline case routed to manual review.",
        "policy_sources": ["risk_bands_and_manual_review_policy.md"],
        "intake_errors": [],
        "trace": [{"agent": "intake", "summary": "ok", "detail": []}],
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_submit_and_retrieve_application(monkeypatch, valid_application):
    monkeypatch.setattr(api_main, "run_pipeline", _fake_run_pipeline)

    submit_response = client.post("/applications/submit", json=valid_application)
    assert submit_response.status_code == 200
    body = submit_response.json()
    assert body["decision"] == "manual_review"
    assert body["risk_band"] == "medium"
    application_id = body["application_id"]

    status_response = client.get(f"/applications/{application_id}/status")
    assert status_response.status_code == 200
    assert status_response.json() == body


def test_status_for_unknown_id_is_404():
    response = client.get("/applications/does-not-exist/status")
    assert response.status_code == 404


def test_submit_rejects_malformed_application(monkeypatch, valid_application):
    monkeypatch.setattr(api_main, "run_pipeline", _fake_run_pipeline)
    bad = dict(valid_application)
    del bad["limit_bal"]

    response = client.post("/applications/submit", json=bad)
    assert response.status_code == 422
