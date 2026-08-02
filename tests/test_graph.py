from agents.graph import run_pipeline
from tests.conftest import FakeLLM, fake_retriever


def test_full_pipeline_reaches_a_decision(valid_application):
    result = run_pipeline(
        valid_application, application_id="app-1",
        llm=FakeLLM(), retriever=fake_retriever,
    )
    assert result["decision"] in {"approve", "deny", "manual_review"}
    assert result["risk_band"] in {"low", "medium", "high"}
    agents_in_trace = [t["agent"] for t in result["trace"]]
    assert agents_in_trace == ["intake", "risk_scoring", "policy_explanation", "decision"]


def test_malformed_application_is_rejected_before_scoring(valid_application):
    malformed = dict(valid_application)
    del malformed["limit_bal"]

    result = run_pipeline(
        malformed, application_id="app-2",
        llm=FakeLLM(), retriever=fake_retriever,
    )
    assert result["decision"] == "rejected_malformed"
    agents_in_trace = [t["agent"] for t in result["trace"]]
    assert agents_in_trace == ["intake"]
    assert "risk_probability" not in result
