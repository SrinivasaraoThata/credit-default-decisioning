from agents.intake_agent import run_intake
from agents.risk_scoring_agent import run_risk_scoring
from agents.schemas import ApplicationInput


def _scored_state(valid_application):
    state = run_intake({"raw_application": valid_application, "trace": []})
    assert state["intake_valid"]
    return run_risk_scoring(state)


def test_probability_in_valid_range(valid_application):
    state = _scored_state(valid_application)
    assert 0.0 <= state["risk_probability"] <= 1.0


def test_top_risk_factors_returned(valid_application):
    state = _scored_state(valid_application)
    assert len(state["risk_factors"]) == 5
    valid_columns = set(ApplicationInput.model_fields.keys())
    engineered = {"pay_trend", "avg_utilization", "credit_limit_to_income_proxy"}
    for rf in state["risk_factors"]:
        assert rf.direction in ("increases_risk", "decreases_risk")
        assert rf.feature.lower() in valid_columns or rf.feature in engineered or rf.feature.upper() == rf.feature


def test_worse_payment_history_increases_risk(valid_application):
    good = dict(valid_application)
    good.update(pay_0=-1, pay_2=-1, pay_3=-1, pay_4=-1, pay_5=-1, pay_6=-1)
    bad = dict(valid_application)
    bad.update(pay_0=4, pay_2=3, pay_3=3, pay_4=2, pay_5=2, pay_6=2)

    good_state = _scored_state(good)
    bad_state = _scored_state(bad)

    assert bad_state["risk_probability"] > good_state["risk_probability"]


def test_trace_records_risk_scoring_step(valid_application):
    state = _scored_state(valid_application)
    agents_in_trace = [t["agent"] for t in state["trace"]]
    assert "risk_scoring" in agents_in_trace
