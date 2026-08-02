from agents.decision_agent import (
    HIGH_RISK_MIN,
    LOW_RISK_MAX,
    NON_CITABLE_FEATURES,
    citable_factors,
    risk_band_for,
    run_decision,
)
from agents.schemas import RiskFactor


def test_risk_band_boundaries():
    assert risk_band_for(0.0) == "low"
    assert risk_band_for(LOW_RISK_MAX - 0.01) == "low"
    assert risk_band_for(LOW_RISK_MAX) == "medium"
    assert risk_band_for((LOW_RISK_MAX + HIGH_RISK_MIN) / 2) == "medium"
    assert risk_band_for(HIGH_RISK_MIN) == "high"
    assert risk_band_for(1.0) == "high"


def test_middle_band_routes_to_manual_review_not_a_hard_cutoff():
    """Same lesson as Project 1: a single threshold turns every borderline
    case into a confident automated decision. There must be a real manual
    review band between approve and deny, not just two zones."""
    state = {"risk_probability": (LOW_RISK_MAX + HIGH_RISK_MIN) / 2, "risk_factors": [], "trace": []}
    result = run_decision(state)
    assert result["risk_band"] == "medium"
    assert result["decision"] == "manual_review"


def test_low_risk_approves_and_high_risk_denies():
    low_state = run_decision({"risk_probability": 0.05, "risk_factors": [], "trace": []})
    assert low_state["decision"] == "approve"

    high_state = run_decision({"risk_probability": 0.95, "risk_factors": [], "trace": []})
    assert high_state["decision"] == "deny"


def test_demographic_features_excluded_from_citable_factors():
    factors = [
        RiskFactor(feature="SEX", shap_value=0.1, direction="increases_risk"),
        RiskFactor(feature="AGE", shap_value=0.05, direction="increases_risk"),
        RiskFactor(feature="PAY_0", shap_value=0.4, direction="increases_risk"),
    ]
    result = citable_factors(factors)
    result_features = {rf.feature for rf in result}
    assert result_features == {"PAY_0"}
    assert result_features.isdisjoint(NON_CITABLE_FEATURES)
