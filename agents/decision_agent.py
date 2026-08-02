"""Decision agent: approve / deny / manual review.

Uses risk bands rather than a single hard probability cutoff. Day 1's
fairness check found a meaningfully higher false-positive rate for male
applicants than female applicants at the same cutoff (22.6% vs 17.2%, see
docs/day1_findings.md). A single threshold turns every borderline case into
a confident-looking but potentially wrong automated decision, which is
exactly where that kind of disparity shows up. Routing the middle band to
manual review means borderline cases (where the model is least reliable,
and where the group gap is most likely to bite) get a human check instead
of an automated call either way.
"""

from agents.schemas import PipelineState

LOW_RISK_MAX = 0.30
HIGH_RISK_MIN = 0.60

# Fields that must not be cited as the reason for a decision shown to an
# applicant, per docs/policy_docs/fairness_and_compliance_policy.md. They can
# still be present in the model's input features, but the applicant-facing
# explanation is grounded in behavior, not demographics.
NON_CITABLE_FEATURES = {"SEX", "AGE", "MARRIAGE", "EDUCATION"}


def risk_band_for(probability: float) -> str:
    if probability < LOW_RISK_MAX:
        return "low"
    if probability >= HIGH_RISK_MIN:
        return "high"
    return "medium"


def citable_factors(risk_factors) -> list:
    return [rf for rf in risk_factors if rf.feature not in NON_CITABLE_FEATURES]


def run_decision(state: PipelineState) -> PipelineState:
    probability = state["risk_probability"]
    band = risk_band_for(probability)

    if band == "low":
        decision = "approve"
    elif band == "high":
        decision = "deny"
    else:
        decision = "manual_review"

    behavioral_factors = citable_factors(state.get("risk_factors", []))
    factor_summary = ", ".join(f"{rf.feature} ({rf.direction})" for rf in behavioral_factors[:3])

    trace = list(state.get("trace", []))
    trace.append({
        "agent": "decision",
        "summary": (
            f"Risk band '{band}' (probability {probability:.3f}) -> decision '{decision}'. "
            f"Cited factors: {factor_summary or 'none'}."
        ),
        "detail": [],
    })

    return {
        **state,
        "risk_band": band,
        "decision": decision,
        "trace": trace,
    }
