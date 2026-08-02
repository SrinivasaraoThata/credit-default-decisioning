"""Risk-scoring agent: calls the trained XGBoost model directly (no LLM)
and explains the score with SHAP values for that single prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from agents.schemas import ApplicationInput, PipelineState, RiskFactor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_model = None
_feature_columns = None
_explainer = None


def _load_model():
    global _model, _feature_columns, _explainer
    if _model is None:
        _model = joblib.load(MODELS_DIR / "xgb_model.joblib")
        _feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")
        _explainer = shap.TreeExplainer(_model)
    return _model, _feature_columns, _explainer


def build_features(application: ApplicationInput) -> pd.DataFrame:
    """Reproduce the feature engineering from notebooks/02_features_modeling.ipynb."""
    pay_chronological = [
        application.pay_6,
        application.pay_5,
        application.pay_4,
        application.pay_3,
        application.pay_2,
        application.pay_0,
    ]
    x_idx = np.arange(len(pay_chronological))
    pay_trend = np.polyfit(x_idx, pay_chronological, 1)[0]

    bill_amts = [
        application.bill_amt1,
        application.bill_amt2,
        application.bill_amt3,
        application.bill_amt4,
        application.bill_amt5,
        application.bill_amt6,
    ]
    utilization = [np.clip(b / application.limit_bal, 0, 3) for b in bill_amts]
    avg_utilization = float(np.mean(utilization))

    pay_amts = [
        application.pay_amt1,
        application.pay_amt2,
        application.pay_amt3,
        application.pay_amt4,
        application.pay_amt5,
        application.pay_amt6,
    ]
    avg_pay_amt = float(np.mean(pay_amts))
    credit_limit_to_income_proxy = application.limit_bal / (avg_pay_amt + 1)

    row = {
        "LIMIT_BAL": application.limit_bal,
        "SEX": application.sex,
        "EDUCATION": application.education,
        "MARRIAGE": application.marriage,
        "AGE": application.age,
        "PAY_0": application.pay_0,
        "PAY_2": application.pay_2,
        "PAY_3": application.pay_3,
        "PAY_4": application.pay_4,
        "PAY_5": application.pay_5,
        "PAY_6": application.pay_6,
        "BILL_AMT1": application.bill_amt1,
        "BILL_AMT2": application.bill_amt2,
        "BILL_AMT3": application.bill_amt3,
        "BILL_AMT4": application.bill_amt4,
        "BILL_AMT5": application.bill_amt5,
        "BILL_AMT6": application.bill_amt6,
        "PAY_AMT1": application.pay_amt1,
        "PAY_AMT2": application.pay_amt2,
        "PAY_AMT3": application.pay_amt3,
        "PAY_AMT4": application.pay_amt4,
        "PAY_AMT5": application.pay_amt5,
        "PAY_AMT6": application.pay_amt6,
        "pay_trend": pay_trend,
        "avg_utilization": avg_utilization,
        "credit_limit_to_income_proxy": credit_limit_to_income_proxy,
    }
    _, feature_columns, _ = _load_model()
    return pd.DataFrame([row], columns=feature_columns)


def run_risk_scoring(state: PipelineState, top_n: int = 5) -> PipelineState:
    model, feature_columns, explainer = _load_model()
    application = state["application"]

    features = build_features(application)
    probability = float(model.predict_proba(features)[0, 1])

    shap_values = explainer.shap_values(features)[0]
    ranked = sorted(
        zip(feature_columns, shap_values),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_n]
    risk_factors = [
        RiskFactor(
            feature=name,
            shap_value=float(value),
            direction="increases_risk" if value > 0 else "decreases_risk",
        )
        for name, value in ranked
    ]

    trace = list(state.get("trace", []))
    trace.append({
        "agent": "risk_scoring",
        "summary": f"Predicted default probability {probability:.3f} from the trained XGBoost model.",
        "detail": [f"{rf.feature} ({rf.direction}, shap={rf.shap_value:.3f})" for rf in risk_factors],
    })

    return {
        **state,
        "risk_probability": probability,
        "risk_factors": risk_factors,
        "trace": trace,
    }
