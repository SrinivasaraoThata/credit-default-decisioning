"""Shared data contracts for the agentic decisioning pipeline."""

from typing import TypedDict

from pydantic import BaseModel, Field


class ApplicationInput(BaseModel):
    """Raw application fields, matching the UCI Default of Credit Card Clients schema."""

    limit_bal: float = Field(gt=0, description="Credit limit (NT dollar)")
    sex: int = Field(description="1 = male, 2 = female")
    education: int = Field(ge=0, le=6, description="1=grad school, 2=university, 3=high school, 4=other, 0/5/6=undocumented")
    marriage: int = Field(ge=0, le=3, description="1=married, 2=single, 3=other, 0=undocumented")
    age: int = Field(ge=18, le=100)
    pay_0: int = Field(ge=-2, le=9, description="Repayment status, most recent month")
    pay_2: int = Field(ge=-2, le=9)
    pay_3: int = Field(ge=-2, le=9)
    pay_4: int = Field(ge=-2, le=9)
    pay_5: int = Field(ge=-2, le=9)
    pay_6: int = Field(ge=-2, le=9)
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float = Field(ge=0)
    pay_amt2: float = Field(ge=0)
    pay_amt3: float = Field(ge=0)
    pay_amt4: float = Field(ge=0)
    pay_amt5: float = Field(ge=0)
    pay_amt6: float = Field(ge=0)


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" or "decreases_risk"


class PipelineState(TypedDict, total=False):
    application_id: str
    raw_application: dict
    application: ApplicationInput | None
    intake_valid: bool
    intake_errors: list[str]
    risk_probability: float
    risk_factors: list[RiskFactor]
    policy_explanation: str
    policy_sources: list[str]
    decision: str
    risk_band: str
    trace: list[dict]
