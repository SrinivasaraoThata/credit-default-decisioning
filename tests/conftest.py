import pytest


@pytest.fixture
def valid_application() -> dict:
    return {
        "limit_bal": 200000, "sex": 1, "education": 2, "marriage": 1, "age": 35,
        "pay_0": 2, "pay_2": 2, "pay_3": 2, "pay_4": 0, "pay_5": 0, "pay_6": 0,
        "bill_amt1": 50000, "bill_amt2": 48000, "bill_amt3": 47000,
        "bill_amt4": 46000, "bill_amt5": 45000, "bill_amt6": 44000,
        "pay_amt1": 1000, "pay_amt2": 1000, "pay_amt3": 1000,
        "pay_amt4": 1000, "pay_amt5": 1000, "pay_amt6": 1000,
    }


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Stands in for ChatGoogleGenerativeAI so tests don't call the real API."""

    def __init__(self, content: str = "This applicant's recent payment history drives the assessment."):
        self.content = content
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return FakeLLMResponse(self.content)


def fake_retriever(query_text: str, k: int = 4):
    return [
        ("Payment History: repayment status is the strongest signal of default risk.", "payment_history_policy.md"),
        ("Risk Bands and Manual Review: borderline cases are routed to manual review.", "risk_bands_and_manual_review_policy.md"),
    ]
