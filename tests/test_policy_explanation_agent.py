from agents.policy_explanation_agent import run_policy_explanation
from agents.schemas import RiskFactor
from tests.conftest import FakeLLM, FakeLLMResponse, fake_retriever


def _base_state():
    return {
        "risk_probability": 0.72,
        "risk_factors": [
            RiskFactor(feature="PAY_0", shap_value=0.5, direction="increases_risk"),
            RiskFactor(feature="SEX", shap_value=0.1, direction="increases_risk"),
        ],
        "trace": [],
    }


def test_explanation_uses_retrieved_policy_and_fake_llm_no_network():
    llm = FakeLLM(content="Recent payment status is driving this assessment.")
    state = run_policy_explanation(_base_state(), llm=llm, retriever=fake_retriever)

    assert state["policy_explanation"] == "Recent payment status is driving this assessment."
    assert set(state["policy_sources"]) == {
        "payment_history_policy.md",
        "risk_bands_and_manual_review_policy.md",
    }
    assert state["trace"][-1]["agent"] == "policy_explanation"


def test_prompt_excludes_demographic_factors():
    llm = FakeLLM()
    run_policy_explanation(_base_state(), llm=llm, retriever=fake_retriever)

    user_prompt = llm.last_messages[1][1]
    assert "PAY_0" in user_prompt
    assert "SEX" not in user_prompt


class _StubLLM:
    """Mimics a thinking-model response where content is a list of blocks
    with an internal thought-signature under extras, not a plain string."""

    def __init__(self, content):
        self.content = content

    def invoke(self, messages):
        return FakeLLMResponse(self.content)


def test_thought_signature_is_stripped_from_explanation():
    llm = _StubLLM(content=[
        {
            "type": "text",
            "text": "Recent payment status is driving this assessment.",
            "extras": {"signature": "abc123-internal-thought-signature"},
        }
    ])
    state = run_policy_explanation(_base_state(), llm=llm, retriever=fake_retriever)

    assert state["policy_explanation"] == "Recent payment status is driving this assessment."
    assert "signature" not in state["policy_explanation"]
    assert "extras" not in state["policy_explanation"]
