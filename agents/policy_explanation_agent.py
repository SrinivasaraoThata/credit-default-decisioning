"""Policy-explanation agent: retrieves relevant credit policy text and
explains the risk assessment in plain language grounded in that text,
using Gemini. The risk score itself comes from the risk-scoring agent, not
the LLM; the LLM only explains it.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from agents.decision_agent import citable_factors, risk_band_for
from agents.policy_store import query_policy
from agents.schemas import PipelineState

CHAT_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = (
    "You are a credit policy explanation assistant for a bank. Explain a "
    "risk assessment in plain, direct language for the applicant, grounded "
    "only in the policy excerpts and risk factors given to you. Do not "
    "invent policy that isn't in the excerpts. Never cite sex, age, "
    "marital status, or education as a reason for the assessment. Keep the "
    "explanation under 120 words, no headers or bullet points."
)


def _build_query(state: PipelineState) -> str:
    factors = citable_factors(state.get("risk_factors", []))
    band = risk_band_for(state["risk_probability"])
    factor_text = ", ".join(f"{rf.feature} ({rf.direction})" for rf in factors)
    return f"risk band {band}, driven by {factor_text}"


def _build_user_prompt(state: PipelineState, excerpts: list[tuple[str, str]]) -> str:
    factors = citable_factors(state.get("risk_factors", []))
    factor_lines = "\n".join(f"- {rf.feature}: {rf.direction} (shap={rf.shap_value:.3f})" for rf in factors)
    policy_lines = "\n".join(f"- ({source}) {text}" for text, source in excerpts)
    band = risk_band_for(state["risk_probability"])
    return (
        f"Predicted default probability: {state['risk_probability']:.3f} (risk band: {band})\n\n"
        f"Top behavioral risk factors:\n{factor_lines}\n\n"
        f"Relevant policy excerpts:\n{policy_lines}\n\n"
        "Write the applicant-facing explanation now."
    )


def run_policy_explanation(state: PipelineState, llm=None, retriever=None) -> PipelineState:
    retriever = retriever or query_policy
    query = _build_query(state)
    excerpts = retriever(query)

    llm = llm or ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", _build_user_prompt(state, excerpts)),
    ]
    response = llm.invoke(messages)
    explanation = response.content if hasattr(response, "content") else str(response)

    sources = sorted({source for _, source in excerpts})

    trace = list(state.get("trace", []))
    trace.append({
        "agent": "policy_explanation",
        "summary": "Generated plain-language explanation grounded in retrieved policy text.",
        "detail": sources,
    })

    return {
        **state,
        "policy_explanation": explanation,
        "policy_sources": sources,
        "trace": trace,
    }
