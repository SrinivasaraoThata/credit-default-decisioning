"""Wires the four agents into a LangGraph pipeline:
intake -> risk_scoring -> policy_explanation -> decision.

A malformed application is rejected at intake and never reaches the model
or the LLM.
"""

from langgraph.graph import END, StateGraph

from agents.decision_agent import run_decision
from agents.intake_agent import run_intake
from agents.policy_explanation_agent import run_policy_explanation
from agents.risk_scoring_agent import run_risk_scoring
from agents.schemas import PipelineState


def _route_after_intake(state: PipelineState) -> str:
    return "continue" if state.get("intake_valid") else "reject"


def _reject(state: PipelineState) -> PipelineState:
    trace = list(state.get("trace", []))
    return {**state, "decision": "rejected_malformed", "risk_band": "n/a", "trace": trace}


def build_graph(llm=None, retriever=None):
    graph = StateGraph(PipelineState)
    graph.add_node("intake", run_intake)
    graph.add_node("risk_scoring", run_risk_scoring)
    graph.add_node("policy_explanation", lambda state: run_policy_explanation(state, llm=llm, retriever=retriever))
    graph.add_node("decision", run_decision)
    graph.add_node("reject", _reject)

    graph.set_entry_point("intake")
    graph.add_conditional_edges("intake", _route_after_intake, {"continue": "risk_scoring", "reject": "reject"})
    graph.add_edge("risk_scoring", "policy_explanation")
    graph.add_edge("policy_explanation", "decision")
    graph.add_edge("decision", END)
    graph.add_edge("reject", END)

    return graph.compile()


def run_pipeline(raw_application: dict, application_id: str, llm=None, retriever=None) -> PipelineState:
    graph = build_graph(llm=llm, retriever=retriever)
    initial_state: PipelineState = {
        "application_id": application_id,
        "raw_application": raw_application,
        "trace": [],
    }
    return graph.invoke(initial_state)
