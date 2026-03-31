"""
Person A + D — LangGraph Agent Orchestration (Weeks 7-8)
Defines the agent graph: Supervisor routes to RAG Agent or Research Agent.

Graph flow:
    User Query → Supervisor → (rag_agent | research_agent) → Response
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

from supervisor import supervisor_node
from rag_agent import rag_agent_node
from research_agent import research_agent_node


class AgentState(TypedDict):
    query: str
    session_id: str
    route: str                              # set by supervisor: "rag" | "research"
    rag_results: list[dict]                 # set by rag_agent
    research_results: list[dict]            # set by research_agent
    final_answer: str                       # set by whichever agent runs
    sources: list[dict]                     # citations


def route_decision(state: AgentState) -> str:
    """Router function: returns next node name based on supervisor's decision."""
    return state.get("route", "rag_agent")


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("research_agent", research_agent_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "rag_agent": "rag_agent",
            "research_agent": "research_agent",
        },
    )

    # Both agents lead to END
    graph.add_edge("rag_agent", END)
    graph.add_edge("research_agent", END)

    return graph.compile()


# Singleton compiled graph
agent_graph = build_graph()
