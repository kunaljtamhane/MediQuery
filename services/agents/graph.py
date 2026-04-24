"""
Person A + D — LangGraph Agent Orchestration (Weeks 7-8)
Defines the full MediQuery agent graph.

Graph flow:
    User Query
        → supervisor_node  (classify intent → set route)
        → [rag_agent | medical_papers_agent | web_search_agent]
        → finalize_node    (guardrail check + MMR merge)
        → END
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

from supervisor import supervisor_node, finalize_node
from rag_agent import rag_agent_node
from research_agent import medical_papers_agent_node
from web_search_agent import web_search_agent_node


class AgentState(TypedDict):
    query: str
    session_id: str
    route: str                       # set by supervisor: rag_agent | medical_papers_agent | web_search_agent
    rag_results: list[dict]          # set by rag_agent
    research_results: list[dict]     # set by medical_papers_agent
    web_results: list[dict]          # set by web_search_agent
    knowledge_graph_context: str     # set by knowledge graph module
    final_answer: str                # set by whichever agent ran, refined by finalize
    sources: list[dict]              # citations from whichever agent ran
    guardrail_triggered: bool        # set by finalize_node


def route_decision(state: AgentState) -> str:
    """Router: returns the next node name based on supervisor's classification."""
    route = state.get("route", "rag_agent")
    valid = {"rag_agent", "medical_papers_agent", "web_search_agent"}
    return route if route in valid else "rag_agent"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("medical_papers_agent", medical_papers_agent_node)
    graph.add_node("web_search_agent", web_search_agent_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "rag_agent": "rag_agent",
            "medical_papers_agent": "medical_papers_agent",
            "web_search_agent": "web_search_agent",
        },
    )

    # All three agents converge into the finalize node
    graph.add_edge("rag_agent", "finalize")
    graph.add_edge("medical_papers_agent", "finalize")
    graph.add_edge("web_search_agent", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


agent_graph = build_graph()
