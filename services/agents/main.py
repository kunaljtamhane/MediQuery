"""
Person A + D — Agent Service Entry Point (Weeks 7-8)
Exposes the LangGraph pipeline as a FastAPI endpoint.
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import logging

from graph import agent_graph

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

app = FastAPI(title="Agent Service")


class AgentRequest(BaseModel):
    query: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: AgentRequest):
    """Run the full agent pipeline and return result."""
    initial_state = {
        "query": req.query,
        "session_id": req.session_id,
        "route": "",
        "rag_results": [],
        "research_results": [],
        "final_answer": "",
        "sources": [],
    }
    result = agent_graph.invoke(initial_state)
    return {
        "answer": result["final_answer"],
        "sources": result["sources"],
        "route_taken": result["route"],
    }
