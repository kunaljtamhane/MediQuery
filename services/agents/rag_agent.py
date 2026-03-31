"""
Person A — RAG Agent (Weeks 7-8)
Wraps the RAG service (retrieval + rerank + generate) as a LangGraph node.
"""
import os
import httpx
import logging

log = logging.getLogger(__name__)
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag:8002")


def rag_agent_node(state: dict) -> dict:
    """
    LangGraph node: calls RAG service and collects the streamed answer.
    In the real frontend this streams token-by-token; here we collect it all.
    """
    query = state["query"]
    session_id = state.get("session_id", "default")

    sources = []
    answer_tokens = []

    try:
        with httpx.stream(
            "POST",
            f"{RAG_SERVICE_URL}/query",
            json={"query": query, "session_id": session_id, "top_k": 5},
            timeout=30.0,
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    break
                import json
                data = json.loads(payload)
                if "sources" in data:
                    sources = data["sources"]
                if "token" in data:
                    answer_tokens.append(data["token"])
    except Exception as e:
        log.error(f"RAG agent failed: {e}")
        answer_tokens = ["I encountered an error retrieving information."]

    return {
        **state,
        "final_answer": "".join(answer_tokens),
        "sources": sources,
    }
