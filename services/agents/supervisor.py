"""
Person A — Supervisor Agent (Weeks 7-8)
Classifies query intent and routes to the correct specialist agent.

Routes:
  - "rag_agent"       → query is about papers already in the corpus
  - "research_agent"  → query needs live arXiv lookup (e.g., "latest papers on X")
"""
import os
import logging
from openai import OpenAI

log = logging.getLogger(__name__)

ROUTING_PROMPT = """You are a routing agent. Classify the user query into one of two categories:

1. "rag_agent"       — The query is asking about AI/ML research concepts, methods, or findings
                        that are likely in our paper corpus. Examples:
                        "What is RLHF?", "Compare BERT and GPT", "How does RAG work?"

2. "research_agent"  — The query explicitly asks for the LATEST or NEWEST papers,
                        or asks for papers on a very specific recent topic.
                        Examples: "Find the latest 2025 papers on LLM agents",
                                  "What papers came out this week about diffusion models?"

Reply with ONLY the category name: rag_agent or research_agent.

Query: {query}"""


def supervisor_node(state: dict) -> dict:
    """LangGraph node: classifies intent and sets state['route']."""
    query = state["query"]
    route = _classify(query)
    log.info(f"Supervisor routed '{query[:60]}' → {route}")
    return {**state, "route": route}


def _classify(query: str) -> str:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": ROUTING_PROMPT.format(query=query)}],
            max_tokens=10,
            temperature=0,
        )
        label = resp.choices[0].message.content.strip().lower()
        if label in ("rag_agent", "research_agent"):
            return label
    except Exception as e:
        log.warning(f"Supervisor LLM call failed ({e}), defaulting to rag_agent")
    return "rag_agent"
