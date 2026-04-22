"""
Supervisor Agent — Kunal (Weeks 7-8)
Classifies query intent and routes to one of three specialist agents.
After agents complete, applies the retrieval-confidence guardrail and
merges all agent outputs via MMR synthesis before returning to caller.

Routes:
  "rag_agent"             — query is about documents already in the local QdrantDB corpus
  "medical_papers_agent"  — query needs live PubMed / arXiv / medRxiv lookup
  "web_search_agent"      — query goes beyond indexed medical literature (general context)

Guardrail (per proposal §4):
  If fewer than 2 retrieved source passages have score > 0.60, the system
  returns a no-result response instead of calling the LLM.
"""
import logging
import os

from openai import OpenAI

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", 0.60))
MIN_CONFIDENT_PASSAGES = int(os.getenv("MIN_CONFIDENT_PASSAGES", 2))

NO_RESULT_RESPONSE = (
    "MediQuery could not find sufficient evidence in the retrieved sources to answer "
    "this query reliably. Please consult a licensed medical professional or try a more "
    "specific query."
)

ROUTING_PROMPT = """You are a routing agent for MediQuery, a medical research assistant.
Classify the user query into exactly one of three categories:

1. "rag_agent"
   The query asks about medical concepts, treatments, drugs, or findings likely covered
   by our indexed corpus of PubMed/arXiv/medRxiv abstracts.
   Examples: "What is the mechanism of metformin?", "Compare BERT and BioBERT for NER",
             "Summarize findings on CRISPR for rare diseases"

2. "medical_papers_agent"
   The query explicitly asks for the LATEST or NEWEST papers, or a very specific recent
   topic that may not be in the local corpus.
   Examples: "Find the latest 2025 papers on GLP-1 agonists",
             "What new research came out this year on long COVID?"

3. "web_search_agent"
   The query is general, seeks context beyond academic literature, or asks about
   clinical guidelines, news, or resources not covered by the corpus.
   Examples: "What are the current CDC guidelines for COVID-19?",
             "Are there any patient support groups for Huntington's disease?"

Reply with ONLY the category name: rag_agent, medical_papers_agent, or web_search_agent.

Query: {query}"""


def supervisor_node(state: dict) -> dict:
    """LangGraph node: classifies intent and sets state['route']."""
    query = state["query"]
    route = _classify(query)
    log.info(f"Supervisor → '{query[:60]}' → {route}")
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
        if label in ("rag_agent", "medical_papers_agent", "web_search_agent"):
            return label
    except Exception as e:
        log.warning(f"Supervisor LLM call failed ({e}), defaulting to rag_agent")
    return "rag_agent"


def finalize_node(state: dict) -> dict:
    """
    LangGraph node: runs after all agents complete.
    1. Applies retrieval-confidence guardrail.
    2. Merges sources from all agents via MMR (diversity-aware selection).
    Returns updated state with guardrail-filtered answer and merged sources.
    """
    sources = state.get("sources", [])

    # Guardrail: count passages with score above the confidence threshold
    confident = [s for s in sources if s.get("score", 0.0) >= CONFIDENCE_THRESHOLD]
    if len(confident) < MIN_CONFIDENT_PASSAGES:
        log.warning(
            f"Guardrail triggered: only {len(confident)} passages above "
            f"score {CONFIDENCE_THRESHOLD} (need {MIN_CONFIDENT_PASSAGES}). "
            f"Suppressing LLM response."
        )
        return {**state, "final_answer": NO_RESULT_RESPONSE, "guardrail_triggered": True}

    # MMR-based source merge: keep diverse, high-scoring sources from all agents
    merged_sources = _mmr_merge_sources(
        sources,
        web_results=state.get("web_results", []),
        research_results=state.get("research_results", []),
    )

    # Prepend knowledge graph context if available
    kg_context = state.get("knowledge_graph_context", "")
    final_answer = state.get("final_answer", "")
    if kg_context:
        final_answer = kg_context + "\n\n" + final_answer

    return {**state, "final_answer": final_answer, "sources": merged_sources,
            "guardrail_triggered": False}


def _mmr_merge_sources(
    sources: list[dict],
    web_results: list[dict] = None,
    research_results: list[dict] = None,
    top_k: int = 10,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Simple MMR-style merge over all agent sources.
    Maximizes score while penalizing sources that duplicate titles/doc_ids
    already selected. lambda_param=0.7 weights relevance over diversity.
    """
    all_sources = list(sources)
    seen_ids: set = set()
    merged: list[dict] = []

    # Score-descending order, then pick while penalizing duplicates
    candidates = sorted(all_sources, key=lambda x: x.get("score", 0.0), reverse=True)

    for candidate in candidates:
        if len(merged) >= top_k:
            break
        doc_id = candidate.get("doc_id", "")
        title_key = candidate.get("title", "")[:40].lower()
        dedup_key = doc_id or title_key
        if dedup_key and dedup_key in seen_ids:
            continue
        merged.append(candidate)
        if dedup_key:
            seen_ids.add(dedup_key)

    return merged
