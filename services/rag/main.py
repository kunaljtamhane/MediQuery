"""
Person A — RAG Service (Weeks 3-4)
Orchestrates: hybrid search → rerank → LLM generation.
POST /query  →  streaming answer with sources
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging

from retrieval import hybrid_search
from rerank import rerank
from generate import generate_stream

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

app = FastAPI(title="RAG Service")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    """
    Full RAG pipeline:
    1. Hybrid search (dense + BM25 + RRF)
    2. Reward model rerank
    3. Deduplication (Weeks 5-6 — skipped for now)
    4. Streaming LLM generation
    """
    log.info(f"Query: {req.query[:80]}")

    # Step 1: Retrieve candidates
    candidates = hybrid_search(req.query, top_k=15)

    # Step 2: Rerank top candidates
    top_passages = rerank(req.query, candidates, top_n=req.top_k)

    # TODO Week 5: Add deduplication check against Redis session cache here
    # from dedup import filter_duplicates
    # top_passages = filter_duplicates(top_passages, req.session_id)

    # Step 3: Stream LLM response
    def stream_with_sources():
        # First yield source metadata as a JSON header line
        import json
        sources = [{"doc_id": p["doc_id"], "title": p["metadata"].get("title", ""), "score": p.get("rerank_score", p["score"])} for p in top_passages]
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        # Then stream answer tokens
        for token in generate_stream(req.query, top_passages):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_with_sources(), media_type="text/event-stream")
