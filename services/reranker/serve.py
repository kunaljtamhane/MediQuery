"""
Reranker Service
Serves the fine-tuned cross-encoder via FastAPI for passage reranking.
Called by services/rag/rerank.py via RERANKER_URL env var.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import os
import logging

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

app = FastAPI(title="Reranker Service")

MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "./model")

try:
    model = CrossEncoder(MODEL_PATH)
    log.info(f"Loaded fine-tuned reranker from {MODEL_PATH}")
except Exception:
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    log.warning("Fine-tuned reranker not found — using base ms-marco checkpoint")


class RerankRequest(BaseModel):
    query: str
    candidates: list[str]


class RerankResponse(BaseModel):
    scores: list[float]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    """Score each (query, candidate) pair. Higher = more relevant."""
    pairs = [[req.query, c] for c in req.candidates]
    scores = model.predict(pairs).tolist()
    return RerankResponse(scores=scores)
