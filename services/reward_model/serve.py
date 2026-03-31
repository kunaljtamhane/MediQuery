"""
Person A — Reward Model Server (Weeks 5-6)
Serves the fine-tuned cross-encoder via FastAPI.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import os
import logging

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

app = FastAPI(title="Reward Model Service")

MODEL_PATH = os.getenv("REWARD_MODEL_PATH", "./model")

# Falls back to base checkpoint if fine-tuned model not yet available
try:
    model = CrossEncoder(MODEL_PATH)
    log.info(f"Loaded fine-tuned reward model from {MODEL_PATH}")
except Exception:
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    log.warning("Fine-tuned model not found — using base ms-marco checkpoint")


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
