"""
Person A — Reranking
Re-ranks candidate passages using the fine-tuned cross-encoder reranker service.
Falls back to original RRF scores if the reranker is unavailable.
"""
import os
import httpx
import logging

log = logging.getLogger(__name__)
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8003")


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-rank candidates using the cross-encoder reranker.
    candidates: list of {doc_id, text, score, metadata}
    Returns top_n passages sorted by reranker score descending.
    """
    try:
        resp = httpx.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "candidates": [c["text"] for c in candidates]},
            timeout=5.0,
        )
        resp.raise_for_status()
        rerank_scores = resp.json()["scores"]
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = s
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
    except Exception as e:
        log.warning(f"Reranker unavailable ({e}), using RRF scores")
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]
