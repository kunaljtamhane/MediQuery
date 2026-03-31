"""
Person A — Reranking (Weeks 3-4, upgraded in Weeks 5-6)
Re-ranks candidate passages using the reward model service.
Falls back to original scores if reward model is unavailable.
"""
import os
import httpx
import logging

log = logging.getLogger(__name__)
REWARD_MODEL_URL = os.getenv("REWARD_MODEL_URL", "http://reward_model:8003")


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-rank candidates using the reward model.
    candidates: list of {doc_id, text, score, metadata}

    Weeks 3-4: reward model isn't trained yet — uses original RRF scores.
    Weeks 5-6: once reward model is trained, this will call /rerank endpoint.
    """
    try:
        resp = httpx.post(
            f"{REWARD_MODEL_URL}/rerank",
            json={"query": query, "candidates": [c["text"] for c in candidates]},
            timeout=5.0,
        )
        resp.raise_for_status()
        rerank_scores = resp.json()["scores"]  # list of floats, same order as candidates
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = s
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
    except Exception as e:
        log.warning(f"Reward model unavailable ({e}), using RRF scores")
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]
