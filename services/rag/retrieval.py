"""
Person A — RAG Retrieval
Hybrid search: dense (QdrantDB via BioMedBERT embeddings) + sparse (BM25),
merged via Reciprocal Rank Fusion, then diversified via Maximal Marginal Relevance.
"""
import os
import httpx
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding:8001")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
COLLECTION_NAME = "papers"


def dense_search(query: str, n: int = 15) -> list[dict]:
    """BioMedBERT dense retrieval via the embedding service."""
    resp = httpx.post(
        f"{EMBEDDING_SERVICE_URL}/query",
        json={"text": query, "n_results": n},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()  # list of {doc_id, text, score, metadata}


def _scroll_all_docs() -> list[dict]:
    """Fetch all stored chunks from Qdrant for BM25 corpus construction."""
    docs = []
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for hit in results:
            docs.append({
                "doc_id": hit.payload.get("doc_id", str(hit.id)),
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k not in ("doc_id", "text")},
            })
        if offset is None:
            break
    return docs


def bm25_search(query: str, corpus: list[dict], n: int = 15) -> list[dict]:
    """BM25 keyword search over the in-memory corpus."""
    if not corpus:
        return []
    tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)
    return [
        {"doc_id": doc["doc_id"], "text": doc["text"], "score": float(s), "metadata": doc["metadata"]}
        for s, doc in ranked[:n]
    ]


def reciprocal_rank_fusion(results_a: list[dict], results_b: list[dict], k: int = 60) -> list[dict]:
    """Merge two ranked lists via Reciprocal Rank Fusion (k=60 per paper)."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for rank, doc in enumerate(results_a):
        did = doc["doc_id"]
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)
        docs[did] = doc
    for rank, doc in enumerate(results_b):
        did = doc["doc_id"]
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)
        docs[did] = doc
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"doc_id": did, "text": docs[did]["text"], "score": score, "metadata": docs[did]["metadata"]}
        for did, score in merged
    ]


def _embed_text(text: str) -> np.ndarray:
    """Get BioMedBERT embedding for a text string via embedding service."""
    resp = httpx.post(
        f"{EMBEDDING_SERVICE_URL}/embed",
        json={"text": text},
        timeout=10.0,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"])


def maximal_marginal_relevance(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    lambda_param: float = 0.5,
) -> list[dict]:
    """
    Maximal Marginal Relevance reranking to reduce redundancy.
    Balances relevance to query vs. diversity among selected passages.
    lambda_param=0.5 gives equal weight to relevance and diversity.
    """
    if not candidates:
        return []

    try:
        query_emb = _embed_text(query)
        doc_embs = [_embed_text(c["text"]) for c in candidates]
    except Exception:
        # Fallback to score-only ranking if embedding service is unavailable
        return candidates[:top_k]

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    while len(selected_indices) < top_k and remaining:
        best_idx, best_score = None, float("-inf")
        for i in remaining:
            relevance = cosine(doc_embs[i], query_emb)
            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = max(cosine(doc_embs[i], doc_embs[j]) for j in selected_indices)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Full retrieval pipeline:
      1. BioMedBERT dense search via QdrantDB
      2. BM25 keyword search over same corpus
      3. RRF merge
      4. MMR reranking for diversity
    Returns top_k diverse, relevant passages.
    """
    dense_results = dense_search(query, n=15)

    corpus = _scroll_all_docs()
    bm25_results = bm25_search(query, corpus, n=15)

    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    return maximal_marginal_relevance(query, fused, top_k=top_k)
