"""
Person A — RAG Retrieval (Weeks 3-4)
Implements hybrid search: dense (ChromaDB) + sparse (BM25) combined via
Reciprocal Rank Fusion (RRF).
"""
import os
import httpx
from rank_bm25 import BM25Okapi
import chromadb

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding:8001")
CHROMA_HOST = os.getenv("CHROMA_HOST", "qdrant")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_or_create_collection("papers")


def dense_search(query: str, n: int = 15) -> list[dict]:
    """Search using vector similarity via ChromaDB."""
    resp = httpx.post(f"{EMBEDDING_SERVICE_URL}/query", json={"text": query, "n_results": n})
    resp.raise_for_status()
    return resp.json()  # list of {doc_id, text, score, metadata}


def bm25_search(query: str, corpus: list[dict], n: int = 15) -> list[dict]:
    """
    BM25 keyword search over an in-memory corpus.
    corpus: list of {doc_id, text, metadata}

    TODO Week 3: If corpus is large (>50k chunks), replace this with
    OpenSearch BM25 for better performance.
    """
    if not corpus:
        return []
    tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)
    return [{"doc_id": doc["doc_id"], "text": doc["text"], "score": float(score), "metadata": doc["metadata"]}
            for score, doc in ranked[:n]]


def reciprocal_rank_fusion(results_a: list[dict], results_b: list[dict], k: int = 60) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    k=60 is the standard constant from the RRF paper.
    """
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
    return [{"doc_id": did, "text": docs[did]["text"], "score": score, "metadata": docs[did]["metadata"]}
            for did, score in merged]


def hybrid_search(query: str, top_k: int = 15) -> list[dict]:
    """
    Full hybrid search: dense + BM25 → RRF merge.
    Returns top_k ranked passages.
    """
    # Dense retrieval
    dense_results = dense_search(query, n=top_k)

    # BM25 over the same candidate pool (fetch all docs from ChromaDB)
    # TODO Week 3: Replace with OpenSearch BM25 query for full corpus
    all_docs_result = collection.get(include=["documents", "metadatas"])
    corpus = [
        {"doc_id": did, "text": text, "metadata": meta}
        for did, text, meta in zip(
            all_docs_result["ids"],
            all_docs_result["documents"],
            all_docs_result["metadatas"],
        )
    ]
    bm25_results = bm25_search(query, corpus, n=top_k)

    return reciprocal_rank_fusion(dense_results, bm25_results)[:top_k]
