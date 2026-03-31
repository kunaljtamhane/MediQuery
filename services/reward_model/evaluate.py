"""
Person A — Reward Model Evaluation (Weeks 5-6)
Measures NDCG@5, Precision@5, MRR on a held-out benchmark.

Usage:
    python evaluate.py --model_dir ./model --benchmark_path ../../data/annotation/benchmark.jsonl
"""
import argparse
import json
import logging
import numpy as np
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def ndcg_at_k(relevances: list[int], k: int = 5) -> float:
    relevances = relevances[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevances: list[int], k: int = 5) -> float:
    return sum(1 for r in relevances[:k] if r > 0) / k


def mrr(relevances: list[int]) -> float:
    for i, r in enumerate(relevances):
        if r > 0:
            return 1 / (i + 1)
    return 0.0


def evaluate(model_dir: str, benchmark_path: str):
    """
    benchmark.jsonl format (each line):
    {
      "query": "...",
      "candidates": [{"text": "...", "label": 2}, {"text": "...", "label": 0}, ...]
    }
    Labels: 0=irrelevant, 1=partially relevant, 2=highly relevant
    """
    model = CrossEncoder(model_dir)

    ndcg_scores, p5_scores, mrr_scores = [], [], []

    with open(benchmark_path) as f:
        for line in f:
            item = json.loads(line)
            query = item["query"]
            candidates = item["candidates"]

            # Score all candidates
            pairs = [[query, c["text"]] for c in candidates]
            scores = model.predict(pairs)

            # Sort by model score
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            relevances = [c["label"] for _, c in ranked]

            ndcg_scores.append(ndcg_at_k(relevances, k=5))
            p5_scores.append(precision_at_k(relevances, k=5))
            mrr_scores.append(mrr(relevances))

    results = {
        "NDCG@5": round(np.mean(ndcg_scores), 4),
        "Precision@5": round(np.mean(p5_scores), 4),
        "MRR": round(np.mean(mrr_scores), 4),
        "num_queries": len(ndcg_scores),
    }
    log.info(f"Results: {results}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./model")
    parser.add_argument("--benchmark_path", default="../../data/annotation/benchmark.jsonl")
    args = parser.parse_args()
    evaluate(args.model_dir, args.benchmark_path)
