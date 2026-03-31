"""
Person D — Annotation Tool (Weeks 5-6)
CLI tool to label (query, passage, relevance) triples for reward model training.

Usage:
    python annotate.py --queries_file seed_queries.txt \
                       --output triples.jsonl \
                       --rag_url http://localhost:8002

Relevance scale:
    0 = Irrelevant
    1 = Partially relevant
    2 = Highly relevant
    3 = Perfect answer
"""
import argparse
import json
import httpx
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SEED_QUERIES = [
    # TODO Week 5: Add 30 queries about your paper corpus topics.
    # These should cover the range of questions users might ask.
    "What is retrieval-augmented generation?",
    "How does RLHF improve language model alignment?",
    "What are the key differences between BERT and GPT?",
    "How does attention mechanism work in transformers?",
    "What is the role of cross-encoders in reranking?",
    # ... add more
]


def fetch_candidates(query: str, rag_url: str, n: int = 10) -> list[dict]:
    """Retrieve top-N candidates from the RAG service for annotation."""
    try:
        # We call the embedding service directly for candidates (not the full pipeline)
        embedding_url = rag_url.replace(":8002", ":8001")
        resp = httpx.post(f"{embedding_url}/query", json={"text": query, "n_results": n}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"Failed to fetch candidates: {e}")
        return []


def annotate(queries_file: str, output_path: str, rag_url: str):
    """Interactive CLI annotation loop."""
    # Load queries
    if queries_file and Path(queries_file).exists():
        queries = Path(queries_file).read_text().strip().splitlines()
    else:
        queries = SEED_QUERIES
        log.info("Using built-in seed queries")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing annotations to allow resuming
    existing = set()
    if Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line)
                existing.add((row["query"], row["positive"]))
        log.info(f"Resuming — {len(existing)} triples already annotated")

    with open(output_path, "a") as out:
        for query in queries:
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print("="*60)

            candidates = fetch_candidates(query, rag_url)
            if not candidates:
                print("  No candidates found, skipping.")
                continue

            labeled = []
            for i, cand in enumerate(candidates):
                print(f"\n[{i+1}] {cand['text'][:300]}...")
                while True:
                    label = input("  Relevance (0-3) or 's' to skip query: ").strip()
                    if label == "s":
                        break
                    if label in ("0", "1", "2", "3"):
                        labeled.append((cand, int(label)))
                        break
                    print("  Invalid input. Enter 0, 1, 2, 3, or 's'.")

            # Write pairwise triples: for each positive (label>=2), pair with a negative (label<=1)
            positives = [(c, l) for c, l in labeled if l >= 2]
            negatives = [(c, l) for c, l in labeled if l <= 1]

            for pos, _ in positives:
                for neg, _ in negatives:
                    triple = {
                        "query": query,
                        "positive": pos["text"],
                        "negative": neg["text"],
                    }
                    if (query, pos["text"]) not in existing:
                        out.write(json.dumps(triple) + "\n")
                        existing.add((query, pos["text"]))

            print(f"  Saved {len(positives)} positives × {len(negatives)} negatives")

    print(f"\nAnnotation complete. Total triples: {len(existing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_file", default="")
    parser.add_argument("--output", default="triples.jsonl")
    parser.add_argument("--rag_url", default="http://localhost:8002")
    args = parser.parse_args()
    annotate(args.queries_file, args.output, args.rag_url)
