#!/usr/bin/env python3
"""
Person D — Annotation Tool (Weeks 5-6)

CLI tool to label (query, passage, relevance) triples for reward model training.

This tool:
1. Loads seed queries from a file or built-in defaults
2. Fetches top candidate passages from the retrieval/embedding service
3. Lets a human assign relevance labels
4. Converts labels into pairwise training triples:
      (query, positive, negative)
5. Saves results to JSONL so training can use them later

Usage:
    python annotate.py --queries_file seed_queries.txt \
                       --output data/annotation/triples.jsonl \
                       --rag_url http://localhost:8002

Relevance scale:
    0 = Irrelevant
    1 = Partially relevant
    2 = Highly relevant
    3 = Perfect answer
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

SEED_QUERIES = [
    "What is retrieval-augmented generation?",
    "How does RLHF improve language model alignment?",
    "What are the key differences between BERT and GPT?",
    "How does attention mechanism work in transformers?",
    "What is the role of cross-encoders in reranking?",
    "How do dense retrieval and BM25 differ?",
    "What is hybrid search in information retrieval?",
    "How are embeddings used for document search?",
    "What are the benefits of multi-agent systems in research tools?",
    "How does reranking improve retrieval quality?",
]


def load_queries(queries_file: str) -> List[str]:
    """Load queries from file if provided, otherwise use defaults."""
    if queries_file:
        path = Path(queries_file)
        if path.exists():
            queries = [
                line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if queries:
                log.info("Loaded %d queries from %s", len(queries), queries_file)
                return queries
            log.warning("Query file exists but is empty. Falling back to built-in seed queries.")
        else:
            log.warning("Query file not found: %s. Falling back to built-in seed queries.", queries_file)

    log.info("Using built-in seed queries")
    return SEED_QUERIES


def normalize_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Extract candidate text from a retrieval result.
    Supports several common field names.
    """
    for key in ("text", "document", "chunk", "content", "passage"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def normalize_candidates(response_json: Any) -> List[Dict[str, Any]]:
    """
    Normalize API response into a list of candidate dicts with at least:
        {
            "text": "...",
            "metadata": {...}
        }
    Supports:
    - list[dict]
    - {"results": [...]}
    - {"documents": [...]}
    - {"matches": [...]}
    """
    raw_items = []

    if isinstance(response_json, list):
        raw_items = response_json
    elif isinstance(response_json, dict):
        for key in ("results", "documents", "matches", "items", "data"):
            if isinstance(response_json.get(key), list):
                raw_items = response_json[key]
                break

    normalized = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        text = normalize_candidate_text(item)
        if not text:
            continue

        normalized.append({
            "text": text,
            "metadata": item.get("metadata", {}),
            "raw": item,
        })

    return normalized


def fetch_candidates(query: str, rag_url: str, n: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve top-N candidates from the embedding/retrieval service.

    Current logic:
    - User may pass RAG URL like http://localhost:8002
    - We convert it to embedding/query service on :8001
    """
    try:
        embedding_url = rag_url.replace(":8002", ":8001")
        url = f"{embedding_url}/query"

        payload = {
            "text": query,
            "n_results": n
        }

        with httpx.Client(timeout=20.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        candidates = normalize_candidates(data)
        log.info("Fetched %d candidates for query: %s", len(candidates), query)
        return candidates

    except Exception as e:
        log.error("Failed to fetch candidates for query '%s': %s", query, e)
        return []


def load_existing_triples(output_path: Path) -> set[Tuple[str, str, str]]:
    """
    Load already saved triples to support resume mode
    and prevent duplicate entries.
    """
    existing = set()

    if not output_path.exists():
        return existing

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                triple_key = (
                    row.get("query", "").strip(),
                    row.get("positive", "").strip(),
                    row.get("negative", "").strip(),
                )
                if all(triple_key):
                    existing.add(triple_key)
            except json.JSONDecodeError:
                continue

    log.info("Resuming — %d existing triples found", len(existing))
    return existing


def print_candidate(index: int, candidate: Dict[str, Any], max_chars: int = 500) -> None:
    """Pretty-print one candidate to the terminal."""
    text = candidate["text"]
    preview = text[:max_chars].replace("\n", " ").strip()

    print(f"\n[{index}]")
    print("-" * 60)
    print(preview + ("..." if len(text) > max_chars else ""))
    print("-" * 60)


def label_candidates(query: str, candidates: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int]]:
    """
    Ask the annotator to label each candidate.
    Returns list of (candidate, relevance_label).
    """
    labeled: List[Tuple[Dict[str, Any], int]] = []

    print(f"\n{'=' * 80}")
    print(f"QUERY: {query}")
    print(f"{'=' * 80}")
    print("Instructions:")
    print("  0 = Irrelevant")
    print("  1 = Partially relevant")
    print("  2 = Highly relevant")
    print("  3 = Perfect answer")
    print("  c = skip this candidate")
    print("  s = skip the entire query")
    print()

    for i, candidate in enumerate(candidates, start=1):
        print_candidate(i, candidate)

        while True:
            label = input("Enter label (0/1/2/3, c, s): ").strip().lower()

            if label == "s":
                print("Skipping entire query.\n")
                return []

            if label == "c":
                print("Candidate skipped.")
                break

            if label in {"0", "1", "2", "3"}:
                labeled.append((candidate, int(label)))
                break

            print("Invalid input. Please enter 0, 1, 2, 3, c, or s.")

    return labeled


def build_triples(
    query: str,
    labeled: List[Tuple[Dict[str, Any], int]]
) -> List[Dict[str, Any]]:
    """
    Convert labeled candidates into pairwise triples.

    Positives:
        labels 2 or 3
    Negatives:
        labels 0 or 1
    """
    positives = [(candidate, label) for candidate, label in labeled if label >= 2]
    negatives = [(candidate, label) for candidate, label in labeled if label <= 1]

    triples = []

    for pos_candidate, pos_label in positives:
        for neg_candidate, neg_label in negatives:
            pos_text = pos_candidate["text"].strip()
            neg_text = neg_candidate["text"].strip()

            if not pos_text or not neg_text or pos_text == neg_text:
                continue

            triple = {
                "query": query,
                "positive": pos_text,
                "negative": neg_text,
                "positive_label": pos_label,
                "negative_label": neg_label,
                "positive_metadata": pos_candidate.get("metadata", {}),
                "negative_metadata": neg_candidate.get("metadata", {}),
            }
            triples.append(triple)

    return triples


def save_triples(
    output_path: Path,
    triples: List[Dict[str, Any]],
    existing: set[Tuple[str, str, str]]
) -> int:
    """Append new unique triples to disk."""
    saved_count = 0

    with output_path.open("a", encoding="utf-8") as out:
        for triple in triples:
            key = (
                triple["query"].strip(),
                triple["positive"].strip(),
                triple["negative"].strip(),
            )

            if key in existing:
                continue

            out.write(json.dumps(triple, ensure_ascii=False) + "\n")
            existing.add(key)
            saved_count += 1

    return saved_count


def annotate(queries_file: str, output_path: str, rag_url: str, n_results: int = 10) -> None:
    """Main interactive annotation loop."""
    queries = load_queries(queries_file)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_triples(output_file)

    total_saved = 0

    for query in queries:
        candidates = fetch_candidates(query=query, rag_url=rag_url, n=n_results)

        if not candidates:
            print(f"\nNo candidates found for query: {query}")
            continue

        labeled = label_candidates(query, candidates)

        if not labeled:
            print("No labels saved for this query.")
            continue

        positives = sum(1 for _, label in labeled if label >= 2)
        negatives = sum(1 for _, label in labeled if label <= 1)

        if positives == 0 or negatives == 0:
            print(
                f"Not enough label variety to form triples. "
                f"Need at least 1 positive (2/3) and 1 negative (0/1). "
                f"Found positives={positives}, negatives={negatives}."
            )
            continue

        triples = build_triples(query, labeled)
        saved_now = save_triples(output_file, triples, existing)
        total_saved += saved_now

        print(
            f"\nSaved {saved_now} new triples "
            f"(positives={positives}, negatives={negatives}, total_pairs={len(triples)})."
        )

    print(f"\nAnnotation complete. Total new triples saved in this run: {total_saved}")
    print(f"Output file: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Annotation CLI for reward model training data")
    parser.add_argument(
        "--queries_file",
        type=str,
        default="",
        help="Optional text file with one query per line"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/annotation/triples.jsonl",
        help="Path to output JSONL triples file"
    )
    parser.add_argument(
        "--rag_url",
        type=str,
        default="http://localhost:8002",
        help="Base URL of the RAG service"
    )
    parser.add_argument(
        "--n_results",
        type=int,
        default=10,
        help="Number of retrieved candidates per query"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    annotate(
        queries_file=args.queries_file,
        output_path=args.output,
        rag_url=args.rag_url,
        n_results=args.n_results
    )


if __name__ == "__main__":
    main()