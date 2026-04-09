"""
Bulk ingestion script — sends the 500 papers with full_text extracted
to the Spring Boot /ingest endpoint, which publishes each to Kafka.

Usage:
    python bulk_ingest.py [--input PATH] [--url URL] [--delay SECONDS]

Defaults:
    --input  ../../data/annotation/papers.jsonl
    --url    http://localhost:8080/ingest
    --delay  0.1   (seconds between requests — be gentle on Kafka)
"""
import argparse
import json
import time
import sys
import httpx

DEFAULT_INPUT = "../../data/annotation/papers.jsonl"
DEFAULT_URL   = "http://localhost:8080/ingest"
DEFAULT_DELAY = 0.1


def load_papers(path: str) -> list[dict]:
    papers = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {i} invalid JSON, skipping: {e}")
                continue
            if d.get("full_text_extracted") and d.get("full_text"):
                papers.append(d)
    return papers


def build_payload(paper: dict) -> dict:
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        authors = ", ".join(authors)
    return {
        "docId":         paper["arxiv_id"],
        "title":         paper["title"],
        "text":          paper["full_text"],
        "authors":       authors,
        "publishedDate": paper.get("published_date", ""),
        "arxivUrl":      paper.get("pdf_url", ""),
    }


def ingest(papers: list[dict], url: str, delay: float):
    total   = len(papers)
    ok      = 0
    failed  = []

    with httpx.Client(timeout=30.0) as client:
        for i, paper in enumerate(papers, 1):
            payload = build_payload(paper)
            try:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                ok += 1
                print(f"[{i}/{total}] queued  {paper['arxiv_id']}")
            except httpx.HTTPStatusError as e:
                print(f"[{i}/{total}] ERROR   {paper['arxiv_id']} — HTTP {e.response.status_code}: {e.response.text[:120]}")
                failed.append(paper["arxiv_id"])
            except Exception as e:
                print(f"[{i}/{total}] ERROR   {paper['arxiv_id']} — {e}")
                failed.append(paper["arxiv_id"])

            if i < total:
                time.sleep(delay)

    print(f"\nDone. {ok}/{total} queued successfully.")
    if failed:
        print(f"{len(failed)} failed:")
        for doc_id in failed:
            print(f"  {doc_id}")
    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest arXiv papers into the research pipeline.")
    parser.add_argument("--input",  default=DEFAULT_INPUT, help="Path to papers.jsonl")
    parser.add_argument("--url",    default=DEFAULT_URL,   help="Spring Boot ingest URL")
    parser.add_argument("--delay",  default=DEFAULT_DELAY, type=float, help="Seconds between requests")
    args = parser.parse_args()

    print(f"Loading papers from: {args.input}")
    papers = load_papers(args.input)
    print(f"Found {len(papers)} papers with full_text extracted")

    if not papers:
        print("No papers to ingest. Check that full_text_extracted=True records exist.")
        sys.exit(1)

    print(f"Sending to: {args.url}  (delay={args.delay}s)\n")
    success = ingest(papers, args.url, args.delay)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
