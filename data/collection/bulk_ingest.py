"""
Bulk ingestion script for the latest multi-source JSONL corpus.

Accepts the combined corpus from data/raw/papers.jsonl and posts each document
to the Spring Boot /ingest endpoint, which publishes the payload to Kafka.
"""
import argparse
import json
import time
import sys
import httpx

DEFAULT_INPUT = "../../data/raw/papers.jsonl"
DEFAULT_URL   = "http://localhost:8080/ingest"
DEFAULT_DELAY = 0.1


def pick_text(paper: dict) -> str:
    for key in ("full_text", "text", "abstract", "summary", "title"):
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pick_doc_id(paper: dict) -> str:
    for key in ("doc_id", "source_id", "pubmed_id", "arxiv_id", "medrxiv_id", "doi"):
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source = (paper.get("source") or "doc").strip()
    title = (paper.get("title") or "untitled").strip().replace(" ", "_")
    return f"{source}:{title[:80]}"


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
            text = pick_text(d)
            title = (d.get("title") or "").strip()
            if text and title:
                d["_ingest_text"] = text
                d["_ingest_doc_id"] = pick_doc_id(d)
                papers.append(d)
    return papers


def build_payload(paper: dict) -> dict:
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        authors = ", ".join(authors)
    url = paper.get("source_url") or paper.get("url") or paper.get("pdf_url") or ""
    return {
        "docId":         paper["_ingest_doc_id"],
        "title":         paper["title"],
        "text":          paper["_ingest_text"],
        "authors":       authors,
        "publishedDate": paper.get("published_date", ""),
        "arxivUrl":      url,
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
                print(f"[{i}/{total}] queued  {paper['_ingest_doc_id']}")
            except httpx.HTTPStatusError as e:
                print(f"[{i}/{total}] ERROR   {paper['_ingest_doc_id']} — HTTP {e.response.status_code}: {e.response.text[:120]}")
                failed.append(paper["_ingest_doc_id"])
            except Exception as e:
                print(f"[{i}/{total}] ERROR   {paper['_ingest_doc_id']} — {e}")
                failed.append(paper["_ingest_doc_id"])

            if i < total:
                time.sleep(delay)

    print(f"\nDone. {ok}/{total} queued successfully.")
    if failed:
        print(f"{len(failed)} failed:")
        for doc_id in failed:
            print(f"  {doc_id}")
    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest the latest multi-source papers into the research pipeline.")
    parser.add_argument("--input",  default=DEFAULT_INPUT, help="Path to papers.jsonl")
    parser.add_argument("--url",    default=DEFAULT_URL,   help="Spring Boot ingest URL")
    parser.add_argument("--delay",  default=DEFAULT_DELAY, type=float, help="Seconds between requests")
    args = parser.parse_args()

    print(f"Loading papers from: {args.input}")
    papers = load_papers(args.input)
    print(f"Found {len(papers)} papers with ingestable text")

    if not papers:
        print("No papers to ingest. Check that title/text or abstract fields exist in the JSONL.")
        sys.exit(1)

    print(f"Sending to: {args.url}  (delay={args.delay}s)\n")
    success = ingest(papers, args.url, args.delay)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
