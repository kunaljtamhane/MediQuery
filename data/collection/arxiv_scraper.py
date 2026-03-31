"""
Person D — arXiv Paper Scraper (Weeks 1-2)
Collects papers from cs.CL, cs.AI, cs.LG categories and outputs JSONL.

Usage:
    python arxiv_scraper.py --output ../../data/raw/papers.jsonl --max_results 3000
"""
import argparse
import json
import logging
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"

# Target categories from the execution plan
CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]


def fetch_batch(category: str, start: int, batch_size: int = 100) -> list[dict]:
    """Fetch one batch of papers from arXiv API."""
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": batch_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_data = resp.read()
    except Exception as e:
        log.error(f"API error for {category} start={start}: {e}")
        return []

    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall(f"{NS}entry"):
        try:
            paper = {
                "doc_id": entry.find(f"{NS}id").text.split("/abs/")[-1].replace("/", "_"),
                "arxiv_id": entry.find(f"{NS}id").text.split("/abs/")[-1],
                "title": entry.find(f"{NS}title").text.strip().replace("\n", " "),
                "abstract": entry.find(f"{NS}summary").text.strip().replace("\n", " "),
                "authors": [a.find(f"{NS}name").text for a in entry.findall(f"{NS}author")],
                "published": entry.find(f"{NS}published").text[:10],
                "url": entry.find(f"{NS}id").text.strip(),
                "pdf_url": next(
                    (l.get("href") for l in entry.findall(f"{NS}link") if l.get("title") == "pdf"),
                    None
                ),
                "category": category,
                "full_text": None,  # Filled in by pdf_extractor.py
            }
            papers.append(paper)
        except Exception as e:
            log.warning(f"Skipping malformed entry: {e}")

    return papers


def scrape(output_path: str, max_results: int = 3000):
    """Collect papers across all categories and write to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    per_category = max_results // len(CATEGORIES)
    seen_ids = set()
    total = 0

    with open(output_path, "w") as out:
        for cat in CATEGORIES:
            log.info(f"Scraping category={cat}, target={per_category} papers")
            start = 0
            batch_size = 100

            while start < per_category:
                papers = fetch_batch(cat, start, batch_size)
                if not papers:
                    break

                for paper in papers:
                    if paper["doc_id"] not in seen_ids:
                        seen_ids.add(paper["doc_id"])
                        out.write(json.dumps(paper) + "\n")
                        total += 1

                log.info(f"  {cat}: fetched {start + len(papers)} / {per_category}")
                start += batch_size

                # Respect arXiv rate limit: 3 seconds between requests
                time.sleep(3)

    log.info(f"Done. Total papers written: {total} → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../../data/raw/papers.jsonl")
    parser.add_argument("--max_results", type=int, default=3000)
    args = parser.parse_args()
    scrape(args.output, args.max_results)
