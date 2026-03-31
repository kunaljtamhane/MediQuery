"""
Person D — PDF Full-Text Extractor (Weeks 1-2)
Downloads and extracts full text from arXiv PDFs.
Updates the JSONL file produced by arxiv_scraper.py.

Usage:
    python pdf_extractor.py --input ../../data/raw/papers.jsonl \
                            --output ../../data/raw/papers_with_text.jsonl \
                            --limit 500
"""
import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

import pymupdf  # pip install pymupdf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def download_pdf(pdf_url: str) -> bytes | None:
    """Download a PDF from arXiv. Returns bytes or None on failure."""
    try:
        # arXiv PDF URL: https://arxiv.org/pdf/{id}
        with urllib.request.urlopen(pdf_url, timeout=30) as resp:
            return resp.read()
    except Exception as e:
        log.warning(f"Failed to download {pdf_url}: {e}")
        return None


def extract_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        return "\n".join(pages).strip()
    except Exception as e:
        log.warning(f"Text extraction failed: {e}")
        return ""


def extract_all(input_path: str, output_path: str, limit: int = 500):
    """Process up to `limit` papers, adding full_text field."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    papers = []
    with open(input_path) as f:
        for line in f:
            papers.append(json.loads(line))

    log.info(f"Loaded {len(papers)} papers. Extracting full text for up to {limit}.")
    extracted = 0

    with open(output_path, "w") as out:
        for paper in papers:
            if extracted < limit and paper.get("pdf_url"):
                log.info(f"Downloading [{extracted+1}/{limit}] {paper['doc_id']}")
                pdf_bytes = download_pdf(paper["pdf_url"])
                if pdf_bytes:
                    paper["full_text"] = extract_text(pdf_bytes)
                    extracted += 1
                # Be polite to arXiv servers
                time.sleep(2)
            out.write(json.dumps(paper) + "\n")

    log.info(f"Done. {extracted} papers with full text → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../../data/raw/papers.jsonl")
    parser.add_argument("--output", default="../../data/raw/papers_with_text.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    extract_all(args.input, args.output, args.limit)
