#!/usr/bin/env python3
"""
PDF downloader and text extractor for the capstone project.

Target repo path:
    data/collection/pdf_extractor.py

Reads papers.jsonl, downloads PDFs, extracts text with PyMuPDF,
and rewrites papers.jsonl with full_text fields filled in.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import fitz
import requests
from tqdm import tqdm


class PDFExtractor:
    def __init__(self, data_dir: str = "./data/collection/output"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.papers_file = self.data_dir / "papers.jsonl"
        self.pdfs_dir = self.data_dir / "pdfs"
        self.pdfs_dir.mkdir(exist_ok=True)

    def load_papers(self) -> List[Dict]:
        papers: List[Dict] = []
        if not self.papers_file.exists():
            return papers

        with self.papers_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    papers.append(json.loads(line))
        return papers

    def rewrite_jsonl(self, papers: List[Dict]) -> None:
        with self.papers_file.open("w", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    def download_pdf(self, paper: Dict) -> bool:
        arxiv_id = paper["arxiv_id"]
        pdf_path = self.pdfs_dir / f"{arxiv_id.replace('/', '_')}.pdf"

        if pdf_path.exists():
            return True

        try:
            response = requests.get(paper["pdf_url"], timeout=30)
            response.raise_for_status()
            with pdf_path.open("wb") as f:
                f.write(response.content)
            time.sleep(3)
            return True
        except Exception as exc:
            print(f"Warning: failed to download {arxiv_id}: {exc}")
            return False

    def extract_text_from_pdf(self, paper: Dict) -> Optional[str]:
        arxiv_id = paper["arxiv_id"]
        pdf_path = self.pdfs_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        if not pdf_path.exists():
            return None

        try:
            doc = fitz.open(pdf_path)
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(pages)
        except Exception as exc:
            print(f"Warning: failed to extract text for {arxiv_id}: {exc}")
            return None

    def process_pdfs(self, num_pdfs: int = 500) -> int:
        papers = self.load_papers()
        if not papers:
            print("No papers found. Run arxiv_scraper.py first.")
            return 0

        pending = [p for p in papers if not p.get("full_text_extracted", False)][:num_pdfs]
        print(f"Processing {len(pending)} papers for PDF download and extraction")

        extracted = 0
        for paper in tqdm(pending, desc="Processing PDFs"):
            if self.download_pdf(paper):
                full_text = self.extract_text_from_pdf(paper)
                if full_text:
                    paper["full_text"] = full_text
                    paper["full_text_extracted"] = True
                    paper["text_extraction_date"] = datetime.now().isoformat()
                    extracted += 1

        self.rewrite_jsonl(papers)
        print(f"Finished. Extracted text for {extracted} papers.")
        return extracted


def main() -> None:
    extractor = PDFExtractor(data_dir="./data/collection/output")
    extractor.process_pdfs(num_pdfs=500)


if __name__ == "__main__":
    main()
