#!/usr/bin/env python3
"""
arXiv metadata collector for the capstone project.

Target repo path:
    data/collection/arxiv_scraper.py

Collects paper metadata from arXiv and saves it as JSONL.
This file focuses only on metadata collection.
PDF download and extraction are handled separately in pdf_extractor.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import arxiv
from tqdm import tqdm


@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published_date: str
    updated_date: str
    categories: List[str]
    primary_category: str
    pdf_url: str
    comment: str | None
    journal_ref: str | None
    doi: str | None
    collected_at: str
    full_text_extracted: bool = False
    full_text: str | None = None
    text_extraction_date: str | None = None


class ArxivScraper:
    def __init__(self, output_dir: str = "./data/collection/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.papers_file = self.output_dir / "papers.jsonl"
        self.categories = ["cs.CL", "cs.AI", "cs.LG"]

    def _build_client(self, page_size: int = 10, delay_seconds: float = 5.0, num_retries: int = 5) -> arxiv.Client:
        """
        Build arXiv API client.

        Smaller page_size helps reduce risk of request failures during testing.
        """
        return arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )

    def _save_batch(self, papers: List[PaperRecord]) -> None:
        """Append a batch of paper records to JSONL."""
        if not papers:
            return

        with self.papers_file.open("a", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")

    def load_papers(self) -> List[Dict]:
        """Load all saved paper records from JSONL."""
        papers: List[Dict] = []
        if not self.papers_file.exists():
            return papers

        with self.papers_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    papers.append(json.loads(line))
        return papers

    def search_papers(
        self,
        category: str,
        max_results: int = 1000,
        start_year: int = 2020,
        max_attempts: int = 3,
    ) -> int:
        """
        Collect metadata for one category and append it to papers.jsonl.

        Returns:
            int: number of papers saved in this call
        """
        print(f"\nSearching category: {category}")

        search_query = f"cat:{category} AND submittedDate:[{start_year}01010000 TO 300001012359]"

        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        total_saved = 0
        batch: List[PaperRecord] = []

        for attempt in range(1, max_attempts + 1):
            client = self._build_client(page_size=min(10, max_results), delay_seconds=5.0, num_retries=5)

            try:
                for result in tqdm(
                    client.results(search),
                    total=max_results,
                    desc=f"Fetching {category} (attempt {attempt}/{max_attempts})",
                ):
                    paper = PaperRecord(
                        arxiv_id=result.entry_id.split("/")[-1],
                        title=result.title,
                        abstract=result.summary,
                        authors=[author.name for author in result.authors],
                        published_date=result.published.isoformat(),
                        updated_date=result.updated.isoformat(),
                        categories=result.categories,
                        primary_category=result.primary_category,
                        pdf_url=result.pdf_url,
                        comment=getattr(result, "comment", None),
                        journal_ref=getattr(result, "journal_ref", None),
                        doi=getattr(result, "doi", None),
                        collected_at=datetime.now().isoformat(),
                    )
                    batch.append(paper)

                    if len(batch) >= 50:
                        self._save_batch(batch)
                        total_saved += len(batch)
                        batch = []

                # Success: save remaining records and stop retrying
                if batch:
                    self._save_batch(batch)
                    total_saved += len(batch)
                    batch = []

                print(f"Saved {total_saved} papers for {category}")
                return total_saved

            except Exception as exc:
                print(f"Warning: attempt {attempt}/{max_attempts} failed for {category}: {exc}")

                # Save whatever was already collected before retrying
                if batch:
                    self._save_batch(batch)
                    total_saved += len(batch)
                    batch = []

                if attempt < max_attempts:
                    wait_time = 10 * attempt
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)

        print(f"Finished with retries. Saved {total_saved} papers for {category}")
        return total_saved

    def get_statistics(self) -> Dict:
        """Return simple collection stats."""
        papers = self.load_papers()
        papers_with_text = sum(1 for p in papers if p.get("full_text_extracted", False))

        categories_count: Dict[str, int] = {}
        for paper in papers:
            for cat in paper.get("categories", []):
                categories_count[cat] = categories_count.get(cat, 0) + 1

        return {
            "total_papers": len(papers),
            "papers_with_full_text": papers_with_text,
            "categories_distribution": categories_count,
            "date_range": {
                "earliest": min((p["published_date"] for p in papers), default=None),
                "latest": max((p["published_date"] for p in papers), default=None),
            },
        }


def main() -> None:
    scraper = ArxivScraper(output_dir="./data/collection/output")

    print("=" * 60)
    print("arXiv Metadata Collection")
    print("=" * 60)

    total = 0
    for category in scraper.categories:
        total += scraper.search_papers(
            category=category,
            max_results=1000,
            start_year=2020,
            max_attempts=3,
        )
        time.sleep(2)

    stats = scraper.get_statistics()
    print("\nCollection complete")
    print(json.dumps(stats, indent=2))
    print(f"\nSaved about {total} new records to {scraper.papers_file}")


if __name__ == "__main__":
    main()