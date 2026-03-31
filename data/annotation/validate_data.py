#!/usr/bin/env python3
"""
Validation script for collected arXiv data.

Suggested repo path:
    data/annotation/validate_data.py
or keep it under data/collection if your team prefers.
"""

import json
from collections import Counter
from pathlib import Path


class DataValidator:
    def __init__(self, data_dir: str = "./data/collection/output"):
        self.data_dir = Path(data_dir)
        self.papers_file = self.data_dir / "papers.jsonl"
        self.pdfs_dir = self.data_dir / "pdfs"

    def load_papers(self):
        papers = []
        if not self.papers_file.exists():
            print("papers.jsonl not found")
            return papers
        with self.papers_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    papers.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"Bad JSON on line {i}: {exc}")
        return papers

    def validate_paper_schema(self, paper: dict):
        required = [
            "arxiv_id",
            "title",
            "abstract",
            "authors",
            "published_date",
            "categories",
            "pdf_url",
        ]
        return [field for field in required if field not in paper]

    def run_validation(self):
        print("=" * 70)
        print("DATA VALIDATION REPORT")
        print("=" * 70)

        papers = self.load_papers()
        if not papers:
            print("No papers found")
            return

        print(f"Total papers: {len(papers)}")
        full_text_count = sum(1 for p in papers if p.get("full_text_extracted", False))
        print(f"Papers with full text: {full_text_count}")

        categories = Counter()
        primary_categories = Counter()
        for paper in papers:
            categories.update(paper.get("categories", []))
            primary_categories.update([paper.get("primary_category", "unknown")])

        print("\nTop categories:")
        for cat, count in categories.most_common(10):
            print(f"  {cat}: {count}")

        print("\nPrimary categories:")
        for cat, count in primary_categories.most_common():
            print(f"  {cat}: {count}")

        schema_errors = 0
        for paper in papers[:100]:
            missing = self.validate_paper_schema(paper)
            if missing:
                schema_errors += 1
        print(f"\nSchema issues in first 100 records: {schema_errors}")

        pdf_count = len(list(self.pdfs_dir.glob("*.pdf"))) if self.pdfs_dir.exists() else 0
        print(f"PDF files on disk: {pdf_count}")

        if papers:
            dates = [p.get("published_date") for p in papers if p.get("published_date")]
            if dates:
                print(f"Earliest published date: {min(dates)[:10]}")
                print(f"Latest published date: {max(dates)[:10]}")

        print("\nSuccess criteria:")
        print(f"  3000+ papers: {'yes' if len(papers) >= 3000 else 'no'}")
        print(f"  500+ full text papers: {'yes' if full_text_count >= 500 else 'no'}")
        print(f"  500+ PDFs: {'yes' if pdf_count >= 500 else 'no'}")
        print(f"  valid schema: {'yes' if schema_errors == 0 else 'no'}")


def main():
    DataValidator(data_dir="./data/collection/output").run_validation()


if __name__ == "__main__":
    main()
