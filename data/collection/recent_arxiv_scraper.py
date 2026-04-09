#!/usr/bin/env python3
"""
Recent arXiv medical-domain collector for the capstone project.

Writes normalized records to:
    data/raw/arxiv_papers.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List

import arxiv
from tqdm import tqdm


DEFAULT_ARXIV_QUERY = (
    '('
    'all:"artificial intelligence" OR all:"machine learning" OR '
    'all:"deep learning" OR all:"large language model"'
    ') AND ('
    'all:medical OR all:clinical OR all:healthcare OR all:medicine OR all:biomedical'
    ')'
)


@dataclass
class ArxivPaperRecord:
    source: str
    source_id: str
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published_date: str
    updated_date: str
    categories: List[str]
    primary_category: str
    pdf_url: str | None
    comment: str | None
    journal_ref: str | None
    doi: str | None
    collected_at: str
    full_text_extracted: bool = False
    full_text: str | None = None
    text_extraction_date: str | None = None
    source_url: str | None = None
    domain: str = "medical"


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


class RecentArxivCollector:
    def __init__(
        self,
        output_path: Path,
        max_results: int,
        start_date: str,
        end_date: str,
        query: str = DEFAULT_ARXIV_QUERY,
    ) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results
        self.start_date = parse_date(start_date)
        self.end_date = parse_date(end_date)
        self.query = query
        self.client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)

    def _load_existing_ids(self) -> set[str]:
        existing_ids: set[str] = set()
        if not self.output_path.exists():
            return existing_ids

        with self.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_ids.add(json.loads(line)["source_id"])
                except (KeyError, json.JSONDecodeError):
                    continue
        return existing_ids

    def _append_records(self, records: List[ArxivPaperRecord]) -> None:
        if not records:
            return

        with self.output_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _in_date_window(self, published: datetime) -> bool:
        published_date = published.date()
        return self.start_date <= published_date <= self.end_date

    def _build_record(self, result: arxiv.Result, collected_at: str) -> ArxivPaperRecord:
        arxiv_id = result.entry_id.split("/")[-1]
        return ArxivPaperRecord(
            source="arxiv",
            source_id=arxiv_id,
            arxiv_id=arxiv_id,
            title=clean_text(result.title),
            abstract=clean_text(result.summary),
            authors=[author.name for author in result.authors],
            published_date=result.published.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            updated_date=result.updated.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            categories=list(result.categories),
            primary_category=result.primary_category or (result.categories[0] if result.categories else "arxiv"),
            pdf_url=result.pdf_url,
            comment=getattr(result, "comment", None),
            journal_ref=getattr(result, "journal_ref", None),
            doi=getattr(result, "doi", None),
            collected_at=collected_at,
            source_url=result.entry_id,
        )

    def fetch_records(self) -> List[ArxivPaperRecord]:
        existing_ids = self._load_existing_ids()
        candidate_limit = max(self.max_results * 8, 500)
        search = arxiv.Search(
            query=self.query,
            max_results=candidate_limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        collected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        collected: List[ArxivPaperRecord] = []

        for result in tqdm(self.client.results(search), total=candidate_limit, desc="Fetching recent arXiv papers"):
            arxiv_id = result.entry_id.split("/")[-1]
            if arxiv_id in existing_ids:
                continue
            if not self._in_date_window(result.published):
                continue
            if not clean_text(result.title) or not clean_text(result.summary):
                continue

            collected.append(self._build_record(result, collected_at))
            if len(collected) >= self.max_results:
                break

        return collected

    def run(self) -> int:
        print("=" * 60)
        print("Recent arXiv Metadata Collection")
        print("=" * 60)
        print(f"Query: {self.query}")
        print(f"Date range: {self.start_date.isoformat()} -> {self.end_date.isoformat()}")

        records = self.fetch_records()
        self._append_records(records)
        print(f"Saved {len(records)} arXiv papers to {self.output_path}")
        return len(records)


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    today = datetime.now(timezone.utc).date()
    parser = argparse.ArgumentParser(description="Collect recent arXiv medical-domain papers into JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "raw" / "arxiv_papers.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--max_results", type=int, default=167, help="Maximum number of arXiv records to collect.")
    parser.add_argument("--start_date", default="2024-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", default=today.isoformat(), help="End date in YYYY-MM-DD format.")
    parser.add_argument("--query", default=DEFAULT_ARXIV_QUERY, help="arXiv query string.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    collector = RecentArxivCollector(
        output_path=args.output,
        max_results=args.max_results,
        start_date=args.start_date,
        end_date=args.end_date,
        query=args.query,
    )
    collector.run()


if __name__ == "__main__":
    main()
