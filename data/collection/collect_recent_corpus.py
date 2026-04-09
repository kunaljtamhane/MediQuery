#!/usr/bin/env python3
"""
Collect a recent 500-paper medical corpus split across arXiv, PubMed, and medRxiv.

Default split:
    arXiv   167
    PubMed  167
    medRxiv 166

Default date window:
    2024-01-01 through current date
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from medrxiv_scraper import MedRxivCollector
from pubmed_scraper import PubMedCollector
from recent_arxiv_scraper import DEFAULT_ARXIV_QUERY, RecentArxivCollector


DEFAULT_PUBMED_QUERY = (
    '("Artificial Intelligence"[MeSH Terms] OR "Machine Learning"[MeSH Terms] '
    'OR "machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract] '
    'OR "large language model"[Title/Abstract] OR "large language models"[Title/Abstract] '
    'OR "generative ai"[Title/Abstract] OR "clinical ai"[Title/Abstract])'
)
PUBMED_FALLBACK_QUERIES = [
    DEFAULT_PUBMED_QUERY,
    '("Artificial Intelligence"[MeSH Terms] OR "Machine Learning"[MeSH Terms] OR "Neural Networks, Computer"[MeSH Terms])',
    '("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract] OR "artificial intelligence"[Title/Abstract])',
]
DEFAULT_MEDRXIV_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "large language model",
    "clinical",
    "medical",
    "healthcare",
    "biomedical",
]


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    today = datetime.now(timezone.utc).date().isoformat()

    parser = argparse.ArgumentParser(description="Collect a recent multi-source medical corpus.")
    parser.add_argument("--total", type=int, default=500, help="Target total across all three sources.")
    parser.add_argument("--arxiv_count", type=int, default=167, help="Recent arXiv record target.")
    parser.add_argument("--pubmed_count", type=int, default=167, help="Recent PubMed record target.")
    parser.add_argument("--medrxiv_count", type=int, default=166, help="Recent medRxiv record target.")
    parser.add_argument("--start_date", default="2024-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", default=today, help="End date in YYYY-MM-DD format.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=project_root / "data" / "raw",
        help="Directory for arxiv_papers.jsonl, pubmed_papers.jsonl, and medrxiv_papers.jsonl.",
    )
    parser.add_argument("--arxiv_query", default=DEFAULT_ARXIV_QUERY, help="arXiv query string.")
    parser.add_argument("--pubmed_query", default=DEFAULT_PUBMED_QUERY, help="PubMed query string.")
    parser.add_argument(
        "--medrxiv_keywords",
        nargs="*",
        default=DEFAULT_MEDRXIV_KEYWORDS,
        help="Keyword filters for medRxiv title, abstract, and category fields.",
    )
    return parser


def collect_pubmed_with_fallbacks(
    output_path: Path,
    target_count: int,
    start_date: str,
    end_date: str,
    primary_query: str,
) -> int:
    total_saved = 0
    queries = [primary_query] + [query for query in PUBMED_FALLBACK_QUERIES if query != primary_query]

    for query in queries:
        remaining = target_count - total_saved
        if remaining <= 0:
            break

        print(f"\nPubMed attempt with query: {query}")
        saved = PubMedCollector(
            output_path=output_path,
            query=query,
            max_results=remaining,
            start_date=start_date.replace("-", "/"),
            end_date=end_date.replace("-", "/"),
        ).run()
        total_saved += saved

    return total_saved


def main() -> None:
    args = build_arg_parser().parse_args()
    total_requested = args.arxiv_count + args.pubmed_count + args.medrxiv_count
    if total_requested != args.total:
        raise SystemExit(
            f"Split mismatch: arXiv ({args.arxiv_count}) + PubMed ({args.pubmed_count}) + "
            f"medRxiv ({args.medrxiv_count}) = {total_requested}, not total={args.total}"
        )

    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Recent Multi-Source Corpus Collection")
    print("=" * 60)
    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(
        f"Split: arXiv={args.arxiv_count}, PubMed={args.pubmed_count}, "
        f"medRxiv={args.medrxiv_count}, total={args.total}"
    )

    arxiv_count = RecentArxivCollector(
        output_path=raw_dir / "arxiv_papers.jsonl",
        max_results=args.arxiv_count,
        start_date=args.start_date,
        end_date=args.end_date,
        query=args.arxiv_query,
    ).run()

    pubmed_count = collect_pubmed_with_fallbacks(
        output_path=raw_dir / "pubmed_papers.jsonl",
        target_count=args.pubmed_count,
        start_date=args.start_date,
        end_date=args.end_date,
        primary_query=args.pubmed_query,
    )

    medrxiv_count = MedRxivCollector(
        output_path=raw_dir / "medrxiv_papers.jsonl",
        max_results=args.medrxiv_count,
        start_date=args.start_date,
        end_date=args.end_date,
        keywords=args.medrxiv_keywords,
    ).run()

    print("\nCollection summary")
    print(f"  arXiv:   {arxiv_count}")
    print(f"  PubMed:  {pubmed_count}")
    print(f"  medRxiv: {medrxiv_count}")
    print(f"  Total:   {arxiv_count + pubmed_count + medrxiv_count}")


if __name__ == "__main__":
    main()
