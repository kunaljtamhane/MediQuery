#!/usr/bin/env python3
"""
Collect a recent 500-paper medical corpus split across PubMed, arXiv, and medRxiv.

Default split:
    PubMed  250
    arXiv   125
    medRxiv 125

This version seeds retrieval from the PubMedQA dataset and applies semantic
ranking before writing JSONL outputs. No PDFs are downloaded.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from semantic_collection import (
    DEFAULT_ARXIV_COUNT,
    DEFAULT_MEDRXIV_COUNT,
    DEFAULT_PUBMED_COUNT,
    collect_semantic_corpus,
)


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    today = datetime.now(timezone.utc).date()
    parser = argparse.ArgumentParser(description="Collect a semantic multi-source medical corpus.")
    parser.add_argument("--total", type=int, default=500, help="Target total across all three sources.")
    parser.add_argument("--pubmed_count", type=int, default=DEFAULT_PUBMED_COUNT, help="PubMed record target.")
    parser.add_argument("--arxiv_count", type=int, default=DEFAULT_ARXIV_COUNT, help="arXiv record target.")
    parser.add_argument("--medrxiv_count", type=int, default=DEFAULT_MEDRXIV_COUNT, help="medRxiv record target.")
    parser.add_argument(
        "--start_date",
        default=(today - timedelta(days=365 * 2)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument("--end_date", default=today.isoformat(), help="End date in YYYY-MM-DD format.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=project_root / "data" / "raw",
        help="Directory for arxiv_papers.jsonl, pubmed_papers.jsonl, medrxiv_papers.jsonl, and papers.jsonl.",
    )
    parser.add_argument(
        "--top_k_keywords",
        type=int,
        default=64,
        help="How many PubMedQA-derived keyword phrases to use for the query bank.",
    )
    parser.add_argument(
        "--pubmed_email",
        default=None,
        help="Optional email for NCBI E-utilities etiquette.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    total_requested = args.pubmed_count + args.arxiv_count + args.medrxiv_count
    if total_requested != args.total:
        raise SystemExit(
            f"Split mismatch: PubMed ({args.pubmed_count}) + arXiv ({args.arxiv_count}) + "
            f"medRxiv ({args.medrxiv_count}) = {total_requested}, not total={args.total}"
        )

    print("=" * 60)
    print("Semantic Multi-Source Corpus Collection")
    print("=" * 60)
    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(
        f"Split: PubMed={args.pubmed_count}, arXiv={args.arxiv_count}, "
        f"medRxiv={args.medrxiv_count}, total={args.total}"
    )
    print("PDF download: disabled")

    summary = collect_semantic_corpus(
        raw_dir=args.raw_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        pubmed_count=args.pubmed_count,
        arxiv_count=args.arxiv_count,
        medrxiv_count=args.medrxiv_count,
        top_k_keywords=args.top_k_keywords,
        pubmed_email=args.pubmed_email,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
