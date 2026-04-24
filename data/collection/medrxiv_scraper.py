#!/usr/bin/env python3
"""
medRxiv metadata collector for the capstone project.

Writes normalized records to:
    data/raw/medrxiv_papers.jsonl

This collector uses the medRxiv details API:
https://api.medrxiv.org/details/medrxiv/help
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

from env_loader import configure_requests_session, load_env_file
from medrxiv_content import fetch_jats_full_text, get_jats_xml_path


MEDRXIV_API_BASE = "https://api.medrxiv.org/details/medrxiv"


@dataclass
class MedRxivPaperRecord:
    source: str
    source_id: str
    medrxiv_id: str
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
    version: str | None = None
    license: str | None = None
    jats_xml_path: str | None = None
    published_journal_ref: str | None = None


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def parse_authors(value: str | None) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    if ";" in text:
        return [author.strip() for author in text.split(";") if author.strip()]
    return [author.strip() for author in text.split(",") if author.strip()]


def to_iso8601(date_value: str | None) -> str | None:
    if not date_value:
        return None
    try:
        return datetime.strptime(date_value, "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00Z")
    except ValueError:
        return None


class MedRxivCollector:
    def __init__(
        self,
        output_path: Path,
        max_results: int,
        start_date: str,
        end_date: str,
        keywords: List[str] | None = None,
    ) -> None:
        load_env_file(Path(__file__).resolve().parents[2] / ".env")
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results
        self.start_date = start_date
        self.end_date = end_date
        self.keywords = [keyword.lower() for keyword in (keywords or []) if keyword.strip()]
        self.session = configure_requests_session(requests.Session())

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

    def _append_records(self, records: List[MedRxivPaperRecord]) -> None:
        if not records:
            return

        with self.output_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _request_page(self, cursor: int) -> dict:
        url = f"{MEDRXIV_API_BASE}/{self.start_date}/{self.end_date}/{cursor}/json"
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                time.sleep(0.2)
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(attempt * 2)
        raise RuntimeError(f"medRxiv request failed: {last_error}")

    def _matches_keywords(self, item: dict) -> bool:
        if not self.keywords:
            return True

        haystack = " ".join(
            [
                clean_text(item.get("title")),
                clean_text(item.get("abstract")),
                clean_text(item.get("category")),
            ]
        ).lower()
        return any(keyword in haystack for keyword in self.keywords)

    def _build_record(self, item: dict, collected_at: str) -> MedRxivPaperRecord | None:
        doi = clean_text(item.get("doi"))
        title = clean_text(item.get("title"))
        abstract = clean_text(item.get("abstract"))
        version = clean_text(item.get("version")) or None
        category = clean_text(item.get("category")) or "medical"
        license_name = clean_text(item.get("license")) or None
        published_ref = clean_text(item.get("published")) or None
        jats_xml_path = get_jats_xml_path(item)

        if not doi or not title or not abstract:
            return None

        landing_page = f"https://www.medrxiv.org/content/{doi}v{version}" if version else f"https://www.medrxiv.org/content/{doi}"
        pdf_url = f"{landing_page}.full.pdf"
        full_text, resolved_jats_url, text_error = fetch_jats_full_text(
            self.session,
            jats_xml_path,
            timeout=60,
        )
        comment_parts = [clean_text(item.get("type")) or None, f"license={license_name}" if license_name else None]
        if text_error and jats_xml_path:
            comment_parts.append(f"jats_error={text_error}")
        comment = ", ".join(part for part in comment_parts if part) or None
        published_date = to_iso8601(clean_text(item.get("date"))) or collected_at

        return MedRxivPaperRecord(
            source="medrxiv",
            source_id=doi,
            medrxiv_id=doi,
            title=title,
            abstract=abstract,
            authors=parse_authors(item.get("authors")),
            published_date=published_date,
            updated_date=published_date,
            categories=[category],
            primary_category=category,
            pdf_url=pdf_url,
            comment=comment,
            journal_ref=published_ref,
            doi=doi,
            collected_at=collected_at,
            full_text_extracted=bool(full_text),
            full_text=full_text,
            text_extraction_date=collected_at if full_text else None,
            source_url=landing_page,
            version=version,
            license=license_name,
            jats_xml_path=resolved_jats_url or jats_xml_path,
            published_journal_ref=published_ref,
        )

    def fetch_records(self, existing_ids: set[str]) -> List[MedRxivPaperRecord]:
        collected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        cursor = 0
        collected: List[MedRxivPaperRecord] = []

        with tqdm(total=self.max_results, desc="Fetching medRxiv papers") as progress:
            while len(collected) < self.max_results:
                payload = self._request_page(cursor)
                page_items = payload.get("collection", [])
                if not page_items:
                    break

                new_records = []
                for item in page_items:
                    if not self._matches_keywords(item):
                        continue
                    record = self._build_record(item, collected_at)
                    if record is not None and record.source_id not in existing_ids:
                        new_records.append(record)

                remaining = self.max_results - len(collected)
                page_records = new_records[:remaining]
                collected.extend(page_records)
                progress.update(len(page_records))

                if len(page_items) < 100:
                    break
                cursor += 100

        return collected

    def run(self) -> int:
        print("=" * 60)
        print("medRxiv Metadata Collection")
        print("=" * 60)
        print(f"Date range: {self.start_date} -> {self.end_date}")
        if self.keywords:
            print(f"Keyword filter: {', '.join(self.keywords)}")

        existing_ids = self._load_existing_ids()
        records = self.fetch_records(existing_ids)
        self._append_records(records)
        print(f"Saved {len(records)} medRxiv papers to {self.output_path}")
        return len(records)


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    today = datetime.now(timezone.utc).date()
    parser = argparse.ArgumentParser(description="Collect medRxiv medical-domain papers into JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "raw" / "medrxiv_papers.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--max_results", type=int, default=1000, help="Maximum number of medRxiv records to collect.")
    parser.add_argument(
        "--start_date",
        default=(today - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end_date",
        default=today.strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=[],
        help="Optional keyword filters applied to title, abstract, and category.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    collector = MedRxivCollector(
        output_path=args.output,
        max_results=args.max_results,
        start_date=args.start_date,
        end_date=args.end_date,
        keywords=args.keywords,
    )
    collector.run()


if __name__ == "__main__":
    main()
