#!/usr/bin/env python3
"""
PubMed metadata collector for the capstone project.

Writes normalized records to:
    data/raw/pubmed_papers.jsonl

This collector uses NCBI E-utilities:
https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import requests
from tqdm import tqdm

from env_loader import configure_requests_session, load_env_file


NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERY = (
    '("Artificial Intelligence"[MeSH Terms] OR "Machine Learning"[MeSH Terms] '
    'OR "machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract] '
    'OR "large language model"[Title/Abstract] OR "large language models"[Title/Abstract] '
    'OR "generative ai"[Title/Abstract] OR "clinical ai"[Title/Abstract])'
)
MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass
class PubMedPaperRecord:
    source: str
    source_id: str
    pubmed_id: str
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
    pmc_id: str | None = None


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def ensure_iso_date(year: str | None, month: str | None = None, day: str | None = None) -> str | None:
    if not year or not year.isdigit():
        return None

    month_value = 1
    day_value = 1

    if month:
        month_clean = month.strip().lower()
        if month_clean.isdigit():
            month_value = max(1, min(12, int(month_clean)))
        else:
            month_value = MONTH_MAP.get(month_clean[:4], 1)

    if day and day.isdigit():
        day_value = max(1, min(31, int(day)))

    return f"{int(year):04d}-{month_value:02d}-{day_value:02d}T00:00:00Z"


class PubMedCollector:
    def __init__(
        self,
        output_path: Path,
        query: str,
        max_results: int,
        start_date: str | None,
        end_date: str | None,
        page_size: int = 200,
        fetch_batch_size: int = 100,
    ) -> None:
        load_env_file(Path(__file__).resolve().parents[2] / ".env")
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.query = query
        self.max_results = max_results
        self.start_date = start_date
        self.end_date = end_date
        self.page_size = page_size
        self.fetch_batch_size = fetch_batch_size
        self.session = configure_requests_session(requests.Session())
        self.email = os.getenv("NCBI_EMAIL", "")
        self.api_key = os.getenv("NCBI_API_KEY", "")
        self.tool_name = os.getenv("NCBI_TOOL", "capstone-pubmed-collector")
        self.delay_seconds = 0.11 if self.api_key else 0.34

    def _request(self, endpoint: str, params: dict, timeout: int = 60) -> requests.Response:
        request_params = {"tool": self.tool_name, **params}
        if self.email:
            request_params["email"] = self.email
        if self.api_key:
            request_params["api_key"] = self.api_key

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.session.get(f"{NCBI_BASE_URL}/{endpoint}", params=request_params, timeout=timeout)
                response.raise_for_status()
                time.sleep(self.delay_seconds)
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(attempt * 2)

        raise RuntimeError(f"PubMed request failed: {last_error}")

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

    def _append_records(self, records: List[PubMedPaperRecord]) -> None:
        if not records:
            return

        with self.output_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def search_ids(self, existing_ids: set[str] | None = None) -> List[str]:
        ids: List[str] = []
        seen_ids = set(existing_ids or set())
        retstart = 0

        while len(ids) < self.max_results:
            params = {
                "db": "pubmed",
                "term": self.query,
                "retmode": "json",
                "retmax": min(self.page_size, self.max_results - len(ids)),
                "retstart": retstart,
                "sort": "pub date",
            }
            if self.start_date and self.end_date:
                params.update(
                    {
                        "datetype": "pdat",
                        "mindate": self.start_date,
                        "maxdate": self.end_date,
                    }
                )

            response = self._request("esearch.fcgi", params)
            payload = response.json()["esearchresult"]
            batch_ids = payload.get("idlist", [])
            if not batch_ids:
                break

            for paper_id in batch_ids:
                if paper_id in seen_ids:
                    continue
                ids.append(paper_id)
                seen_ids.add(paper_id)
                if len(ids) >= self.max_results:
                    break

            retstart += len(batch_ids)

            if retstart >= int(payload.get("count", "0")):
                break

        return ids[: self.max_results]

    def fetch_records(self, ids: List[str]) -> List[PubMedPaperRecord]:
        collected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        records: List[PubMedPaperRecord] = []

        for batch in tqdm(list(chunked(ids, self.fetch_batch_size)), desc="Fetching PubMed batches"):
            response = self._request(
                "efetch.fcgi",
                {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "xml",
                    "rettype": "abstract",
                },
                timeout=120,
            )
            root = ET.fromstring(response.text)
            for article in root.findall(".//PubmedArticle"):
                record = self._parse_article(article, collected_at)
                if record is not None:
                    records.append(record)

        return records

    def _parse_article(self, article: ET.Element, collected_at: str) -> PubMedPaperRecord | None:
        pmid = clean_text(article.findtext(".//MedlineCitation/PMID"))
        title = clean_text("".join(article.find(".//Article/ArticleTitle").itertext())) if article.find(".//Article/ArticleTitle") is not None else ""
        abstract = self._extract_abstract(article)

        if not pmid or not title or not abstract:
            return None

        authors = self._extract_authors(article)
        categories = self._extract_categories(article)
        primary_category = categories[0] if categories else "medical"
        published_date = self._extract_published_date(article)
        updated_date = self._extract_updated_date(article) or published_date

        doi = None
        pmc_id = None
        for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            id_type = article_id.attrib.get("IdType", "").lower()
            value = clean_text(article_id.text)
            if id_type == "doi" and value:
                doi = value
            elif id_type in {"pmc", "pmcid"} and value:
                pmc_id = value if value.startswith("PMC") else f"PMC{value}"

        publication_types = [
            clean_text(node.text)
            for node in article.findall(".//PublicationTypeList/PublicationType")
            if clean_text(node.text)
        ]
        comment = ", ".join(publication_types) if publication_types else None

        return PubMedPaperRecord(
            source="pubmed",
            source_id=pmid,
            pubmed_id=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            published_date=published_date or collected_at,
            updated_date=updated_date or published_date or collected_at,
            categories=categories,
            primary_category=primary_category,
            pdf_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/" if pmc_id else None,
            comment=comment,
            journal_ref=self._build_journal_reference(article),
            doi=doi,
            collected_at=collected_at,
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            pmc_id=pmc_id,
        )

    def _extract_abstract(self, article: ET.Element) -> str:
        pieces: List[str] = []
        for abstract_text in article.findall(".//Article/Abstract/AbstractText"):
            label = clean_text(abstract_text.attrib.get("Label"))
            text = clean_text("".join(abstract_text.itertext()))
            if not text:
                continue
            if label:
                pieces.append(f"{label}: {text}")
            else:
                pieces.append(text)
        return "\n".join(pieces)

    def _extract_authors(self, article: ET.Element) -> List[str]:
        authors: List[str] = []
        for author in article.findall(".//Article/AuthorList/Author"):
            collective_name = clean_text(author.findtext("CollectiveName"))
            if collective_name:
                authors.append(collective_name)
                continue

            last_name = clean_text(author.findtext("LastName"))
            fore_name = clean_text(author.findtext("ForeName"))
            initials = clean_text(author.findtext("Initials"))
            if last_name and fore_name:
                authors.append(f"{fore_name} {last_name}")
            elif last_name and initials:
                authors.append(f"{initials} {last_name}")
            elif last_name:
                authors.append(last_name)
        return authors

    def _extract_categories(self, article: ET.Element) -> List[str]:
        mesh_terms = [
            clean_text(node.findtext("DescriptorName"))
            for node in article.findall(".//MeshHeadingList/MeshHeading")
            if clean_text(node.findtext("DescriptorName"))
        ]
        keywords = [
            clean_text(node.text)
            for node in article.findall(".//KeywordList/Keyword")
            if clean_text(node.text)
        ]

        seen: set[str] = set()
        categories: List[str] = []
        for value in mesh_terms + keywords:
            if value and value not in seen:
                categories.append(value)
                seen.add(value)
        return categories or ["medical"]

    def _extract_published_date(self, article: ET.Element) -> str | None:
        article_date = article.find(".//Article/ArticleDate")
        if article_date is not None:
            return ensure_iso_date(
                clean_text(article_date.findtext("Year")),
                clean_text(article_date.findtext("Month")),
                clean_text(article_date.findtext("Day")),
            )

        pub_date = article.find(".//JournalIssue/PubDate")
        if pub_date is not None:
            return ensure_iso_date(
                clean_text(pub_date.findtext("Year")),
                clean_text(pub_date.findtext("Month")),
                clean_text(pub_date.findtext("Day")),
            )

        return None

    def _extract_updated_date(self, article: ET.Element) -> str | None:
        revised_date = article.find(".//DateRevised")
        if revised_date is None:
            return None
        return ensure_iso_date(
            clean_text(revised_date.findtext("Year")),
            clean_text(revised_date.findtext("Month")),
            clean_text(revised_date.findtext("Day")),
        )

    def _build_journal_reference(self, article: ET.Element) -> str | None:
        journal = clean_text(article.findtext(".//Article/Journal/Title"))
        year = clean_text(article.findtext(".//JournalIssue/PubDate/Year"))
        volume = clean_text(article.findtext(".//JournalIssue/Volume"))
        issue = clean_text(article.findtext(".//JournalIssue/Issue"))
        pages = clean_text(article.findtext(".//Pagination/MedlinePgn"))

        parts = [part for part in [journal, year] if part]
        detail = volume
        if issue:
            detail = f"{detail}({issue})" if detail else f"({issue})"
        if pages:
            detail = f"{detail}:{pages}" if detail else pages
        if detail:
            parts.append(detail)
        return ". ".join(parts) if parts else None

    def run(self) -> int:
        print("=" * 60)
        print("PubMed Metadata Collection")
        print("=" * 60)
        print(f"Query: {self.query}")
        if self.start_date and self.end_date:
            print(f"Date range: {self.start_date} -> {self.end_date}")

        existing_ids = self._load_existing_ids()
        candidate_ids = self.search_ids(existing_ids=existing_ids)

        print(f"Found {len(candidate_ids)} new PubMed ids")
        records = self.fetch_records(candidate_ids)
        self._append_records(records)
        print(f"Saved {len(records)} PubMed papers to {self.output_path}")
        return len(records)


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Collect PubMed medical-domain papers into JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "raw" / "pubmed_papers.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="PubMed query string. Defaults to a medical AI query.",
    )
    parser.add_argument("--max_results", type=int, default=1000, help="Maximum number of PubMed records to collect.")
    parser.add_argument("--start_date", default="2020/01/01", help="Start publication date for PubMed search.")
    parser.add_argument(
        "--end_date",
        default=datetime.now(timezone.utc).strftime("%Y/%m/%d"),
        help="End publication date for PubMed search.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    collector = PubMedCollector(
        output_path=args.output,
        query=args.query,
        max_results=args.max_results,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    collector.run()


if __name__ == "__main__":
    main()
