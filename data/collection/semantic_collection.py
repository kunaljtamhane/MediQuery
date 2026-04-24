#!/usr/bin/env python3
"""
Semantic multi-source medical corpus collector.

Uses PubMedQA keywords as the query bank, fetches metadata from PubMed, arXiv,
and medRxiv, semantically ranks the candidate pool, and exports JSONL only.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import requests

from collection_utils import utc_now_iso
from env_loader import env_flag, load_env_file
from medrxiv_content import fetch_jats_full_text, get_jats_xml_path
from medrxiv_scraper import MedRxivCollector
from pubmed_scraper import PubMedCollector
from recent_arxiv_scraper import DEFAULT_ARXIV_QUERY, RecentArxivCollector


DEFAULT_PUBMED_COUNT = 250
DEFAULT_ARXIV_COUNT = 125
DEFAULT_MEDRXIV_COUNT = 125

PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
ARXIV_API = "http://export.arxiv.org/api/query"
MEDRXIV_API_BASE = "https://api.medrxiv.org/details/medrxiv"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

JSONL_FIELDS = [
    "source",
    "source_id",
    "source_rank",
    "title",
    "abstract",
    "summary",
    "authors",
    "author_string",
    "published_date",
    "updated_date",
    "categories",
    "primary_category",
    "domain",
    "doc_id",
    "text",
    "url",
    "source_url",
    "journal_ref",
    "journal",
    "publisher",
    "doi",
    "pubmed_id",
    "pmc_id",
    "arxiv_id",
    "medrxiv_id",
    "version",
    "license",
    "jats_xml_path",
    "comment",
    "keywords",
    "mesh_terms",
    "publication_types",
    "is_peer_reviewed",
    "preprint_server",
    "retrieval_query",
    "matched_keywords",
    "semantic_score",
    "query_bank",
    "collected_at",
    "pdf_url",
    "full_text_extracted",
    "full_text",
    "text_extraction_date",
    "raw_payload",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "if",
    "in", "into", "is", "it", "its", "may", "might", "more", "most", "of",
    "on", "or", "our", "should", "than", "that", "the", "their", "them",
    "there", "these", "this", "those", "to", "was", "were", "what", "when",
    "which", "who", "why", "will", "with", "would", "study", "studies",
    "result", "results", "patient", "patients", "disease", "clinical",
    "association", "associated", "treatment", "outcome",
}

GENERIC_QUERY_TERMS = {
    "group",
    "groups",
    "risk",
    "risks",
    "role",
    "roles",
    "effect",
    "effects",
    "women",
    "men",
    "child",
    "children",
    "adult",
    "adults",
    "year",
    "years",
    "use",
    "used",
    "using",
    "difference",
    "differences",
    "change",
    "changes",
}

BIOMEDICAL_QUERY_ANCHORS = (
    "medical",
    "clinical",
    "health",
    "biomed",
    "disease",
    "syndrome",
    "diagn",
    "therap",
    "treat",
    "cancer",
    "tumor",
    "drug",
    "patient",
    "surgery",
    "virus",
    "vaccine",
    "gene",
    "genetic",
    "protein",
    "cell",
    "brain",
    "card",
    "renal",
    "diabet",
    "pregnan",
    "mortality",
    "infection",
    "epidemi",
    "screen",
    "trial",
    "imaging",
    "hospital",
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    elif isinstance(value, dict):
        text = " ".join(normalize_text(v) for v in value.values())
    elif isinstance(value, (list, tuple, set)):
        text = " ".join(normalize_text(v) for v in value)
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text: str) -> List[str]:
    tokens = re.findall(r"[a-z][a-z0-9\-]*", text.lower())
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]


def make_record(**kwargs: Any) -> Dict[str, Any]:
    record = {field: None for field in JSONL_FIELDS}
    record.update(kwargs)
    return record


def batched(items: Sequence[str], size: int) -> Iterator[List[str]]:
    for index in range(0, len(items), size):
        yield list(items[index:index + size])


def iter_dataset_rows(dataset_obj: Any) -> Iterator[Dict[str, Any]]:
    split_names = getattr(dataset_obj, "keys", None)
    if callable(split_names):
        for split_name in dataset_obj.keys():
            for row in dataset_obj[split_name]:
                yield row
        return
    for row in dataset_obj:
        yield row


def build_pubmedqa_query_bank(top_k: int = 64, min_frequency: int = 2) -> List[str]:
    from datasets import load_dataset

    dataset_obj = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG)
    counts: Counter[str] = Counter()

    for row in iter_dataset_rows(dataset_obj):
        combined = normalize_text(" ".join([
            normalize_text(row.get("question")),
            normalize_text(row.get("context")),
            normalize_text(row.get("long_answer")),
            normalize_text(row.get("final_decision")),
        ]))
        tokens = tokenize_text(combined)
        for size in (1, 2, 3):
            for idx in range(len(tokens) - size + 1):
                phrase = " ".join(tokens[idx:idx + size])
                if len(phrase) >= 5:
                    counts[phrase] += 1

    ranked = sorted(
        ((phrase, count) for phrase, count in counts.items() if count >= min_frequency),
        key=lambda item: (-item[1], -len(item[0].split()), item[0]),
    )

    selected: List[str] = []
    seen: set[str] = set()
    for phrase, _ in ranked:
        tokens = phrase.split()
        novelty = sum(token not in seen for token in tokens)
        if novelty == 0 and len(selected) >= top_k // 2:
            continue
        selected.append(phrase)
        seen.update(tokens)
        if len(selected) >= top_k:
            break

    return selected


class SemanticRanker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.backend = "lexical"
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                local_files_only=not env_flag("SEMANTIC_RANKER_ALLOW_DOWNLOAD", False),
            )
            self.backend = "sentence-transformers"
        except Exception:
            self.model = None

    def rank(self, records: Sequence[Dict[str, Any]], query_bank: Sequence[str]) -> List[Dict[str, Any]]:
        if self.model is not None:
            return self._rank_embeddings(records, query_bank)
        return self._rank_lexical(records, query_bank)

    def _record_text(self, record: Dict[str, Any]) -> str:
        return normalize_text(" ".join([
            normalize_text(record.get("title")),
            normalize_text(record.get("abstract")),
            normalize_text(record.get("summary")),
            normalize_text(record.get("journal")),
            normalize_text(record.get("categories")),
            normalize_text(record.get("keywords")),
            normalize_text(record.get("mesh_terms")),
        ]))

    def _rank_embeddings(self, records: Sequence[Dict[str, Any]], query_bank: Sequence[str]) -> List[Dict[str, Any]]:
        import numpy as np

        doc_embeddings = self.model.encode(
            [self._record_text(record) for record in records],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embeddings = self.model.encode(
            list(query_bank),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarity = np.matmul(doc_embeddings, query_embeddings.T)
        ranked: List[Dict[str, Any]] = []
        for index, record in enumerate(records):
            item = dict(record)
            item["semantic_score"] = float(similarity[index].max())
            ranked.append(item)
        ranked.sort(key=lambda item: item["semantic_score"], reverse=True)
        return ranked

    def _rank_lexical(self, records: Sequence[Dict[str, Any]], query_bank: Sequence[str]) -> List[Dict[str, Any]]:
        query_sets = [set(tokenize_text(query)) for query in query_bank if query]
        ranked: List[Dict[str, Any]] = []
        for record in records:
            tokens = set(tokenize_text(self._record_text(record)))
            max_overlap = 0.0
            for query_set in query_sets:
                if query_set:
                    max_overlap = max(max_overlap, len(tokens & query_set) / len(query_set))
            item = dict(record)
            item["semantic_score"] = max_overlap
            ranked.append(item)
        ranked.sort(key=lambda item: item["semantic_score"], reverse=True)
        return ranked


def collect_keyword_hits(record: Dict[str, Any], query_bank: Sequence[str], limit: int = 12) -> List[str]:
    haystack = normalize_text(" ".join([
        normalize_text(record.get("title")),
        normalize_text(record.get("abstract")),
        normalize_text(record.get("summary")),
        normalize_text(record.get("categories")),
        normalize_text(record.get("keywords")),
        normalize_text(record.get("mesh_terms")),
    ])).lower()
    return [query for query in query_bank if query.lower() in haystack][:limit]


def write_jsonl(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            normalized = {field: None for field in JSONL_FIELDS}
            normalized.update(record)
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")


def dedupe_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        key = (record.get("source"), record.get("source_id"))
        if key not in seen:
            seen[key] = record
    return list(seen.values())


def select_source_queries(
    query_bank: Sequence[str],
    source: str,
    max_queries: int,
) -> List[str]:
    selected: List[str] = []

    for query in query_bank:
        text = normalize_text(query).lower()
        if not text or text in GENERIC_QUERY_TERMS:
            continue

        tokens = text.split()
        if source in {"arxiv", "medrxiv"}:
            if len(tokens) < 2:
                continue
            if not any(anchor in text for anchor in BIOMEDICAL_QUERY_ANCHORS):
                continue

        selected.append(query)
        if len(selected) >= max_queries:
            return selected

    if selected:
        return selected

    # Graceful fallback if aggressive filtering removes everything.
    fallback = [query for query in query_bank if len(normalize_text(query).split()) >= 2]
    return fallback[:max_queries] if fallback else list(query_bank[:max_queries])


def finalize_records(
    records: Sequence[Dict[str, Any]],
    query_bank: Sequence[str],
    ranker: SemanticRanker,
    limit: int,
) -> List[Dict[str, Any]]:
    ranked = ranker.rank(records, query_bank)
    final: List[Dict[str, Any]] = []
    for rank, record in enumerate(ranked[:limit], start=1):
        item = dict(record)
        item["source_rank"] = rank
        item["matched_keywords"] = collect_keyword_hits(item, query_bank)
        item["query_bank"] = list(query_bank)
        final.append(item)
    return final


def _null_out_pdf_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(record)
    item["pdf_url"] = None
    item["full_text_extracted"] = False
    item["full_text"] = None
    item["text_extraction_date"] = None
    return item


def adapt_pubmed_record(record: Any) -> Dict[str, Any]:
    payload = asdict(record)
    return _null_out_pdf_fields(
        make_record(
            source=payload.get("source"),
            source_id=payload.get("source_id"),
            title=payload.get("title"),
            abstract=payload.get("abstract"),
            summary=payload.get("abstract"),
            authors=payload.get("authors"),
            author_string=", ".join(payload.get("authors") or []) or None,
            published_date=payload.get("published_date"),
            updated_date=payload.get("updated_date"),
            categories=payload.get("categories"),
            primary_category=payload.get("primary_category"),
            domain=payload.get("domain", "medical"),
            doc_id=f"pubmed:{payload.get('source_id')}",
            text=payload.get("abstract") or payload.get("title"),
            url=payload.get("source_url"),
            source_url=payload.get("source_url"),
            journal_ref=payload.get("journal_ref"),
            journal=payload.get("journal_ref"),
            publisher=None,
            doi=payload.get("doi"),
            pubmed_id=payload.get("pubmed_id"),
            pmc_id=payload.get("pmc_id"),
            comment=payload.get("comment"),
            keywords=None,
            mesh_terms=payload.get("categories"),
            publication_types=None,
            is_peer_reviewed=True,
            preprint_server=None,
            retrieval_query=None,
            collected_at=payload.get("collected_at"),
            raw_payload=payload,
        )
    )


def adapt_arxiv_record(record: Any) -> Dict[str, Any]:
    payload = asdict(record)
    return _null_out_pdf_fields(
        make_record(
            source=payload.get("source"),
            source_id=payload.get("source_id"),
            title=payload.get("title"),
            abstract=payload.get("abstract"),
            summary=payload.get("abstract"),
            authors=payload.get("authors"),
            author_string=", ".join(payload.get("authors") or []) or None,
            published_date=payload.get("published_date"),
            updated_date=payload.get("updated_date"),
            categories=payload.get("categories"),
            primary_category=payload.get("primary_category"),
            domain=payload.get("domain", "medical"),
            doc_id=f"arxiv:{payload.get('source_id')}",
            text=payload.get("abstract") or payload.get("title"),
            url=payload.get("source_url"),
            source_url=payload.get("source_url"),
            journal_ref=payload.get("journal_ref"),
            journal=None,
            publisher=None,
            doi=payload.get("doi"),
            arxiv_id=payload.get("arxiv_id"),
            comment=payload.get("comment"),
            keywords=None,
            mesh_terms=None,
            publication_types=None,
            is_peer_reviewed=False,
            preprint_server="arxiv",
            retrieval_query=None,
            collected_at=payload.get("collected_at"),
            raw_payload=payload,
        )
    )


def adapt_medrxiv_record(record: Any) -> Dict[str, Any]:
    payload = asdict(record)
    return make_record(
        source=payload.get("source"),
        source_id=payload.get("source_id"),
        title=payload.get("title"),
        abstract=payload.get("abstract"),
        summary=payload.get("abstract"),
        authors=payload.get("authors"),
        author_string=", ".join(payload.get("authors") or []) or None,
        published_date=payload.get("published_date"),
        updated_date=payload.get("updated_date"),
        categories=payload.get("categories"),
        primary_category=payload.get("primary_category"),
        domain=payload.get("domain", "medical"),
        doc_id=f"medrxiv:{payload.get('source_id')}",
        text=payload.get("full_text") or payload.get("abstract") or payload.get("title"),
        url=payload.get("source_url"),
        source_url=payload.get("source_url"),
        journal_ref=payload.get("journal_ref"),
        journal=payload.get("published_journal_ref"),
        publisher="medRxiv",
        doi=payload.get("doi"),
        medrxiv_id=payload.get("medrxiv_id"),
        version=payload.get("version"),
        license=payload.get("license"),
        jats_xml_path=payload.get("jats_xml_path"),
        comment=payload.get("comment"),
        keywords=None,
        mesh_terms=None,
        publication_types=None,
        is_peer_reviewed=False,
        preprint_server="medrxiv",
        retrieval_query=None,
        collected_at=payload.get("collected_at"),
        pdf_url=payload.get("pdf_url"),
        full_text_extracted=bool(payload.get("full_text")),
        full_text=payload.get("full_text"),
        text_extraction_date=payload.get("text_extraction_date"),
        raw_payload=payload,
    )


def fetch_pubmed_candidates_via_scraper(
    raw_dir: Path,
    query_bank: Sequence[str],
    start_date: str,
    end_date: str,
    limit: int,
    pubmed_email: str | None = None,
) -> List[Dict[str, Any]]:
    queries = select_source_queries(query_bank, source="pubmed", max_queries=24)
    collector = PubMedCollector(
        output_path=raw_dir / "_tmp_pubmed_unused.jsonl",
        query=queries[0] if queries else "medical",
        max_results=max(limit * 5, 100),
        start_date=start_date.replace("-", "/"),
        end_date=end_date.replace("-", "/"),
    )
    if pubmed_email:
        collector.email = pubmed_email

    candidate_ids: List[str] = []
    seen_ids: set[str] = set()
    per_query_cap = max(25, min(100, limit))
    for query in queries:
        collector.query = query
        ids = collector.search_ids(existing_ids=seen_ids)
        for paper_id in ids[:per_query_cap]:
            if paper_id not in seen_ids:
                seen_ids.add(paper_id)
                candidate_ids.append(paper_id)
        if len(candidate_ids) >= limit * 5:
            break

    collector.fetch_batch_size = 100
    records = collector.fetch_records(candidate_ids[: limit * 5])
    return dedupe_records(adapt_pubmed_record(record) for record in records)


def fetch_arxiv_candidates_via_scraper(
    raw_dir: Path,
    query_bank: Sequence[str],
    start_date: str,
    end_date: str,
    limit: int,
) -> List[Dict[str, Any]]:
    queries = select_source_queries(query_bank, source="arxiv", max_queries=20)
    records: List[Dict[str, Any]] = []
    per_query_cap = max(30, min(100, max(limit, 50)))
    query_variants: List[str] = []

    for query in queries:
        query_variants.append(
            f'all:"{query}" AND '
            "(all:medical OR all:clinical OR all:healthcare OR all:medicine OR all:biomedical)"
        )

    # Broad fallback candidates so semantic reranking has enough recall.
    query_variants.extend(
        [
            DEFAULT_ARXIV_QUERY,
            '(cat:q-bio.* OR cat:cs.AI OR cat:cs.LG OR cat:stat.ML) AND '
            '(all:medical OR all:clinical OR all:healthcare OR all:medicine OR all:biomedical)',
            '(all:"clinical decision support" OR all:"medical imaging" OR all:"electronic health record" '
            'OR all:"biomedical" OR all:"healthcare") AND '
            '(cat:cs.AI OR cat:cs.LG OR cat:q-bio.* OR cat:stat.ML)',
        ]
    )

    seen_queries: set[str] = set()
    for arxiv_query in query_variants:
        if arxiv_query in seen_queries:
            continue
        seen_queries.add(arxiv_query)
        collector = RecentArxivCollector(
            output_path=raw_dir / "_tmp_arxiv_unused.jsonl",
            max_results=per_query_cap,
            start_date=start_date,
            end_date=end_date,
            query=arxiv_query,
        )
        fetched = collector.fetch_records()
        records.extend(adapt_arxiv_record(record) for record in fetched)
        if len(dedupe_records(records)) >= limit * 8:
            break

    return dedupe_records(records)


def fetch_medrxiv_candidates_via_scraper(
    raw_dir: Path,
    query_bank: Sequence[str],
    start_date: str,
    end_date: str,
    limit: int,
) -> List[Dict[str, Any]]:
    queries = select_source_queries(query_bank, source="medrxiv", max_queries=20)
    records: List[Dict[str, Any]] = []
    window_end = datetime.strptime(end_date, "%Y-%m-%d").date()
    absolute_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    per_window_cap = max(10, min(25, max(limit // 2, 15)))

    def fetch_window(
        window_start: date,
        window_end: date,
        target_cap: int,
        keyword_filters: Sequence[str],
    ) -> List[Dict[str, Any]]:
        collector = MedRxivCollector(
            output_path=raw_dir / "_tmp_medrxiv_unused.jsonl",
            max_results=target_cap,
            start_date=window_start.isoformat(),
            end_date=window_end.isoformat(),
            keywords=list(keyword_filters),
        )
        try:
            fetched = collector.fetch_records(existing_ids=set())
            return [adapt_medrxiv_record(record) for record in fetched]
        except Exception as exc:
            span_days = (window_end - window_start).days + 1
            if span_days <= 7:
                print(
                    f"Warning: skipping medRxiv window {window_start.isoformat()} -> "
                    f"{window_end.isoformat()} after repeated failures: {exc}"
                )
                return []

            midpoint = window_start + timedelta(days=span_days // 2)
            left_end = midpoint - timedelta(days=1)
            left_records = (
                fetch_window(window_start, left_end, target_cap, keyword_filters)
                if left_end >= window_start
                else []
            )
            remaining = max(0, target_cap - len(dedupe_records(left_records)))
            right_records = fetch_window(
                midpoint,
                window_end,
                remaining or target_cap,
                keyword_filters,
            )
            return dedupe_records(left_records + right_records)[:target_cap]

    def run_pass(keyword_filters: Sequence[str], existing_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass_records = list(existing_records)
        pass_window_end = window_end
        while pass_window_end >= absolute_start and len(dedupe_records(pass_records)) < limit * 8:
            pass_window_start = max(absolute_start, pass_window_end - timedelta(days=13))
            remaining = max(0, limit * 8 - len(dedupe_records(pass_records)))
            if remaining <= 0:
                break
            target_cap = min(per_window_cap, remaining)
            pass_records.extend(fetch_window(pass_window_start, pass_window_end, target_cap, keyword_filters))
            pass_window_end = pass_window_start - timedelta(days=1)
        return dedupe_records(pass_records)

    # First pass: focused keyword-filtered windows.
    records = run_pass(queries, records)

    # Second pass: broad medRxiv crawl with no keyword gate if recall is still low.
    if len(records) < limit * 3:
        records = run_pass([], records)

    return dedupe_records(records)


class PubMedSemanticCollector:
    def __init__(self, session: requests.Session, start_date: str, end_date: str, email: str | None = None) -> None:
        load_env_file(Path(__file__).resolve().parents[2] / ".env")
        self.session = session
        self.start_date = start_date
        self.end_date = end_date
        self.email = email or os.getenv("NCBI_EMAIL", "")
        self.api_key = os.getenv("NCBI_API_KEY", "")
        self.tool_name = os.getenv("NCBI_TOOL", "capstone-semantic-collector")
        self.delay_seconds = 0.11 if self.api_key else 0.34

    def _request(self, endpoint: str, params: Dict[str, Any], timeout: int = 90) -> requests.Response:
        payload = {"tool": self.tool_name, **params}
        if self.email:
            payload["email"] = self.email
        if self.api_key:
            payload["api_key"] = self.api_key
        response = self.session.get(f"{NCBI_BASE_URL}/{endpoint}", params=payload, timeout=timeout)
        response.raise_for_status()
        time.sleep(self.delay_seconds)
        return response

    def _search_ids(self, query: str, retmax: int = 50) -> List[str]:
        response = self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": retmax,
                "sort": "relevance",
                "datetype": "pdat",
                "mindate": self.start_date.replace("-", "/"),
                "maxdate": self.end_date.replace("-", "/"),
            },
        )
        return response.json().get("esearchresult", {}).get("idlist", [])

    def _fetch_summary(self, ids: Sequence[str]) -> Dict[str, Any]:
        response = self._request("esummary.fcgi", {"db": "pubmed", "retmode": "json", "id": ",".join(ids)})
        return response.json().get("result", {})

    def _fetch_xml(self, ids: Sequence[str]) -> ET.Element:
        response = self._request(
            "efetch.fcgi",
            {"db": "pubmed", "retmode": "xml", "rettype": "abstract", "id": ",".join(ids)},
            timeout=120,
        )
        return ET.fromstring(response.text)

    def collect(self, query_bank: Sequence[str], limit: int) -> List[Dict[str, Any]]:
        candidate_ids: List[str] = []
        for query in query_bank:
            candidate_ids.extend(self._search_ids(query))
            if len(set(candidate_ids)) >= limit * 5:
                break
        unique_ids = list(dict.fromkeys(candidate_ids))[: limit * 5]
        summary_map: Dict[str, Any] = {}
        xml_map: Dict[str, Dict[str, Any]] = {}

        for chunk in batched(unique_ids, 100):
            summary = self._fetch_summary(chunk)
            for uid in summary.get("uids", []):
                if isinstance(summary.get(uid), dict):
                    summary_map[uid] = summary[uid]
            xml_root = self._fetch_xml(chunk)
            for article in xml_root.findall(".//PubmedArticle"):
                parsed = self._parse_article(article)
                if parsed:
                    xml_map[parsed["pubmed_id"]] = parsed

        records: List[Dict[str, Any]] = []
        collected_at = utc_now_iso()
        for pubmed_id, summary in summary_map.items():
            xml_bits = xml_map.get(pubmed_id, {})
            article_ids = summary.get("articleids", []) or []
            doi = None
            pmc_id = None
            for article_id in article_ids:
                id_type = normalize_text(article_id.get("idtype")).lower()
                value = normalize_text(article_id.get("value"))
                if id_type == "doi" and value:
                    doi = value
                if id_type == "pmc" and value:
                    pmc_id = value
            authors = [author.get("name") for author in summary.get("authors", []) if author.get("name")]
            title = normalize_text(summary.get("title"))
            abstract = xml_bits.get("abstract")
            records.append(make_record(
                source="pubmed",
                source_id=pubmed_id,
                title=title,
                abstract=abstract,
                summary=abstract,
                authors=authors,
                author_string=", ".join(authors) if authors else None,
                published_date=normalize_text(summary.get("pubdate")) or None,
                updated_date=normalize_text(summary.get("sortpubdate")) or None,
                categories=None,
                primary_category="medical",
                domain="medical",
                doc_id=f"pubmed:{pubmed_id}",
                text=abstract or title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                journal_ref=normalize_text(summary.get("source")) or None,
                journal=normalize_text(summary.get("fulljournalname") or summary.get("source")) or None,
                publisher=None,
                doi=doi,
                pubmed_id=pubmed_id,
                pmc_id=pmc_id,
                comment=None,
                keywords=xml_bits.get("keywords"),
                mesh_terms=xml_bits.get("mesh_terms"),
                publication_types=xml_bits.get("publication_types"),
                is_peer_reviewed=True,
                preprint_server=None,
                retrieval_query=next(
                    (query for query in query_bank if query.lower() in title.lower()),
                    query_bank[0],
                ),
                collected_at=collected_at,
                pdf_url=None,
                full_text_extracted=False,
                full_text=None,
                text_extraction_date=None,
                raw_payload={"summary": summary, "xml_extract": xml_bits},
            ))
        return dedupe_records(records)

    def _parse_article(self, article: ET.Element) -> Dict[str, Any] | None:
        pubmed_id = normalize_text(article.findtext(".//MedlineCitation/PMID"))
        if not pubmed_id:
            return None
        abstract_parts = [
            normalize_text("".join(node.itertext()))
            for node in article.findall(".//Abstract/AbstractText")
            if normalize_text("".join(node.itertext()))
        ]
        keywords = [
            normalize_text("".join(node.itertext()))
            for node in article.findall(".//KeywordList/Keyword")
            if normalize_text("".join(node.itertext()))
        ]
        mesh_terms = [
            normalize_text("".join(node.itertext()))
            for node in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if normalize_text("".join(node.itertext()))
        ]
        publication_types = [
            normalize_text("".join(node.itertext()))
            for node in article.findall(".//PublicationTypeList/PublicationType")
            if normalize_text("".join(node.itertext()))
        ]
        return {
            "pubmed_id": pubmed_id,
            "abstract": " ".join(abstract_parts) if abstract_parts else None,
            "keywords": keywords or None,
            "mesh_terms": mesh_terms or None,
            "publication_types": publication_types or None,
        }


class ArxivSemanticCollector:
    def __init__(self, session: requests.Session, start_date: str, end_date: str) -> None:
        self.session = session
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    def _request(self, query: str) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = self.session.get(
                    ARXIV_API,
                    params={
                        "search_query": (
                            f'all:"{query}" AND '
                            "(all:medical OR all:clinical OR all:healthcare OR all:medicine OR all:biomedical)"
                        ),
                        "start": 0,
                        "max_results": 40,
                        "sortBy": "relevance",
                        "sortOrder": "descending",
                    },
                    timeout=90,
                    headers={"User-Agent": "capstone-semantic-collector/1.0"},
                )
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    sleep_seconds = float(retry_after) if retry_after else min(30.0, 3.0 * attempt)
                    time.sleep(sleep_seconds)
                    continue
                response.raise_for_status()
                time.sleep(3.5)
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 5:
                    raise
                time.sleep(min(30.0, 2.0 * attempt))
        raise RuntimeError(f"arXiv request failed: {last_error}")

    def collect(self, query_bank: Sequence[str], limit: int) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        collected_at = utc_now_iso()
        for query in select_source_queries(query_bank, source="arxiv", max_queries=12):
            response = self._request(query)
            root = ET.fromstring(response.text)
            for entry in root.findall("atom:entry", ARXIV_NS):
                entry_id = normalize_text(
                    entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
                )
                arxiv_id = entry_id.rstrip("/").split("/")[-1]
                title = normalize_text(
                    entry.findtext("atom:title", default="", namespaces=ARXIV_NS)
                )
                abstract = normalize_text(
                    entry.findtext("atom:summary", default="", namespaces=ARXIV_NS)
                )
                published = normalize_text(
                    entry.findtext("atom:published", default="", namespaces=ARXIV_NS)
                )
                published_date = datetime.fromisoformat(published.replace("Z", "+00:00")).date()
                if not (self.start_date <= published_date <= self.end_date):
                    continue
                authors = [
                    normalize_text(author.findtext("atom:name", default="", namespaces=ARXIV_NS))
                    for author in entry.findall("atom:author", ARXIV_NS)
                ]
                categories = [
                    node.attrib.get("term")
                    for node in entry.findall("atom:category", ARXIV_NS)
                    if node.attrib.get("term")
                ]
                primary_node = entry.find("arxiv:primary_category", ARXIV_NS)
                primary_category = (
                    primary_node.attrib.get("term")
                    if primary_node is not None
                    else (categories[0] if categories else "arxiv")
                )
                doi_node = entry.find("arxiv:doi", ARXIV_NS)
                journal_node = entry.find("arxiv:journal_ref", ARXIV_NS)
                comment_node = entry.find("arxiv:comment", ARXIV_NS)
                records.append(make_record(
                    source="arxiv",
                    source_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    summary=abstract,
                    authors=[author for author in authors if author],
                    author_string=", ".join(author for author in authors if author) or None,
                    published_date=published,
                    updated_date=normalize_text(
                        entry.findtext("atom:updated", default="", namespaces=ARXIV_NS)
                    ) or None,
                    categories=categories or None,
                    primary_category=primary_category,
                    domain="medical",
                    doc_id=f"arxiv:{arxiv_id}",
                    text=abstract or title,
                    url=entry_id,
                    source_url=entry_id,
                    journal_ref=(
                        normalize_text(journal_node.text)
                        if journal_node is not None and journal_node.text
                        else None
                    ),
                    journal=None,
                    publisher=None,
                    doi=(
                        normalize_text(doi_node.text)
                        if doi_node is not None and doi_node.text
                        else None
                    ),
                    arxiv_id=arxiv_id,
                    comment=(
                        normalize_text(comment_node.text)
                        if comment_node is not None and comment_node.text
                        else None
                    ),
                    keywords=None,
                    mesh_terms=None,
                    publication_types=None,
                    is_peer_reviewed=False,
                    preprint_server="arxiv",
                    retrieval_query=query,
                    collected_at=collected_at,
                    pdf_url=None,
                    full_text_extracted=False,
                    full_text=None,
                    text_extraction_date=None,
                    raw_payload={
                        "entry_id": entry_id,
                        "categories": categories,
                        "primary_category": primary_category,
                    },
                ))
            if len(dedupe_records(records)) >= limit * 4:
                break
        return dedupe_records(records)


class MedrxivSemanticCollector:
    def __init__(self, session: requests.Session, start_date: str, end_date: str) -> None:
        self.session = session
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    def _request_page(self, start_date: date, end_date: date, cursor: int) -> requests.Response:
        url = f"{MEDRXIV_API_BASE}/{start_date.isoformat()}/{end_date.isoformat()}/{cursor}/json"
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = self.session.get(
                    url,
                    timeout=180,  # increased from 90 → 180 s
                    headers={"User-Agent": "capstone-semantic-collector/1.0"},
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    sleep_secs = min(90.0, 10.0 * attempt)
                    print(
                        f"medRxiv HTTP {response.status_code} on attempt {attempt}/5, "
                        f"retrying in {sleep_secs:.0f}s …"
                    )
                    time.sleep(sleep_secs)
                    continue
                response.raise_for_status()
                time.sleep(1.0)  # polite inter-request gap
                return response
            except requests.exceptions.ReadTimeout as exc:
                last_error = exc
                sleep_secs = min(90.0, 15.0 * attempt)
                print(
                    f"medRxiv read timeout (attempt {attempt}/5) for window "
                    f"{start_date} → {end_date}, retrying in {sleep_secs:.0f}s …"
                )
                time.sleep(sleep_secs)
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 5:
                    raise
                time.sleep(min(60.0, 5.0 * attempt))
        raise RuntimeError(f"medRxiv request failed after 5 attempts: {last_error}")

    def _iter_date_windows(self, window_days: int = 30) -> Iterator[tuple[date, date]]:
        # 30-day windows (down from 90) → smaller payloads → fewer timeouts
        current_end = self.end_date
        delta = timedelta(days=window_days - 1)
        while current_end >= self.start_date:
            current_start = max(self.start_date, current_end - delta)
            yield current_start, current_end
            current_end = current_start - timedelta(days=1)

    def _build_record(
        self,
        item: dict,
        selected_queries: Sequence[str],
        collected_at: str,
    ) -> Dict[str, Any] | None:
        title = normalize_text(item.get("title"))
        abstract = normalize_text(item.get("abstract"))
        category = normalize_text(item.get("category")) or "medical"
        haystack = f"{title} {abstract} {category}".lower()
        if selected_queries and not any(query.lower() in haystack for query in selected_queries):
            return None

        doi = normalize_text(item.get("doi"))
        version = normalize_text(item.get("version")) or None
        version_suffix = f"v{version}" if version else ""
        landing_page = f"https://www.medrxiv.org/content/{doi}{version_suffix}" if doi else None
        pdf_url = f"{landing_page}.full.pdf" if landing_page else None
        jats_xml_path = get_jats_xml_path(item)
        full_text, resolved_jats_url, jats_error = fetch_jats_full_text(
            self.session,
            jats_xml_path,
            timeout=60,
        )
        raw_payload = dict(item)
        if jats_error and jats_xml_path:
            raw_payload["jats_fetch_error"] = jats_error
        authors_text = normalize_text(item.get("authors"))
        authors = [author.strip() for author in re.split(r"[;,]", authors_text) if author.strip()]

        return make_record(
            source="medrxiv",
            source_id=doi or title,
            title=title,
            abstract=abstract,
            summary=abstract,
            authors=authors or None,
            author_string=", ".join(authors) if authors else None,
            published_date=normalize_text(item.get("date")) or None,
            updated_date=normalize_text(item.get("published")) or None,
            categories=[category],
            primary_category=category,
            domain="medical",
            doc_id=f"medrxiv:{doi or title}",
            text=full_text or abstract or title,
            url=landing_page,
            source_url=landing_page,
            journal_ref=normalize_text(item.get("published")) or None,
            journal=None,
            publisher="medRxiv",
            doi=doi or None,
            medrxiv_id=doi or None,
            version=version,
            license=normalize_text(item.get("license")) or None,
            jats_xml_path=resolved_jats_url or jats_xml_path,
            comment=normalize_text(item.get("type")) or None,
            keywords=None,
            mesh_terms=None,
            publication_types=None,
            is_peer_reviewed=False,
            preprint_server="medrxiv",
            retrieval_query="pubmedqa-semantic-medrxiv",
            collected_at=collected_at,
            pdf_url=pdf_url,
            full_text_extracted=bool(full_text),
            full_text=full_text,
            text_extraction_date=collected_at if full_text else None,
            raw_payload=raw_payload,
        )

    def _collect_window(
        self,
        window_start: date,
        window_end: date,
        selected_queries: Sequence[str],
        collected_at: str,
        target_cap: int,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        cursor = 0

        while len(dedupe_records(records)) < target_cap:
            try:
                response = self._request_page(window_start, window_end, cursor)
            except (requests.RequestException, RuntimeError) as exc:
                window_span = (window_end - window_start).days + 1
                if window_span <= 7:
                    # Minimum granularity reached — skip gracefully instead of crashing.
                    print(
                        f"Warning: skipping medRxiv window "
                        f"{window_start.isoformat()} → {window_end.isoformat()} "
                        f"after repeated failures: {exc}"
                    )
                    break

                # Binary-split the window and recurse.
                midpoint = window_start + timedelta(days=window_span // 2)
                left_end = midpoint - timedelta(days=1)
                left_records = (
                    self._collect_window(
                        window_start, left_end, selected_queries, collected_at, target_cap
                    )
                    if left_end >= window_start
                    else []
                )
                remaining = max(0, target_cap - len(dedupe_records(left_records)))
                right_records = self._collect_window(
                    midpoint,
                    window_end,
                    selected_queries,
                    collected_at,
                    remaining or target_cap,
                )
                return dedupe_records(left_records + right_records)[:target_cap]

            collection = response.json().get("collection", [])
            if not collection:
                break

            for item in collection:
                record = self._build_record(item, selected_queries, collected_at)
                if record is not None:
                    records.append(record)

            if len(collection) < 100:
                break
            cursor += 100

        return dedupe_records(records)[:target_cap]

    def collect(self, query_bank: Sequence[str], limit: int) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        collected_at = utc_now_iso()
        selected_queries = select_source_queries(query_bank, source="medrxiv", max_queries=20)

        for window_start, window_end in self._iter_date_windows(window_days=30):
            remaining = max(0, limit * 5 - len(dedupe_records(records)))
            if remaining <= 0:
                break
            try:
                new_records = self._collect_window(
                    window_start,
                    window_end,
                    selected_queries,
                    collected_at,
                    remaining,
                )
                records.extend(new_records)
            except Exception as exc:
                # Top-level safety net: log and move on to the next window.
                print(
                    f"Warning: medRxiv window {window_start} → {window_end} "
                    f"failed entirely, skipping: {exc}"
                )
                continue

            if len(dedupe_records(records)) >= limit * 5:
                break

        return dedupe_records(records)


def collect_semantic_corpus(
    raw_dir: Path,
    start_date: str,
    end_date: str,
    pubmed_count: int = DEFAULT_PUBMED_COUNT,
    arxiv_count: int = DEFAULT_ARXIV_COUNT,
    medrxiv_count: int = DEFAULT_MEDRXIV_COUNT,
    top_k_keywords: int = 64,
    pubmed_email: str | None = None,
) -> Dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    query_bank = build_pubmedqa_query_bank(top_k=top_k_keywords)
    ranker = SemanticRanker()

    pubmed_records = finalize_records(
        fetch_pubmed_candidates_via_scraper(
            raw_dir=raw_dir,
            query_bank=query_bank,
            start_date=start_date,
            end_date=end_date,
            limit=pubmed_count,
            pubmed_email=pubmed_email,
        ),
        query_bank=query_bank,
        ranker=ranker,
        limit=pubmed_count,
    )
    arxiv_records = finalize_records(
        fetch_arxiv_candidates_via_scraper(
            raw_dir=raw_dir,
            query_bank=query_bank,
            start_date=start_date,
            end_date=end_date,
            limit=arxiv_count,
        ),
        query_bank=query_bank,
        ranker=ranker,
        limit=arxiv_count,
    )
    medrxiv_records = finalize_records(
        fetch_medrxiv_candidates_via_scraper(
            raw_dir=raw_dir,
            query_bank=query_bank,
            start_date=start_date,
            end_date=end_date,
            limit=medrxiv_count,
        ),
        query_bank=query_bank,
        ranker=ranker,
        limit=medrxiv_count,
    )

    write_jsonl(pubmed_records, raw_dir / "pubmed_papers.jsonl")
    write_jsonl(arxiv_records, raw_dir / "arxiv_papers.jsonl")
    write_jsonl(medrxiv_records, raw_dir / "medrxiv_papers.jsonl")
    write_jsonl(
        pubmed_records + arxiv_records + medrxiv_records,
        raw_dir / "papers.jsonl",
    )

    return {
        "query_bank_size": len(query_bank),
        "semantic_backend": ranker.backend,
        "counts": {
            "pubmed": len(pubmed_records),
            "arxiv": len(arxiv_records),
            "medrxiv": len(medrxiv_records),
            "total": len(pubmed_records) + len(arxiv_records) + len(medrxiv_records),
        },
    }
