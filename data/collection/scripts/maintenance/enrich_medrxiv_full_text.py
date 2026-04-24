#!/usr/bin/env python3
"""
Enrich existing medRxiv JSONL records with JATS XML full text.

This is the recommended medRxiv path for the project:
API metadata first, JATS XML full text second, PDF download last/optional.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests
from tqdm import tqdm

COLLECTION_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = COLLECTION_DIR.parents[1]

sys.path.insert(0, str(COLLECTION_DIR))

from collection_utils import rebuild_papers_jsonl, utc_now_iso
from env_loader import configure_requests_session, load_env_file
from medrxiv_content import fetch_jats_full_text, get_jats_xml_path, normalize_jats_url


MEDRXIV_API_BASE = "https://api.medrxiv.org/details/medrxiv"

def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def date_only(value: Any) -> str | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value or ""))
    return match.group(0) if match else None


def _fetch_medrxiv_doi_metadata(session: requests.Session, doi: str, timeout: int) -> dict[str, Any] | None:
    if not doi:
        return None
    url = f"{MEDRXIV_API_BASE}/{quote(doi, safe='/')}/na/json"
    response = session.get(
        url,
        timeout=timeout,
        headers={
            "Accept": "application/json",
            "User-Agent": "capstone-medrxiv-enricher/1.0",
        },
    )
    response.raise_for_status()
    payload = response.json()
    collection = payload.get("collection") or []
    if not collection:
        return None
    return collection[0]


def _fetch_medrxiv_date_metadata(
    session: requests.Session,
    doi: str,
    published_date: str | None,
    timeout: int,
) -> dict[str, Any] | None:
    day = date_only(published_date)
    if not doi or not day:
        return None

    cursor = 0
    while True:
        url = f"{MEDRXIV_API_BASE}/{day}/{day}/{cursor}/json"
        response = session.get(
            url,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "capstone-medrxiv-enricher/1.0",
            },
        )
        response.raise_for_status()
        payload = response.json()
        collection = payload.get("collection") or []
        for candidate in collection:
            if str(candidate.get("doi") or "").strip().lower() == doi.lower():
                return candidate
        if len(collection) < 100:
            return None
        cursor += len(collection)


def fetch_medrxiv_metadata(
    session: requests.Session,
    doi: str,
    timeout: int,
    *,
    published_date: str | None = None,
) -> dict[str, Any] | None:
    metadata = _fetch_medrxiv_doi_metadata(session, doi, timeout)
    if metadata:
        return metadata
    return _fetch_medrxiv_date_metadata(session, doi, published_date, timeout)


def enrich_record(
    record: dict[str, Any],
    session: requests.Session,
    *,
    timeout: int,
) -> tuple[dict[str, Any], bool]:
    item = dict(record)
    source = str(item.get("source") or "").lower()
    if source and source != "medrxiv":
        return item, False

    doi = str(item.get("doi") or item.get("medrxiv_id") or item.get("source_id") or "").strip()
    if not doi:
        return item, False

    raw_payload = item.get("raw_payload") if isinstance(item.get("raw_payload"), dict) else {}
    jats_xml_path = get_jats_xml_path(item, raw_payload)
    metadata: dict[str, Any] | None = None

    if not jats_xml_path:
        metadata = fetch_medrxiv_metadata(
            session,
            doi,
            timeout,
            published_date=item.get("published_date") or item.get("date"),
        )
        if metadata:
            jats_xml_path = get_jats_xml_path(metadata)

    full_text, resolved_jats_url, error = fetch_jats_full_text(session, jats_xml_path, timeout=timeout)
    if not full_text:
        item["jats_xml_path"] = normalize_jats_url(jats_xml_path)
        item["jats_fetch_error"] = error
        # Always try the API as fallback — even if we had a jats_xml_path it may be 403-blocked.
        if metadata is None:
            try:
                metadata = fetch_medrxiv_metadata(
                    session,
                    doi,
                    timeout,
                    published_date=item.get("published_date") or item.get("date"),
                )
            except Exception:
                metadata = None
        if metadata:
            abstract = str(metadata.get("abstract") or item.get("abstract") or "").strip()
            version = str(metadata.get("version") or item.get("version") or "").strip()
            article_url = f"https://www.medrxiv.org/content/{doi}v{version}" if version else f"https://www.medrxiv.org/content/{doi}"
            if abstract:
                item["abstract"] = abstract
                item["summary"] = abstract
                # Use abstract as text if full body text was unavailable
                item["text"] = item.get("text") or abstract
            item["source_url"] = item.get("source_url") or article_url
            item["url"] = item.get("url") or article_url
            item["pdf_url"] = item.get("pdf_url") or f"{article_url}.full.pdf"
            item["version"] = item.get("version") or (version or None)
            item["license"] = item.get("license") or metadata.get("license")
            item["published_journal_ref"] = item.get("published_journal_ref") or metadata.get("published")
            if metadata.get("jatsxml"):
                item["jats_xml_path"] = normalize_jats_url(metadata["jatsxml"])
        item["full_text_extracted"] = False
        return item, False

    if metadata is None:
        try:
            metadata = fetch_medrxiv_metadata(
                session,
                doi,
                timeout,
                published_date=item.get("published_date") or item.get("date"),
            )
        except Exception:
            metadata = None

    if metadata:
        version = str(metadata.get("version") or item.get("version") or "").strip()
        article_url = f"https://www.medrxiv.org/content/{doi}v{version}" if version else f"https://www.medrxiv.org/content/{doi}"
        item.setdefault("source_url", article_url)
        item.setdefault("url", article_url)
        item.setdefault("pdf_url", f"{article_url}.full.pdf")
        item.setdefault("version", version or None)
        item.setdefault("license", metadata.get("license"))
        item.setdefault("published_journal_ref", metadata.get("published"))

    now = utc_now_iso()
    item["jats_xml_path"] = resolved_jats_url
    item["full_text"] = full_text
    item["text"] = full_text
    item["full_text_extracted"] = True
    item["text_extraction_date"] = now
    item.pop("jats_fetch_error", None)
    return item, True

def main() -> None:
    load_env_file(PROJECT_ROOT / ".env")
    raw_dir = PROJECT_ROOT / "data" / "raw"
    parser = argparse.ArgumentParser(description="Enrich medRxiv JSONL records with JATS XML full text.")
    parser.add_argument("--input", type=Path, default=raw_dir / "medrxiv_papers.jsonl")
    parser.add_argument("--output", type=Path, help="Output JSONL path. Defaults to updating --input in place.")
    parser.add_argument("--limit", type=int, help="Process at most this many records.")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout per request in seconds.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Polite delay between records.")
    parser.add_argument("--no_rebuild", action="store_true", help="Skip rebuilding papers.jsonl after enrichment.")
    args = parser.parse_args()

    output_path = args.output or args.input
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    records = list(iter_jsonl(args.input))
    processed = 0
    enriched = 0
    session = configure_requests_session(requests.Session())
    # Longer connect timeout, larger read timeout for JATS XML downloads
    session.headers.update({"User-Agent": "capstone-medrxiv-enricher/1.0"})

    with temp_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc="Enriching medRxiv JATS"):
            if args.limit is not None and processed >= args.limit:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            if str(record.get("source") or "medrxiv").lower() == "medrxiv":
                processed += 1
                try:
                    record, ok = enrich_record(record, session, timeout=args.timeout)
                    if ok:
                        enriched += 1
                except Exception as exc:
                    record = dict(record)
                    record["jats_fetch_error"] = str(exc)
                time.sleep(args.sleep)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    temp_path.replace(output_path)
    result = {"processed": processed, "enriched": enriched, "output": str(output_path)}
    print(json.dumps(result, indent=2))

    if not args.no_rebuild:
        total = rebuild_papers_jsonl(output_path.parent)
        print(f"Rebuilt papers.jsonl: {total:,} total records.")


if __name__ == "__main__":
    main()
