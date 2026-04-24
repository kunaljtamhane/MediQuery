#!/usr/bin/env python3
"""
Enrich PubMed JSONL records with full-body text from PubMed Central.

For every PubMed record that has a pmc_id and no full_text yet, this script
fetches the PMC XML via NCBI eFetch, extracts the article body, and writes
the enriched records back in place.  Rebuilds papers.jsonl when done.

Usage:
    python data/collection/enrich_pubmed_pmc_full_text.py
    python data/collection/enrich_pubmed_pmc_full_text.py --batch_size 5 --no_rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

COLLECTION_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = COLLECTION_DIR.parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

sys.path.insert(0, str(COLLECTION_DIR))

from collection_utils import rebuild_papers_jsonl
from env_loader import configure_requests_session, load_env_file
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Section types that belong to back-matter and should be skipped
_SKIP_SEC_TYPES = frozenset({"ref", "references", "supplementary-material", "supplementary", "financial-disclosure", "conflict-interest"})


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _walk_body(element: ET.Element, parts: list[str]) -> None:
    tag = _local_name(element.tag)
    sec_type = (element.attrib.get("sec-type") or "").lower()
    if tag == "sec" and any(skip in sec_type for skip in _SKIP_SEC_TYPES):
        return
    if tag in ("title", "p", "label"):
        text = _clean(" ".join(element.itertext()))
        if len(text) >= 20:
            parts.append(text)
        return
    for child in element:
        _walk_body(child, parts)


def extract_pmc_full_text(article_xml: str) -> str | None:
    try:
        root = ET.fromstring(article_xml)
    except ET.ParseError:
        return None

    parts: list[str] = []

    # Abstract (under <front><article-meta><abstract>)
    for elem in root.iter():
        if _local_name(elem.tag) == "abstract":
            text = _clean(" ".join(elem.itertext()))
            if len(text) >= 20:
                parts.append(text)
            break

    # Body text
    for elem in root.iter():
        if _local_name(elem.tag) == "body":
            _walk_body(elem, parts)
            break

    # Fallback: article title if nothing found yet
    if not parts:
        for elem in root.iter():
            if _local_name(elem.tag) == "article-title":
                text = _clean(" ".join(elem.itertext()))
                if text:
                    parts.append(text)
                break

    joined = "\n\n".join(parts)
    return joined if len(joined) >= 100 else None

class PmcFetcher:
    def __init__(self, email: str = "", api_key: str = "") -> None:
        self.email = email
        self.api_key = api_key
        self.session = configure_requests_session(requests.Session())
        self.delay = 0.11 if api_key else 0.34

    def _params(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {"tool": "capstone-pmc-enricher", **extra}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def fetch_batch_xml(self, pmc_ids: List[str]) -> Optional[str]:
        numeric = [pid.replace("PMC", "") for pid in pmc_ids]
        params = self._params({
            "db": "pmc",
            "id": ",".join(numeric),
            "rettype": "xml",
            "retmode": "xml",
        })
        last_err: Exception | None = None
        for attempt in range(1, 5):
            try:
                r = self.session.get(
                    f"{NCBI_BASE_URL}/efetch.fcgi",
                    params=params,
                    timeout=120,
                )
                r.raise_for_status()
                time.sleep(self.delay)
                return r.text
            except requests.RequestException as exc:
                last_err = exc
                if attempt == 4:
                    print(f"  [PMC] efetch failed: {exc}")
                    return None
                time.sleep(attempt * 3)
        return None

    def fetch_batch(self, pmc_ids: List[str]) -> Dict[str, str]:
        """Return {pmc_id_with_prefix → full_text}."""
        xml_text = self.fetch_batch_xml(pmc_ids)
        if not xml_text:
            return {}

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            print(f"  [PMC] XML parse error: {exc}")
            return {}

        results: Dict[str, str] = {}
        for article in root.iter("article"):
            pmc_key = None
            for aid in article.findall(".//{*}article-id"):
                id_type = (aid.attrib.get("pub-id-type") or "").lower()
                text = (aid.text or "").strip()
                if not text:
                    continue
                if id_type == "pmcid":
                    # Value is already "PMC12345678"
                    pmc_key = text if text.startswith("PMC") else f"PMC{text}"
                    break
                if id_type == "pmcaid":
                    pmc_key = f"PMC{text}"
                    break
            if not pmc_key:
                continue
            full_text = extract_pmc_full_text(ET.tostring(article, encoding="unicode"))
            if full_text:
                results[pmc_key] = full_text

        return results


def main() -> None:
    load_env_file(PROJECT_ROOT / ".env")
    email = os.getenv("NCBI_EMAIL", "")
    api_key = os.getenv("NCBI_API_KEY", "")

    parser = argparse.ArgumentParser(description="Enrich PubMed records with PMC full-body text.")
    parser.add_argument("--input", type=Path, default=RAW_DIR / "pubmed_papers.jsonl")
    parser.add_argument("--batch_size", type=int, default=10, help="PMC IDs per eFetch call (max 100).")
    parser.add_argument("--no_rebuild", action="store_true", help="Skip rebuilding papers.jsonl.")
    args = parser.parse_args()

    output_path: Path = args.input
    temp_path = output_path.with_suffix(".jsonl.tmp")

    records: List[Dict[str, Any]] = []
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    to_enrich: List[tuple[int, str]] = []  # (index, pmc_id)
    already_done = 0
    for i, rec in enumerate(records):
        pmc_id = rec.get("pmc_id")
        if not pmc_id:
            continue
        if rec.get("full_text") and rec.get("full_text_extracted"):
            already_done += 1
            continue
        to_enrich.append((i, str(pmc_id)))

    print(f"PubMed records  : {len(records):,}")
    print(f"With PMC IDs    : {len(to_enrich) + already_done:,}")
    print(f"Already enriched: {already_done:,}")
    print(f"To enrich now   : {len(to_enrich):,}")

    if not to_enrich:
        print("Nothing to do.")
        return

    fetcher = PmcFetcher(email=email, api_key=api_key)
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    enriched = 0
    batch_size = min(max(1, args.batch_size), 100)

    # Build lookup: pmc_id → record index
    pmc_to_idx: Dict[str, int] = {pmc_id: idx for idx, pmc_id in to_enrich}
    pmc_ids_list = [pmc_id for _, pmc_id in to_enrich]

    for start in tqdm(range(0, len(pmc_ids_list), batch_size), desc="PMC full-text batches"):
        batch = pmc_ids_list[start : start + batch_size]
        results = fetcher.fetch_batch(batch)
        for pmc_id, full_text in results.items():
            idx = pmc_to_idx.get(pmc_id)
            if idx is None:
                continue
            rec = dict(records[idx])
            rec["full_text"] = full_text
            rec["full_text_extracted"] = True
            rec["text"] = full_text
            rec["text_extraction_date"] = now_iso
            records[idx] = rec
            enriched += 1

    print(f"\nEnriched {enriched:,} / {len(to_enrich):,} records with PMC full text.")

    with temp_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    temp_path.replace(output_path)
    print(f"Saved: {output_path}")

    if not args.no_rebuild:
        total = rebuild_papers_jsonl(output_path.parent)
        print(f"Rebuilt papers.jsonl: {total:,} total records.")


if __name__ == "__main__":
    main()
