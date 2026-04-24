#!/usr/bin/env python3
"""
Helpers for medRxiv full-text retrieval.

medRxiv exposes JATS XML paths in its details API. These XML documents are a
better ingestion source than PDFs for search/vectorization because they are
structured text and avoid PDF-download blocking.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urljoin

import requests


MEDRXIV_BASE_URL = "https://www.medrxiv.org"


def normalize_jats_url(jats_xml_path: str | None) -> str | None:
    value = (jats_xml_path or "").strip()
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        # Collapse consecutive slashes in path (keep the // after the scheme)
        proto, rest = value.split("://", 1)
        rest = re.sub(r"/{2,}", "/", rest)
        return f"{proto}://{rest}"
    # Relative path — collapse consecutive slashes before joining
    value = re.sub(r"/{2,}", "/", value)
    return urljoin(MEDRXIV_BASE_URL, value)


def get_jats_xml_path(*records: dict[str, Any] | None) -> str | None:
    """Return the JATS XML URL/path from medRxiv API variants."""
    keys = ("jatsxml", "jats_xml_path", "jats xml path")
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in keys:
            value = _clean_text(record.get(key))
            if value:
                return value
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return _clean_text(" ".join(element.itertext()))


def _first_text(root: ET.Element, tag_name: str) -> str:
    for element in root.iter():
        if _local_name(element.tag) == tag_name:
            return _element_text(element)
    return ""


def extract_jats_text(xml_text: str) -> str:
    root = ET.fromstring(xml_text)

    parts: list[str] = []
    title = _first_text(root, "article-title")
    abstract = _first_text(root, "abstract")
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)

    body = None
    for element in root.iter():
        if _local_name(element.tag) == "body":
            body = element
            break

    if body is not None:
        previous = ""
        for element in body.iter():
            tag = _local_name(element.tag)
            if tag not in {"title", "p"}:
                continue
            text = _element_text(element)
            if len(text) < 20 or text == previous:
                continue
            parts.append(text)
            previous = text

    # Preserve section boundaries enough for chunking while avoiding noisy XML.
    return "\n\n".join(part for part in parts if part)


def fetch_jats_full_text(
    session: requests.Session,
    jats_xml_path: str | None,
    *,
    timeout: int = 60,
) -> tuple[str | None, str | None, str | None]:
    url = normalize_jats_url(jats_xml_path)
    if not url:
        return None, None, "no JATS XML path"

    try:
        response = session.get(
            url,
            timeout=timeout,
            headers={
                "Accept": "application/xml,text/xml,*/*;q=0.8",
                "Referer": MEDRXIV_BASE_URL + "/",
                "User-Agent": "capstone-medrxiv-jats-fetcher/1.0",
            },
        )
        response.raise_for_status()
        text = extract_jats_text(response.text)
        if not text:
            return None, url, "JATS XML did not contain extractable text"
        return text, url, None
    except Exception as exc:
        return None, url, str(exc)
