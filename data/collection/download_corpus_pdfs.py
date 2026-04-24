#!/usr/bin/env python3
"""
Download PDFs for the recent multi-source corpus.

Default selection:
    125 arXiv PDFs
    250 PubMed PDFs
    125 medRxiv PDFs

Input files:
    data/raw/arxiv_papers.jsonl
    data/raw/pubmed_papers.jsonl
    data/raw/medrxiv_papers.jsonl

Output directory:
    data/raw/pdfs/
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urljoin

import requests
from requests import Response
from tqdm import tqdm

from download_corpus_pdf_browser_mixin import CorpusPDFBrowserMixin
from env_loader import configure_requests_session, load_env_file

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional runtime dependency
    PlaywrightTimeoutError = None
    sync_playwright = None

try:
    from selenium import webdriver
    from selenium.common.exceptions import (
        NoSuchElementException,
        StaleElementReferenceException,
        TimeoutException as SeleniumTimeoutException,
        WebDriverException,
    )
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.edge.options import Options as EdgeOptions
except ImportError:  # pragma: no cover - optional runtime dependency
    webdriver = None
    ChromeOptions = None
    EdgeOptions = None
    By = None
    NoSuchElementException = None
    StaleElementReferenceException = None
    SeleniumTimeoutException = None
    WebDriverException = None


@dataclass
class DownloadItem:
    source: str
    source_id: str
    title: str
    pdf_url: str
    published_date: str
    doi: str | None = None


def sanitize_filename(value: str) -> str:
    cleaned = value.replace("/", "_").replace("\\", "_").replace(":", "_")
    cleaned = cleaned.replace("*", "_").replace("?", "_").replace('"', "_")
    cleaned = cleaned.replace("<", "_").replace(">", "_").replace("|", "_")
    return cleaned


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_pmc_article_url(url: str) -> str:
    value = str(url)
    value = re.sub(r"/pdf/[^/?#]+\.pdf(?:\?[^#]*)?$", "/", value)
    return re.sub(r"/pdf/?(?:\?[^#]*)?$", "/", value)


def normalize_pmc_pdf_wrapper_url(url: str) -> str:
    value = str(url)
    if re.match(r"^https?://pmc\.ncbi\.nlm\.nih\.gov/articles/PMC[^/]+/?$", value):
        return value.rstrip("/") + "/pdf/"
    return value


def resolve_pdf_url(paper: Dict, source: str) -> str | None:
    raw_payload = paper.get("raw_payload") or {}
    pdf_url = paper.get("pdf_url") or raw_payload.get("pdf_url")
    if source == "arxiv":
        if pdf_url:
            return str(pdf_url)
        arxiv_id = paper.get("arxiv_id") or paper.get("source_id")
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        source_url = paper.get("source_url") or paper.get("url")
        if source_url and "/abs/" in str(source_url):
            return str(source_url).replace("/abs/", "/pdf/") + ".pdf"

    if source == "pubmed":
        if pdf_url:
            return str(pdf_url)
        pmc_id = paper.get("pmc_id")
        if pmc_id:
            return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/"

    if source == "medrxiv":
        if pdf_url:
            return str(pdf_url)
        landing_page = paper.get("source_url") or paper.get("url")
        if landing_page:
            landing_page = str(landing_page).rstrip("/")
            return f"{landing_page}.full.pdf"
        medrxiv_id = paper.get("medrxiv_id") or paper.get("doi") or paper.get("source_id")
        if medrxiv_id:
            return f"https://www.medrxiv.org/content/{medrxiv_id}v1.full.pdf"

    if pdf_url:
        return str(pdf_url)

    return None


class CorpusPDFDownloader(CorpusPDFBrowserMixin):
    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        arxiv_count: int,
        pubmed_count: int,
        medrxiv_count: int,
        timeout: int = 60,
        pubmed_mode: str = "http",
        pubmed_headless: bool = False,
        pubmed_browser_timeout: int = 120,
        pubmed_profile_dir: Path | None = None,
        medrxiv_headless: bool = False,
        medrxiv_browser_timeout: int = 180,
        medrxiv_sleep_seconds: float = 4.0,
        retry_failed_only: bool = False,
        retry_missing_only: bool = False,
        retry_source: str | None = None,
        skip_source_ids: set[str] | None = None,
        only_source_ids: set[str] | None = None,
        pubmed_recycle_every: int = 20,
        retry_manifest_offset: int = 0,
        retry_manifest_limit: int | None = None,
    ) -> None:
        load_env_file(Path(__file__).resolve().parents[2] / ".env")
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "download_manifest.jsonl"
        self.retry_manifest_backup_path = self.output_dir / "download_manifest.retry_input.jsonl"
        self.arxiv_count = arxiv_count
        self.pubmed_count = pubmed_count
        self.medrxiv_count = medrxiv_count
        self.timeout = timeout
        self.pubmed_mode = pubmed_mode
        self.pubmed_headless = pubmed_headless
        self.pubmed_browser_timeout_ms = pubmed_browser_timeout * 1000
        self.pubmed_profile_dir = pubmed_profile_dir or (self.raw_dir / "playwright_pubmed_profile")
        self.pubmed_profile_dir.mkdir(parents=True, exist_ok=True)
        self.medrxiv_headless = medrxiv_headless
        self.medrxiv_browser_timeout = max(30, medrxiv_browser_timeout)
        self.medrxiv_sleep_seconds = max(1.0, medrxiv_sleep_seconds)
        self.retry_failed_only = retry_failed_only
        self.retry_missing_only = retry_missing_only
        self.retry_source = retry_source
        self.skip_source_ids = {value.strip() for value in (skip_source_ids or set()) if value and value.strip()}
        self.only_source_ids = {value.strip() for value in (only_source_ids or set()) if value and value.strip()}
        self.pubmed_recycle_every = max(0, pubmed_recycle_every)
        self.retry_manifest_offset = max(0, retry_manifest_offset)
        self.retry_manifest_limit = retry_manifest_limit if retry_manifest_limit is None else max(0, retry_manifest_limit)
        self.session = configure_requests_session(requests.Session())
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._playwright: Any | None = None
        self._pubmed_context: Any | None = None
        self._pubmed_playwright_success_count = 0

    def _request_headers(self, source: str, *, html: bool = False) -> Dict[str, str]:
        if source == "pubmed":
            referer = "https://pmc.ncbi.nlm.nih.gov/"
        elif source == "medrxiv":
            referer = "https://www.medrxiv.org/"
        elif source == "arxiv":
            referer = "https://arxiv.org/"
        else:
            referer = "https://www.google.com/"

        accept = (
            "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"
            if html
            else "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8"
        )
        return {
            "Accept": accept,
            "Referer": referer,
        }

    def _has_pdf_signature(self, path: Path) -> bool:
        if not path.exists() or path.stat().st_size < 5:
            return False
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"

    def _remove_if_invalid_pdf(self, path: Path) -> None:
        if path.exists() and not self._has_pdf_signature(path):
            path.unlink(missing_ok=True)

    def _rate_limit_delay(self, item: DownloadItem) -> float:
        if item.source == "medrxiv":
            return 1.5
        if item.source == "pubmed":
            return 3.0 if self.pubmed_mode == "playwright" else 0.5
        return 0.2

    def _select_items(self, filename: str, source: str, count: int) -> List[DownloadItem]:
        if count <= 0:
            return []

        path = self.raw_dir / filename
        selected: List[DownloadItem] = []
        seen_ids: set[str] = set()

        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")

        for paper in iter_jsonl(path):
            pdf_url = resolve_pdf_url(paper, source)
            source_id = (
                paper.get("source_id")
                or paper.get("pubmed_id")
                or paper.get("arxiv_id")
                or paper.get("medrxiv_id")
                or paper.get("doi")
            )
            title = paper.get("title", "")
            source_id_text = str(source_id).strip() if source_id else ""
            if (
                not pdf_url
                or not source_id_text
                or source_id_text in seen_ids
                or source_id_text in self.skip_source_ids
                or (self.only_source_ids and source_id_text not in self.only_source_ids)
            ):
                continue
            doi = paper.get("doi")
            item = DownloadItem(
                source=source,
                source_id=source_id_text,
                title=title,
                pdf_url=pdf_url,
                published_date=paper.get("published_date", ""),
                doi=str(doi) if doi else None,
            )
            # Skip items whose target PDF already exists and is valid so reruns
            # focus on the remaining missing papers.
            if self._has_pdf_signature(self._target_path(item)):
                seen_ids.add(source_id_text)
                continue

            selected.append(item)
            seen_ids.add(source_id_text)
            if len(selected) >= count:
                break

        return selected

    def plan_downloads(self) -> List[DownloadItem]:
        if self.retry_failed_only:
            return self._load_failed_manifest_items()
        if self.retry_missing_only:
            return self._load_missing_items()

        items = []
        items.extend(self._select_items("arxiv_papers.jsonl", "arxiv", self.arxiv_count))
        items.extend(self._select_items("pubmed_papers.jsonl", "pubmed", self.pubmed_count))
        items.extend(self._select_items("medrxiv_papers.jsonl", "medrxiv", self.medrxiv_count))
        return items

    def _load_missing_items(self) -> List[DownloadItem]:
        items: List[DownloadItem] = []
        sources = [
            ("arxiv_papers.jsonl", "arxiv", self.arxiv_count),
            ("pubmed_papers.jsonl", "pubmed", self.pubmed_count),
            ("medrxiv_papers.jsonl", "medrxiv", self.medrxiv_count),
        ]
        for filename, source, count in sources:
            if self.retry_source and source != self.retry_source:
                continue
            if count <= 0:
                continue
            items.extend(self._select_items(filename, source, count))

        start = self.retry_manifest_offset
        end = None if self.retry_manifest_limit is None else start + self.retry_manifest_limit
        return items[start:end]

    def _load_failed_manifest_items(self) -> List[DownloadItem]:
        source_manifest = self.manifest_path
        if (
            source_manifest.exists()
            and source_manifest.stat().st_size == 0
            and self.retry_manifest_backup_path.exists()
            and self.retry_manifest_backup_path.stat().st_size > 0
        ):
            source_manifest = self.retry_manifest_backup_path
        elif not source_manifest.exists() and self.retry_manifest_backup_path.exists():
            source_manifest = self.retry_manifest_backup_path

        if not source_manifest.exists():
            raise FileNotFoundError(f"Missing manifest for retry mode: {self.manifest_path}")

        source_limits: Dict[str, int | None] = {
            "arxiv": self.arxiv_count,
            "pubmed": self.pubmed_count,
            "medrxiv": self.medrxiv_count,
        }
        candidates: List[DownloadItem] = []
        selected_counts = {key: 0 for key in source_limits}
        seen_keys: set[tuple[str, str]] = set()

        for entry in iter_jsonl(source_manifest):
            source = str(entry.get("source") or "").strip().lower()
            source_id = str(entry.get("source_id") or "").strip()
            pdf_url = entry.get("pdf_url")
            if entry.get("status") != "failed" or not source or not source_id or not pdf_url:
                continue
            if self.retry_source and source != self.retry_source:
                continue
            if source not in source_limits:
                continue
            limit = source_limits[source]
            if limit <= 0:
                continue
            if limit is not None and selected_counts[source] >= limit:
                continue

            key = (source, source_id)
            if key in seen_keys:
                continue

            candidates.append(
                DownloadItem(
                    source=source,
                    source_id=source_id,
                    title=str(entry.get("title") or ""),
                    pdf_url=str(pdf_url),
                    published_date=str(entry.get("published_date") or ""),
                    doi=(str(entry.get("doi")) if entry.get("doi") else None),
                )
            )
            selected_counts[source] += 1
            seen_keys.add(key)

        start = self.retry_manifest_offset
        end = None if self.retry_manifest_limit is None else start + self.retry_manifest_limit
        return candidates[start:end]

    def _resolve_pubmed_request_url(self, item: DownloadItem) -> tuple[str, str | None]:
        request_url = item.pdf_url
        fallback_pdf_url: str | None = None
        pmc_id = self._extract_pmc_id_from_url(item.pdf_url)
        if pmc_id:
            oa_pdf_url = self._get_pmc_oa_pdf_url(pmc_id)
            if oa_pdf_url:
                fallback_pdf_url = request_url
                request_url = oa_pdf_url
        return request_url, fallback_pdf_url

    def _target_path(self, item: DownloadItem) -> Path:
        return self.output_dir / f"{item.source}_{sanitize_filename(item.source_id)}.pdf"

    def _write_manifest_entry(self, item: DownloadItem, file_path: Path, status: str, error: str | None = None) -> None:
        entry = {
            "source": item.source,
            "source_id": item.source_id,
            "title": item.title,
            "published_date": item.published_date,
            "doi": item.doi,
            "pdf_url": item.pdf_url,
            "file_path": str(file_path),
            "status": status,
            "error": error,
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _download_binary(self, url: str, target_path: Path, source: str) -> None:
        temp_path = target_path.with_suffix(target_path.suffix + ".part")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

        with self.session.get(
            url,
            timeout=self.timeout,
            stream=True,
            allow_redirects=True,
            headers=self._request_headers(source),
        ) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            temp_path.unlink(missing_ok=True)
            raise ValueError("Downloaded file is empty")

        temp_path.replace(target_path)

    def _write_streamed_response(self, response: Response, target_path: Path) -> None:
        temp_path = target_path.with_suffix(target_path.suffix + ".part")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    handle.write(chunk)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            temp_path.unlink(missing_ok=True)
            raise ValueError("Downloaded file is empty")

        temp_path.replace(target_path)

    def _fetch_full_html(self, url: str, source: str) -> str:
        with self.session.get(
            url,
            timeout=self.timeout,
            allow_redirects=True,
            headers=self._request_headers(source, html=True),
        ) as response:
            response.raise_for_status()
            return response.text

    def _download_via_doi_fallback(self, item: DownloadItem, target_path: Path) -> tuple[bool, str | None]:
        if not item.doi:
            return False, "no DOI available"

        doi_url = f"https://doi.org/{item.doi}"
        html_headers = {
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        pdf_headers = {
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            with self.session.get(
                doi_url,
                timeout=self.timeout,
                allow_redirects=True,
                headers=html_headers,
            ) as response:
                response.raise_for_status()
                final_url = response.url
                content_type = response.headers.get("content-type", "").lower()

                if "pdf" in content_type or final_url.lower().endswith(".pdf"):
                    temp_path = target_path.with_suffix(target_path.suffix + ".part")
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                    temp_path.write_bytes(response.content)
                    temp_path.replace(target_path)
                    return True, None

                html = response.text
                resolved_pdf_url = self._extract_pdf_url_from_html(html, final_url)
                if not resolved_pdf_url:
                    browser_ok, browser_error = self._try_download_via_doi_fallback_browser(item, target_path)
                    if browser_ok:
                        return True, None
                    if browser_error and browser_error != "no DOI available":
                        return False, (
                            f"DOI landing page did not expose a PDF URL at {final_url}; "
                            f"browser DOI fallback failed: {browser_error}"
                        )
                    return False, f"DOI landing page did not expose a PDF URL at {final_url}"

            temp_path = target_path.with_suffix(target_path.suffix + ".part")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            with self.session.get(
                resolved_pdf_url,
                timeout=self.timeout,
                stream=True,
                allow_redirects=True,
                headers={**pdf_headers, "Referer": final_url},
            ) as response:
                response.raise_for_status()
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            handle.write(chunk)

            if not temp_path.exists() or temp_path.stat().st_size == 0:
                temp_path.unlink(missing_ok=True)
                browser_ok, browser_error = self._try_download_via_doi_fallback_browser(item, target_path)
                if browser_ok:
                    return True, None
                if browser_error and browser_error != "no DOI available":
                    return False, f"DOI fallback downloaded an empty file; browser DOI fallback failed: {browser_error}"
                return False, "DOI fallback downloaded an empty file"

            temp_path.replace(target_path)
            if not self._has_pdf_signature(target_path):
                target_path.unlink(missing_ok=True)
                browser_ok, browser_error = self._try_download_via_doi_fallback_browser(item, target_path)
                if browser_ok:
                    return True, None
                if browser_error and browser_error != "no DOI available":
                    return False, (
                        f"DOI fallback did not return a valid PDF from {resolved_pdf_url}; "
                        f"browser DOI fallback failed: {browser_error}"
                    )
                return False, f"DOI fallback did not return a valid PDF from {resolved_pdf_url}"
            return True, None
        except Exception as exc:
            browser_ok, browser_error = self._try_download_via_doi_fallback_browser(item, target_path)
            if browser_ok:
                return True, None
            if browser_error and browser_error != "no DOI available":
                return False, f"{exc}; browser DOI fallback failed: {browser_error}"
            return False, str(exc)

    def _try_download_via_doi_fallback_browser(
        self,
        item: DownloadItem,
        target_path: Path,
    ) -> tuple[bool, str | None]:
        try:
            return self._download_via_doi_fallback_browser(item, target_path)
        except Exception as exc:
            return False, str(exc)

    def _download_via_doi_fallback_browser(
        self,
        item: DownloadItem,
        target_path: Path,
    ) -> tuple[bool, str | None]:
        if not item.doi:
            return False, "no DOI available"

        context = self._ensure_pubmed_browser()
        page = None
        doi_url = f"https://doi.org/{item.doi}"
        try:
            page = context.new_page()
            page.goto(doi_url, wait_until="domcontentloaded", timeout=self.pubmed_browser_timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass
            self._dismiss_cookie_banners(page)
            page_html = self._safe_page_content(page)
            if self._is_verification_page(page.url, page_html):
                self._handle_pdf_verification_challenge(page, doi_url)
                self._dismiss_cookie_banners(page)
                page_html = self._safe_page_content(page)

            page_title = page.title().strip().lower()
            if page_title == "403" or "403 forbidden" in page_html[:1000].lower():
                raise RuntimeError(f"DOI landing page returned 403 Forbidden at {page.url}")

            pdf_link = self._discover_pdf_link_on_page(page)
            if pdf_link:
                try:
                    body = self._navigate_to_pdf(
                        page,
                        pdf_link,
                        headers={
                            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
                            "Referer": page.url,
                        },
                    )
                    target_path.write_bytes(body)
                    if not self._has_pdf_signature(target_path):
                        target_path.unlink(missing_ok=True)
                        raise ValueError(f"DOI browser fallback did not return a valid PDF from {pdf_link}")
                    return True, None
                except Exception as fetch_exc:
                    if self._page_looks_like_pdf_viewer(page):
                        viewer_ok, viewer_error = self._try_pdf_viewer_download(page, target_path)
                        if viewer_ok:
                            return True, None
                        saved_ok, saved_error = self._save_page_as_pdf(page, target_path)
                        if saved_ok:
                            return True, None
                        raise RuntimeError(
                            f"{fetch_exc}; PDF viewer fallback failed: {viewer_error or saved_error}"
                        ) from fetch_exc
                    if self._open_full_text_article_view(page):
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=10_000)
                        except Exception:
                            pass
                        if self._page_looks_like_full_text_article(page):
                            saved_ok, saved_error = self._save_page_as_pdf(page, target_path)
                            if saved_ok:
                                return True, None
                            raise RuntimeError(
                                f"{fetch_exc}; full-text HTML to PDF fallback failed: {saved_error}"
                            ) from fetch_exc
                    if self._page_looks_like_full_text_article(page):
                        saved_ok, saved_error = self._save_page_as_pdf(page, target_path)
                        if saved_ok:
                            return True, None
                        raise RuntimeError(
                            f"{fetch_exc}; article page HTML to PDF fallback failed: {saved_error}"
                        ) from fetch_exc
                    click_ok, click_error = self._try_browser_download_click(page, target_path)
                    if click_ok:
                        return True, None
                    raise RuntimeError(
                            f"{fetch_exc}; browser click download failed: {click_error}"
                        ) from fetch_exc

            if self._page_looks_like_pdf_viewer(page):
                viewer_ok, viewer_error = self._try_pdf_viewer_download(page, target_path)
                if viewer_ok:
                    return True, None
                saved_ok, saved_error = self._save_page_as_pdf(page, target_path)
                if saved_ok:
                    return True, None
                raise RuntimeError(f"PDF viewer fallback failed: {viewer_error or saved_error}")

            if self._open_full_text_article_view(page):
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=10_000)
                except Exception:
                    pass
                if self._page_looks_like_full_text_article(page):
                    saved_ok, saved_error = self._save_page_as_pdf(page, target_path)
                    if saved_ok:
                        return True, None
                    raise RuntimeError(f"full-text HTML to PDF fallback failed: {saved_error}")

            if self._page_looks_like_full_text_article(page):
                saved_ok, saved_error = self._save_page_as_pdf(page, target_path)
                if saved_ok:
                    return True, None
                raise RuntimeError(f"article page HTML to PDF fallback failed: {saved_error}")

            click_ok, click_error = self._try_browser_download_click(page, target_path)
            if click_ok:
                return True, None
            raise ValueError(
                f"DOI landing page did not expose a PDF URL at {page.url}; "
                f"browser click download failed: {click_error}"
            )
        except Exception as exc:
            return False, str(exc)
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

    def _ensure_pubmed_browser(self) -> Any:
        if self._pubmed_context is not None:
            return self._pubmed_context

        if sync_playwright is None:
            raise RuntimeError(
                "Playwright is not installed. Run 'pip install -r data/collection/requirements.txt' first."
            )

        self._playwright = sync_playwright().start()
        browser_type = self._playwright.chromium
        launch_attempts = [
            {"channel": "msedge"},
            {"channel": "chrome"},
            {},
        ]
        last_error: Exception | None = None
        for extra in launch_attempts:
            try:
                context = browser_type.launch_persistent_context(
                    user_data_dir=str(self.pubmed_profile_dir),
                    accept_downloads=True,
                    headless=self.pubmed_headless,
                    downloads_path=str(self.output_dir),
                    ignore_default_args=["--enable-automation"],
                    no_viewport=True,
                    locale="en-US",
                    timezone_id="America/Chicago",
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-features=Translate,OptimizationHints,MediaRouter",
                        "--start-maximized",
                    ],
                    **extra,
                )
                self._pubmed_context = context
                self._configure_pubmed_browser_context(context)
                # Warm up the session so PMC sees a real browser visit
                # before we start requesting article pages.
                warm_page = context.new_page()
                try:
                    warm_page.goto("https://www.ncbi.nlm.nih.gov/", wait_until="domcontentloaded", timeout=30_000)
                except Exception:
                    pass
                finally:
                    warm_page.close()
                return context
            except Exception as exc:  # pragma: no cover - depends on local browser install
                last_error = exc

        raise RuntimeError(
            "Unable to launch a Playwright browser. Install Microsoft Edge or Chrome, "
            "or run 'playwright install chromium'."
        ) from last_error

    def _configure_pubmed_browser_context(self, context: Any) -> None:
        try:
            context.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
        except Exception:
            pass

        try:
            context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {
                  get: () => undefined,
                });
                Object.defineProperty(navigator, 'platform', {
                  get: () => 'Win32',
                });
                Object.defineProperty(navigator, 'language', {
                  get: () => 'en-US',
                });
                Object.defineProperty(navigator, 'languages', {
                  get: () => ['en-US', 'en'],
                });
                Object.defineProperty(navigator, 'hardwareConcurrency', {
                  get: () => 8,
                });
                window.chrome = window.chrome || { runtime: {} };
                """
            )
        except Exception:
            pass

    def _browser_fetch_bytes(
        self,
        page: Any,
        url: str,
        headers: Dict[str, str] | None = None,
    ) -> tuple[int, str, str, bytes]:
        request_headers = headers or self._request_headers("pubmed")
        try:
            response = page.context.request.get(
                url,
                headers=request_headers,
                fail_on_status_code=False,
                timeout=self.pubmed_browser_timeout_ms,
            )
            headers = response.headers or {}
            return int(response.status), str(response.url), str(headers.get("content-type", "")), response.body()
        except Exception:
            payload = page.evaluate(
                """
                async ({ targetUrl, headers }) => {
                  const response = await fetch(targetUrl, {
                    credentials: 'include',
                    redirect: 'follow',
                    headers,
                  });
                  const buffer = await response.arrayBuffer();
                  const bytes = new Uint8Array(buffer);
                  const chunkSize = 0x8000;
                  let binary = '';
                  for (let index = 0; index < bytes.length; index += chunkSize) {
                    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
                  }
                  return {
                    ok: response.ok,
                    status: response.status,
                    url: response.url,
                    contentType: response.headers.get('content-type') || '',
                    bodyBase64: btoa(binary),
                  };
                }
                """,
                {"targetUrl": url, "headers": request_headers},
            )
            body = base64.b64decode(payload["bodyBase64"]) if payload.get("bodyBase64") else b""
            return int(payload["status"]), str(payload["url"]), str(payload["contentType"]), body

    def _is_verification_page(self, url: str, html: str) -> bool:
        url_lower = (url or "").lower()
        html_lower = (html or "")[:6000].lower()
        return any(
            marker in url_lower or marker in html_lower
            for marker in (
                "challengepage",
                "recaptcha",
                "google.com/recaptcha",
                "www.google.com",
                "gstatic.com",
                "csp.withgoogle.com",
                "verify you are human",
                "security verification",
                "performing security verification",
                "not a bot",
                "security service to protect against malicious bots",
                "checking your browser",
                "cf-browser-verification",
                "cloudflare",
                "captcha-delivery",
                "just a moment",
                "please enable javascript and cookies",
                "needs to review the security of your connection",
                "checking if the site connection is secure",
                "sorry, you have been blocked",
                "attention required",
            )
        )

    def _safe_page_content(self, page: Any, retries: int = 5) -> str:
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                page.wait_for_load_state("domcontentloaded", timeout=5_000)
            except Exception:
                pass
            try:
                return page.content()
            except Exception as exc:
                last_error = exc
                if "page is navigating" in str(exc).lower():
                    time.sleep(1)
                    continue
                raise
        raise RuntimeError(f"Unable to retrieve stable page content: {last_error}")

    def _handle_pdf_verification_challenge(self, page: Any, url: str) -> None:
        page.goto(url, wait_until="domcontentloaded", timeout=self.pubmed_browser_timeout_ms)
        page_html = self._safe_page_content(page)
        if self._is_verification_page(page.url, page_html):
            print(
                "Security verification page detected. Complete it in the opened browser "
                "window, then wait for the site to continue."
            )
            deadline = time.monotonic() + (self.pubmed_browser_timeout_ms / 1000)
            while time.monotonic() < deadline:
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=5_000)
                except Exception:
                    pass
                current_html = self._safe_page_content(page, retries=2)
                if not self._is_verification_page(page.url, current_html):
                    page_html = current_html
                    break
                time.sleep(2)
            else:
                raise RuntimeError(f"Security verification did not complete in time for {page.url}")
        time.sleep(2)

    def _navigate_to_pdf(
        self,
        page: Any,
        pdf_link: str,
        headers: Dict[str, str] | None = None,
    ) -> bytes:
        """Fetch PDF bytes through the live browser session.

        We deliberately use in-page fetch() here instead of page.goto() because
        Chromium navigations to PDF assets are intercepted by the built-in PDF
        viewer and return viewer HTML, not the raw file bytes.
        """
        current = pdf_link
        for _ in range(3):
            status, final_url, _, body = self._browser_fetch_bytes(page, current, headers=headers)
            if status >= 400:
                try:
                    page.goto(current, wait_until="domcontentloaded", timeout=self.pubmed_browser_timeout_ms)
                    page_html = self._safe_page_content(page, retries=2)
                    if self._is_verification_page(page.url, page_html):
                        self._handle_pdf_verification_challenge(page, current)
                        continue
                except Exception:
                    pass
                raise RuntimeError(f"PDF fetch returned HTTP {status} for {current}")
            if not body or len(body) < 5:
                try:
                    page.goto(current, wait_until="domcontentloaded", timeout=self.pubmed_browser_timeout_ms)
                    page_html = self._safe_page_content(page, retries=2)
                    if self._is_verification_page(page.url, page_html):
                        self._handle_pdf_verification_challenge(page, current)
                        continue
                except Exception:
                    pass
                raise ValueError(f"Empty response for {current}")
            if body.startswith(b"%PDF-"):
                return body

            # HTML response — find a deeper PDF link and follow it.
            html = body.decode("utf-8", errors="ignore")
            if self._is_verification_page(final_url or current, html):
                self._handle_pdf_verification_challenge(page, current)
                continue

            next_url = self._extract_pdf_url_from_html(html, final_url or current)
            if not next_url or next_url == current:
                raise ValueError(
                    f"Response is not a PDF (starts with {body[:60]!r}) "
                    f"and no PDF link found in HTML at {current}"
                )
            current = next_url
        raise ValueError(f"Could not retrieve a PDF from {pdf_link} within 3 hops")

    def _download_one_pubmed_playwright(self, item: DownloadItem, target_path: Path) -> tuple[bool, Path, str | None]:
        if target_path.exists() and self._has_pdf_signature(target_path):
            return True, target_path, None
        self._remove_if_invalid_pdf(target_path)

        last_error: str | None = None
        for browser_attempt in range(1, 4):
            context = self._ensure_pubmed_browser()
            page = None
            try:
                page = context.new_page()
                article_url = normalize_pmc_article_url(item.pdf_url)
                page.goto(article_url, wait_until="domcontentloaded", timeout=self.pubmed_browser_timeout_ms)

                if "recaptcha" in page.url.lower() or "challengepage" in page.url.lower():
                    print(
                        f"PMC verification page detected for {item.source_id}. "
                        "Complete it in the opened browser window and wait for the article page."
                    )
                    page.wait_for_url(
                        lambda url: "pmc.ncbi.nlm.nih.gov/articles/" in url and "challengepage" not in url.lower(),
                        timeout=self.pubmed_browser_timeout_ms,
                    )

                page_html = self._safe_page_content(page)
                page_title = page.title().strip().lower()
                if page_title == "403" or "403 forbidden" in page_html[:1000].lower():
                    raise RuntimeError("PMC article page returned 403 Forbidden")

                # Determine the direct PDF URL — prefer HTML/selector discovery
                # over the generic /pdf/ wrapper so we get the actual asset URL.
                pdf_link: str | None = self._extract_pdf_url_from_html(page_html, page.url)

                if not pdf_link:
                    for selector in [
                        "a:has-text('PDF')",
                        "a[href*='/pdf/']",
                        "a[href$='.pdf']",
                    ]:
                        locator = page.locator(selector).first
                        if locator.count():
                            href = locator.get_attribute("href")
                            if href:
                                pdf_link = urljoin(page.url, href)
                                break

                if not pdf_link:
                    pdf_link = normalize_pmc_pdf_wrapper_url(item.pdf_url)

                body = self._navigate_to_pdf(page, pdf_link)

                target_path.write_bytes(body)
                self._pubmed_playwright_success_count += 1
                if (
                    self.pubmed_recycle_every
                    and self._pubmed_playwright_success_count % self.pubmed_recycle_every == 0
                ):
                    self.close()
                time.sleep(self._rate_limit_delay(item))
                return True, target_path, None

            except Exception as exc:
                last_error = str(exc)
                # A 403 means the article is restricted — the browser session
                # is still healthy, so do NOT restart it. Only restart if the
                # browser itself has crashed or been closed.
                if (
                    "Target page, context or browser has been closed" in last_error
                ) and browser_attempt < 3:
                    self.close()
                    self._remove_if_invalid_pdf(target_path)
                    time.sleep(15 * browser_attempt)
                    continue
                doi_ok, doi_error = self._download_via_doi_fallback(item, target_path)
                if doi_ok:
                    return True, target_path, None
                if doi_error and doi_error != "no DOI available":
                    last_error = f"{last_error}; DOI fallback failed: {doi_error}"
                return False, target_path, last_error
            finally:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass

        return False, target_path, last_error

    def _download_one_medrxiv_playwright(self, item: DownloadItem, target_path: Path) -> tuple[bool, Path, str | None]:
        if target_path.exists() and self._has_pdf_signature(target_path):
            return True, target_path, None
        self._remove_if_invalid_pdf(target_path)

        last_error: str | None = None
        for browser_attempt in range(1, 4):
            context = self._ensure_pubmed_browser()
            page = None
            try:
                page = context.new_page()
                api_article_url, api_pdf_url = self._resolve_medrxiv_urls_via_api(item)
                article_url = api_article_url or self._normalize_medrxiv_article_url(item.pdf_url)
                page.goto(article_url, wait_until="domcontentloaded", timeout=self.pubmed_browser_timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=10_000)
                except Exception:
                    pass

                page_html = self._safe_page_content(page)
                if self._is_verification_page(page.url, page_html):
                    print(
                        f"medRxiv security verification detected for {item.source_id}. "
                        "Complete it in the opened browser window and wait for the site to continue."
                    )
                    self._handle_pdf_verification_challenge(page, item.pdf_url)
                    page_html = self._safe_page_content(page)

                page_title = page.title().strip().lower()
                if page_title == "403" or "403 forbidden" in page_html[:1000].lower():
                    raise RuntimeError("medRxiv page returned 403 Forbidden")

                pdf_link = self._discover_pdf_link_on_page(page)
                if not pdf_link:
                    for selector in [
                        "a[href*='.full.pdf']",
                        "a[href$='.pdf']",
                        "a:has-text('PDF')",
                        "a:has-text('Download PDF')",
                    ]:
                        locator = page.locator(selector).first
                        if locator.count():
                            href = locator.get_attribute("href")
                            if href:
                                pdf_link = urljoin(page.url, href)
                                break
                if not pdf_link:
                    pdf_link = api_pdf_url or item.pdf_url

                body = self._navigate_to_pdf(
                    page,
                    pdf_link,
                    headers=self._request_headers("medrxiv"),
                )
                target_path.write_bytes(body)
                if not self._has_pdf_signature(target_path):
                    target_path.unlink(missing_ok=True)
                    raise ValueError("medRxiv browser download did not return a valid PDF")

                time.sleep(self._rate_limit_delay(item))
                return True, target_path, None
            except Exception as exc:
                last_error = str(exc)
                if (
                    "Target page, context or browser has been closed" in last_error
                ) and browser_attempt < 3:
                    self.close()
                    self._remove_if_invalid_pdf(target_path)
                    time.sleep(10 * browser_attempt)
                    continue
                return False, target_path, last_error
            finally:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass

        return False, target_path, last_error

    def _build_selenium_browser(self) -> Any:
        if webdriver is None:
            raise RuntimeError(
                "Selenium is not installed. Add 'selenium' to data/collection/requirements.txt and install it first."
            )

        download_dir = str(self.output_dir.resolve())
        launchers: list[tuple[str, Any]] = []
        if EdgeOptions is not None:
            edge_options = EdgeOptions()
            edge_options.use_chromium = True
            edge_options.add_experimental_option(
                "prefs",
                {
                    "download.default_directory": download_dir,
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "plugins.always_open_pdf_externally": True,
                },
            )
            edge_options.add_argument("--disable-blink-features=AutomationControlled")
            edge_options.add_argument("--start-maximized")
            if self.medrxiv_headless:
                edge_options.add_argument("--headless=new")
            launchers.append(("edge", edge_options))

        if ChromeOptions is not None:
            chrome_options = ChromeOptions()
            chrome_options.add_experimental_option(
                "prefs",
                {
                    "download.default_directory": download_dir,
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "plugins.always_open_pdf_externally": True,
                },
            )
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--start-maximized")
            if self.medrxiv_headless:
                chrome_options.add_argument("--headless=new")
            launchers.append(("chrome", chrome_options))

        last_error: Exception | None = None
        for browser_name, options in launchers:
            try:
                if browser_name == "edge":
                    driver = webdriver.Edge(options=options)
                else:
                    driver = webdriver.Chrome(options=options)
                driver.set_page_load_timeout(self.medrxiv_browser_timeout)
                return driver
            except Exception as exc:  # pragma: no cover - depends on local browser install
                last_error = exc

        raise RuntimeError(
            "Unable to launch a Selenium browser. Install Microsoft Edge or Chrome and ensure Selenium can access a driver."
        ) from last_error

    def _wait_for_verification_clear_selenium(self, driver: Any, label: str) -> None:
        deadline = time.monotonic() + self.medrxiv_browser_timeout
        while time.monotonic() < deadline:
            try:
                current_url = str(driver.current_url or "")
            except Exception:
                current_url = ""
            try:
                html = str(driver.page_source or "")
            except Exception:
                html = ""
            if not self._is_verification_page(current_url, html):
                return
            print(
                f"{label} security verification detected. Waiting {self.medrxiv_sleep_seconds:.0f}s "
                "for the site to clear automatically."
            )
            time.sleep(self.medrxiv_sleep_seconds)
        raise RuntimeError(f"{label} security verification did not clear in time")

    def _wait_for_downloaded_pdf(
        self,
        before_files: set[str],
        target_path: Path,
    ) -> tuple[bool, str | None]:
        deadline = time.monotonic() + self.medrxiv_browser_timeout
        while time.monotonic() < deadline:
            partials = {
                entry.name
                for entry in self.output_dir.glob("*")
                if entry.is_file() and entry.suffix.lower() in {".crdownload", ".tmp", ".part"}
            }
            for candidate in self.output_dir.glob("*.pdf"):
                if candidate.name in before_files:
                    continue
                if candidate.resolve() == target_path.resolve():
                    if self._has_pdf_signature(candidate):
                        return True, None
                    continue
                if any(candidate.name + suffix in partials for suffix in ("", ".crdownload", ".tmp", ".part")):
                    continue
                if not self._has_pdf_signature(candidate):
                    candidate.unlink(missing_ok=True)
                    continue
                target_path.unlink(missing_ok=True)
                os.replace(candidate, target_path)
                return True, None
            time.sleep(1)
        return False, "Timed out waiting for Selenium PDF download"

    def _find_medrxiv_pdf_url_selenium(self, driver: Any, fallback_pdf_url: str) -> str:
        candidates = [
            "//a[contains(@href, '.full.pdf')]",
            "//a[contains(@href, '.pdf')]",
            "//a[contains(normalize-space(.), 'Download PDF')]",
            "//a[contains(normalize-space(.), 'PDF')]",
        ]
        for xpath in candidates:
            try:
                element = driver.find_element(By.XPATH, xpath)
                href = element.get_attribute("href")
            except Exception:
                continue
            if href:
                return urljoin(str(driver.current_url or ""), href)

        try:
            html = str(driver.page_source or "")
        except Exception:
            html = ""
        match = re.search(r'href=["\']([^"\']+\.full\.pdf[^"\']*)["\']', html, flags=re.IGNORECASE)
        if match:
            return urljoin(str(driver.current_url or ""), match.group(1))
        return fallback_pdf_url

    def _sync_selenium_cookies_to_session(self, driver: Any) -> None:
        for cookie in driver.get_cookies():
            name = cookie.get("name")
            value = cookie.get("value")
            if not name or value is None:
                continue
            self.session.cookies.set(
                name,
                value,
                domain=cookie.get("domain"),
                path=cookie.get("path") or "/",
            )

    def _download_medrxiv_pdf_with_selenium_cookies(
        self,
        driver: Any,
        pdf_url: str,
        target_path: Path,
    ) -> None:
        self._sync_selenium_cookies_to_session(driver)
        try:
            user_agent = driver.execute_script("return navigator.userAgent")
        except Exception:
            user_agent = self.session.headers.get("User-Agent")

        headers = self._request_headers("medrxiv")
        headers.update(
            {
                "User-Agent": str(user_agent or self.session.headers.get("User-Agent", "")),
                "Referer": str(driver.current_url or "https://www.medrxiv.org/"),
            }
        )

        temp_path = target_path.with_suffix(target_path.suffix + ".part")
        temp_path.unlink(missing_ok=True)
        with self.session.get(
            pdf_url,
            timeout=self.timeout,
            stream=True,
            allow_redirects=True,
            headers=headers,
        ) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            temp_path.unlink(missing_ok=True)
            raise ValueError("Downloaded medRxiv PDF is empty")
        temp_path.replace(target_path)
        if not self._has_pdf_signature(target_path):
            with target_path.open("rb") as handle:
                prefix = handle.read(80)
            target_path.unlink(missing_ok=True)
            raise ValueError(f"medRxiv response is not a PDF (starts with {prefix!r})")

    def _download_one_medrxiv_selenium(self, item: DownloadItem, target_path: Path) -> tuple[bool, Path, str | None]:
        if target_path.exists() and self._has_pdf_signature(target_path):
            return True, target_path, None
        self._remove_if_invalid_pdf(target_path)

        api_article_url, api_pdf_url = self._resolve_medrxiv_urls_via_api(item)
        article_url = api_article_url or self._normalize_medrxiv_article_url(item.pdf_url)
        fallback_pdf_url = api_pdf_url or item.pdf_url
        last_error: str | None = None

        for attempt in range(1, 3):
            driver = None
            try:
                driver = self._build_selenium_browser()
                before_files = {entry.name for entry in self.output_dir.glob("*.pdf")}

                driver.get(article_url)
                time.sleep(self.medrxiv_sleep_seconds)
                self._wait_for_verification_clear_selenium(driver, "medRxiv")
                time.sleep(self.medrxiv_sleep_seconds)

                pdf_url = self._find_medrxiv_pdf_url_selenium(driver, fallback_pdf_url)
                try:
                    self._download_medrxiv_pdf_with_selenium_cookies(driver, pdf_url, target_path)
                    time.sleep(self._rate_limit_delay(item))
                    return True, target_path, None
                except Exception as exc:
                    last_error = str(exc)

                download_candidates = [
                    "//a[contains(@href, '.full.pdf')]",
                    "//a[contains(normalize-space(.), 'Download PDF')]",
                    "//a[contains(normalize-space(.), 'PDF')]",
                ]
                clicked = False
                for xpath in download_candidates:
                    try:
                        element = driver.find_element(By.XPATH, xpath)
                    except Exception:
                        continue
                    try:
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                    except Exception:
                        pass
                    time.sleep(self.medrxiv_sleep_seconds)
                    try:
                        element.click()
                    except Exception:
                        href = element.get_attribute("href")
                        if href:
                            driver.get(href)
                        else:
                            continue
                    clicked = True
                    break

                if not clicked:
                    driver.get(fallback_pdf_url)

                time.sleep(self.medrxiv_sleep_seconds)
                self._wait_for_verification_clear_selenium(driver, "medRxiv PDF")
                ok, download_error = self._wait_for_downloaded_pdf(before_files, target_path)
                if ok:
                    time.sleep(self._rate_limit_delay(item))
                    return True, target_path, None
                last_error = download_error
            except Exception as exc:
                last_error = str(exc)
            finally:
                if driver is not None:
                    try:
                        driver.quit()
                    except Exception:
                        pass
            if attempt < 2:
                time.sleep(self.medrxiv_sleep_seconds * attempt)

        return False, target_path, last_error

    def close(self) -> None:
        try:
            if self._pubmed_context is not None:
                self._pubmed_context.close()
        except Exception:
            pass
        finally:
            self._pubmed_context = None
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        finally:
            self._playwright = None

    def _download_one_via_http(self, item: DownloadItem, target_path: Path) -> tuple[bool, Path, str | None]:
        target_path = self._target_path(item)
        if target_path.exists() and self._has_pdf_signature(target_path):
            return True, target_path, None
        self._remove_if_invalid_pdf(target_path)
        request_url = item.pdf_url
        fallback_pdf_url: str | None = None
        if item.source == "pubmed":
            request_url, fallback_pdf_url = self._resolve_pubmed_request_url(item)

        last_error: str | None = None
        for attempt in range(1, 4):
            try:
                with self.session.get(
                    request_url,
                    timeout=self.timeout,
                    stream=True,
                    allow_redirects=True,
                    headers=self._request_headers(item.source),
                ) as response:
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 3.0 * attempt)
                        raise requests.HTTPError(
                            f"429 Client Error: Too Many Requests for url: {response.url}",
                            response=response,
                        ) from None

                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    final_url = response.url
                    is_html = "html" in content_type
                    is_binary_pdf = (
                        "pdf" in content_type
                        or "octet-stream" in content_type
                        or (
                            not is_html
                            and (
                                final_url.lower().endswith(".pdf")
                                or request_url.lower().endswith(".pdf")
                                or final_url.lower().endswith("/pdf/")
                                or final_url.lower().endswith("/pdf")
                                or request_url.lower().endswith("/pdf/")
                                or request_url.lower().endswith("/pdf")
                            )
                        )
                    )

                    if is_binary_pdf:
                        self._write_streamed_response(response, target_path)
                    elif item.source in {"pubmed", "medrxiv"} and is_html:
                        # The initial request is streamed for binary downloads.
                        # Article pages can arrive truncated through that path,
                        # so fetch the full HTML page before scraping for a PDF
                        # action link or citation_pdf_url metadata.
                        html = self._fetch_full_html(final_url, item.source)
                        resolved_pdf_url = self._extract_pdf_url_from_html(html, final_url)

                        # For PMC: if scraping the /pdf/ wrapper page failed, also try
                        # the plain article page which reliably has citation_pdf_url meta tags.
                        if not resolved_pdf_url and item.source == "pubmed":
                            article_url = normalize_pmc_article_url(final_url)
                            if article_url != final_url:
                                try:
                                    article_html = self._fetch_full_html(article_url, item.source)
                                    resolved_pdf_url = self._extract_pdf_url_from_html(article_html, article_url)
                                except Exception:
                                    pass

                        # PMC Open-Access API — returns a direct FTP/HTTPS PDF link for OA articles.
                        if not resolved_pdf_url and item.source == "pubmed":
                            pmc_id = self._extract_pmc_id_from_url(item.pdf_url or final_url)
                            if pmc_id:
                                resolved_pdf_url = self._get_pmc_oa_pdf_url(pmc_id)

                        if not resolved_pdf_url and item.source == "medrxiv":
                            resolved_pdf_url = fallback_pdf_url
                        if not resolved_pdf_url:
                            if item.source == "pubmed":
                                raise ValueError(f"Could not resolve direct PMC PDF URL from {final_url}")
                            raise ValueError(f"Could not resolve direct medRxiv PDF URL from {final_url}")
                        self._download_binary(resolved_pdf_url, target_path, item.source)
                    else:
                        raise ValueError(f"Unexpected content-type {content_type or 'unknown'}")

                if not self._has_pdf_signature(target_path):
                    raise ValueError("Downloaded file does not have a valid PDF signature")

                time.sleep(self._rate_limit_delay(item))
                return True, target_path, None
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                last_error = str(exc)
                if item.source == "pubmed" and fallback_pdf_url and request_url != fallback_pdf_url:
                    request_url = fallback_pdf_url
                    fallback_pdf_url = None
                    time.sleep(1)
                    continue
                if status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
                    wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 3.0 * attempt)
                    time.sleep(wait_seconds)
                else:
                    time.sleep(attempt)
            except Exception as exc:
                last_error = str(exc)
                if item.source == "pubmed" and fallback_pdf_url and request_url != fallback_pdf_url:
                    request_url = fallback_pdf_url
                    fallback_pdf_url = None
                    time.sleep(1)
                    continue
                time.sleep(attempt)

            temp_path = target_path.with_suffix(target_path.suffix + ".part")
            temp_path.unlink(missing_ok=True)
            if target_path.exists() and (target_path.stat().st_size == 0 or not self._has_pdf_signature(target_path)):
                target_path.unlink(missing_ok=True)

        return False, target_path, last_error

    def _download_one_medrxiv(self, item: DownloadItem, target_path: Path) -> tuple[bool, Path, str | None]:
        ok, resolved_path, error = self._download_one_via_http(item, target_path)
        if ok:
            return ok, resolved_path, error

        last_error = error
        if sync_playwright is not None:
            ok, resolved_path, playwright_error = self._download_one_medrxiv_playwright(item, target_path)
            if ok:
                return ok, resolved_path, playwright_error
            if playwright_error:
                last_error = f"{last_error}; Playwright fallback failed: {playwright_error}" if last_error else playwright_error

        ok, resolved_path, selenium_error = self._download_one_medrxiv_selenium(item, target_path)
        if ok:
            return ok, resolved_path, selenium_error
        if selenium_error:
            last_error = f"{last_error}; Selenium fallback failed: {selenium_error}" if last_error else selenium_error
        return False, resolved_path, last_error

    def _download_one(self, item: DownloadItem) -> tuple[bool, Path, str | None]:
        target_path = self._target_path(item)
        if item.source == "pubmed" and self.pubmed_mode == "playwright":
            return self._download_one_pubmed_playwright(item, target_path)
        if item.source == "medrxiv":
            return self._download_one_medrxiv(item, target_path)
        return self._download_one_via_http(item, target_path)

    def run(self) -> dict:
        items = self.plan_downloads()
        if self.retry_failed_only and self.manifest_path.exists() and self.manifest_path.stat().st_size > 0:
            shutil.copy2(self.manifest_path, self.retry_manifest_backup_path)
        if items or not (self.retry_failed_only or self.retry_missing_only):
            self.manifest_path.write_text("", encoding="utf-8")
        print("=" * 60)
        print("Corpus PDF Download")
        print("=" * 60)
        if self.retry_failed_only:
            retry_label = self.retry_source or "all sources"
            print(f"Retry mode: failed manifest entries only ({retry_label})")
            print(
                f"Retry manifest slice: offset={self.retry_manifest_offset}, "
                f"limit={self.retry_manifest_limit if self.retry_manifest_limit is not None else 'all'}"
            )
        elif self.retry_missing_only:
            retry_label = self.retry_source or "all sources"
            print(f"Retry mode: missing local PDFs only ({retry_label})")
            print(
                f"Retry slice: offset={self.retry_manifest_offset}, "
                f"limit={self.retry_manifest_limit if self.retry_manifest_limit is not None else 'all'}"
            )
        print(
            f"Planned downloads: arXiv={self.arxiv_count}, PubMed={self.pubmed_count}, "
            f"medRxiv={self.medrxiv_count}, total={len(items)}"
        )
        print(f"Output directory: {self.output_dir}")
        if self.pubmed_mode == "playwright":
            print(f"PubMed download mode: playwright (headless={self.pubmed_headless})")
            print(f"PubMed browser profile: {self.pubmed_profile_dir}")

        success = 0
        failed = 0
        try:
            for item in tqdm(items, desc="Downloading PDFs"):
                ok, target_path, error = self._download_one(item)
                if ok:
                    success += 1
                    self._write_manifest_entry(item, target_path, "downloaded")
                else:
                    failed += 1
                    self._write_manifest_entry(item, target_path, "failed", error=error)
        finally:
            self.close()

        summary = {"planned": len(items), "downloaded": success, "failed": failed}
        print(json.dumps(summary, indent=2))
        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Download PDFs for the recent multi-source corpus.")
    parser.add_argument("--raw_dir", type=Path, default=project_root / "data" / "raw", help="Directory containing source JSONL files.")
    parser.add_argument("--output_dir", type=Path, default=project_root / "data" / "raw" / "pdfs", help="Directory to save PDFs.")
    parser.add_argument("--arxiv_count", type=int, default=125, help="Number of arXiv PDFs to download.")
    parser.add_argument("--pubmed_count", type=int, default=250, help="Number of PubMed PDFs to download.")
    parser.add_argument("--medrxiv_count", type=int, default=125, help="Number of medRxiv PDFs to download.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--pubmed_mode",
        choices=("http", "playwright"),
        default="http",
        help="How PubMed/PMC PDFs should be downloaded.",
    )
    parser.add_argument(
        "--pubmed_headless",
        action="store_true",
        help="Run the Playwright PubMed browser in headless mode.",
    )
    parser.add_argument(
        "--pubmed_browser_timeout",
        type=int,
        default=120,
        help="Per-paper browser timeout in seconds for Playwright PubMed downloads.",
    )
    parser.add_argument(
        "--pubmed_profile_dir",
        type=Path,
        default=project_root / "data" / "raw" / "playwright_pubmed_profile",
        help="Persistent browser profile directory for Playwright PubMed downloads.",
    )
    parser.add_argument(
        "--medrxiv_headless",
        action="store_true",
        help="Run the Selenium medRxiv browser in headless mode.",
    )
    parser.add_argument(
        "--medrxiv_browser_timeout",
        type=int,
        default=180,
        help="Per-paper browser timeout in seconds for Selenium medRxiv downloads.",
    )
    parser.add_argument(
        "--medrxiv_sleep_seconds",
        type=float,
        default=4.0,
        help="Sleep interval in seconds between Selenium medRxiv automation steps.",
    )
    parser.add_argument(
        "--retry_failed_only",
        action="store_true",
        help="Retry only the failed entries recorded in output_dir/download_manifest.jsonl.",
    )
    parser.add_argument(
        "--retry_missing_only",
        action="store_true",
        help="Retry items whose target PDFs are still missing from output_dir, ignoring the manifest state.",
    )
    parser.add_argument(
        "--retry_source",
        choices=("arxiv", "pubmed", "medrxiv"),
        help="When retrying from the manifest, limit retries to one source.",
    )
    parser.add_argument(
        "--skip_source_id",
        action="append",
        default=[],
        help="Skip one or more source_id values when selecting items to download. Repeat the flag for multiple IDs.",
    )
    parser.add_argument(
        "--only_source_id",
        action="append",
        default=[],
        help="Restrict selection to one or more source_id values. Repeat the flag for multiple IDs.",
    )
    parser.add_argument(
        "--only_source_id_file",
        action="append",
        default=[],
        type=Path,
        help="Path to a text file containing one source_id per line to regenerate.",
    )
    parser.add_argument(
        "--pubmed_recycle_every",
        type=int,
        default=20,
        help="Recycle the Playwright PubMed browser after this many successful PMC downloads.",
    )
    parser.add_argument(
        "--retry_manifest_offset",
        type=int,
        default=0,
        help="When retrying from the manifest, start at this failed-entry offset.",
    )
    parser.add_argument(
        "--retry_manifest_limit",
        type=int,
        help="When retrying from the manifest, process at most this many failed entries.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    only_source_ids = {value.strip() for value in args.only_source_id if value and value.strip()}
    for path in args.only_source_id_file:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    only_source_ids.add(value)
    downloader = CorpusPDFDownloader(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        arxiv_count=args.arxiv_count,
        pubmed_count=args.pubmed_count,
        medrxiv_count=args.medrxiv_count,
        timeout=args.timeout,
        pubmed_mode=args.pubmed_mode,
        pubmed_headless=args.pubmed_headless,
        pubmed_browser_timeout=args.pubmed_browser_timeout,
        pubmed_profile_dir=args.pubmed_profile_dir,
        medrxiv_headless=args.medrxiv_headless,
        medrxiv_browser_timeout=args.medrxiv_browser_timeout,
        medrxiv_sleep_seconds=args.medrxiv_sleep_seconds,
        retry_failed_only=args.retry_failed_only,
        retry_missing_only=args.retry_missing_only,
        retry_source=args.retry_source,
        skip_source_ids=set(args.skip_source_id),
        only_source_ids=only_source_ids,
        pubmed_recycle_every=args.pubmed_recycle_every,
        retry_manifest_offset=args.retry_manifest_offset,
        retry_manifest_limit=args.retry_manifest_limit,
    )
    downloader.run()


if __name__ == "__main__":
    main()
