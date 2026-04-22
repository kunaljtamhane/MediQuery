#!/usr/bin/env python3
"""
Download PDFs for the recent multi-source corpus.

Default selection:
    167 arXiv PDFs
    167 PubMed PDFs
    166 medRxiv PDFs

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
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from requests import Response
from tqdm import tqdm

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional runtime dependency
    PlaywrightTimeoutError = None
    sync_playwright = None


@dataclass
class DownloadItem:
    source: str
    source_id: str
    title: str
    pdf_url: str
    published_date: str


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
    return re.sub(r"/pdf/?$", "/", str(url))


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


class CorpusPDFDownloader:
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
        retry_failed_only: bool = False,
        retry_source: str | None = None,
        pubmed_recycle_every: int = 20,
        retry_manifest_offset: int = 0,
        retry_manifest_limit: int | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "download_manifest.jsonl"
        self.arxiv_count = arxiv_count
        self.pubmed_count = pubmed_count
        self.medrxiv_count = medrxiv_count
        self.timeout = timeout
        self.pubmed_mode = pubmed_mode
        self.pubmed_headless = pubmed_headless
        self.pubmed_browser_timeout_ms = pubmed_browser_timeout * 1000
        self.pubmed_profile_dir = pubmed_profile_dir or (self.raw_dir / "playwright_pubmed_profile")
        self.pubmed_profile_dir.mkdir(parents=True, exist_ok=True)
        self.retry_failed_only = retry_failed_only
        self.retry_source = retry_source
        self.pubmed_recycle_every = max(0, pubmed_recycle_every)
        self.retry_manifest_offset = max(0, retry_manifest_offset)
        self.retry_manifest_limit = retry_manifest_limit if retry_manifest_limit is None else max(0, retry_manifest_limit)
        self.session = requests.Session()
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
            if not pdf_url or not source_id or source_id in seen_ids:
                continue
            selected.append(
                DownloadItem(
                    source=source,
                    source_id=source_id,
                    title=title,
                    pdf_url=pdf_url,
                    published_date=paper.get("published_date", ""),
                )
            )
            seen_ids.add(source_id)
            if len(selected) >= count:
                break

        return selected

    def plan_downloads(self) -> List[DownloadItem]:
        if self.retry_failed_only:
            return self._load_failed_manifest_items()

        items = []
        items.extend(self._select_items("arxiv_papers.jsonl", "arxiv", self.arxiv_count))
        items.extend(self._select_items("pubmed_papers.jsonl", "pubmed", self.pubmed_count))
        items.extend(self._select_items("medrxiv_papers.jsonl", "medrxiv", self.medrxiv_count))
        return items

    def _load_failed_manifest_items(self) -> List[DownloadItem]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest for retry mode: {self.manifest_path}")

        source_limits = {
            "arxiv": self.arxiv_count,
            "pubmed": self.pubmed_count,
            "medrxiv": self.medrxiv_count,
        }
        candidates: List[DownloadItem] = []
        selected_counts = {key: 0 for key in source_limits}
        seen_keys: set[tuple[str, str]] = set()

        for entry in iter_jsonl(self.manifest_path):
            source = str(entry.get("source") or "").strip().lower()
            source_id = str(entry.get("source_id") or "").strip()
            pdf_url = entry.get("pdf_url")
            if entry.get("status") != "failed" or not source or not source_id or not pdf_url:
                continue
            if self.retry_source and source != self.retry_source:
                continue
            if source not in source_limits:
                continue
            if selected_counts[source] >= source_limits[source]:
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
                )
            )
            selected_counts[source] += 1
            seen_keys.add(key)

        start = self.retry_manifest_offset
        end = None if self.retry_manifest_limit is None else start + self.retry_manifest_limit
        return candidates[start:end]

    def _target_path(self, item: DownloadItem) -> Path:
        return self.output_dir / f"{item.source}_{sanitize_filename(item.source_id)}.pdf"

    def _write_manifest_entry(self, item: DownloadItem, file_path: Path, status: str, error: str | None = None) -> None:
        entry = {
            "source": item.source,
            "source_id": item.source_id,
            "title": item.title,
            "published_date": item.published_date,
            "pdf_url": item.pdf_url,
            "file_path": str(file_path),
            "status": status,
            "error": error,
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _extract_pdf_url_from_html(self, html: str, base_url: str) -> str | None:
        meta_match = re.search(
            r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            flags=re.IGNORECASE,
        )
        if meta_match:
            return urljoin(base_url, meta_match.group(1))

        pmc_pdf_match = re.search(
            r'href=["\']([^"\']*?/articles/PMC\d+/pdf/[^"\']+\.pdf)["\']',
            html,
            flags=re.IGNORECASE,
        )
        if pmc_pdf_match:
            return urljoin(base_url, pmc_pdf_match.group(1))

        # PMC article pages often expose the download action as a visible "PDF"
        # button whose href can vary by journal/platform.
        pdf_button_match = re.search(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>\s*PDF(?:\s*\([^<]+\))?\s*</a>',
            html,
            flags=re.IGNORECASE,
        )
        if pdf_button_match:
            return urljoin(base_url, pdf_button_match.group(1))

        pmc_asset_match = re.search(
            r'href=["\']([^"\']*(?:/pdf/|/bin/)[^"\']*\.pdf(?:\?[^"\']*)?)["\']',
            html,
            flags=re.IGNORECASE,
        )
        if pmc_asset_match:
            return urljoin(base_url, pmc_asset_match.group(1))

        href_match = re.search(
            r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
            html,
            flags=re.IGNORECASE,
        )
        if href_match:
            return urljoin(base_url, href_match.group(1))

        for pattern in [
            r'<iframe[^>]+src=["\']([^"\']+)["\']',
            r'<embed[^>]+src=["\']([^"\']+)["\']',
            r'<object[^>]+data=["\']([^"\']+)["\']',
            r'(?:src|data|content)=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        ]:
            match = re.search(pattern, html, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = urljoin(base_url, match.group(1))
            resolved = self._unwrap_pdf_viewer_url(candidate)
            if resolved:
                return resolved

        generic_url_match = re.search(
            r'https?://[^"\']+\.pdf(?:\?[^"\']*)?',
            html,
            flags=re.IGNORECASE,
        )
        if generic_url_match:
            resolved = self._unwrap_pdf_viewer_url(generic_url_match.group(0))
            if resolved:
                return resolved

        return None

    def _unwrap_pdf_viewer_url(self, url: str) -> str | None:
        candidate = str(url).strip()
        if not candidate:
            return None

        parsed = urlparse(candidate)
        lower_path = parsed.path.lower()
        if lower_path.endswith(".pdf") or "/pdf/" in lower_path:
            return candidate

        query = parse_qs(parsed.query)
        for key in ("src", "file", "url"):
            values = query.get(key) or []
            for value in values:
                decoded = unquote(value)
                decoded_parsed = urlparse(decoded)
                decoded_path = decoded_parsed.path.lower()
                if decoded_path.endswith(".pdf") or "/pdf/" in decoded_path:
                    return decoded

        return None

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
                    **extra,
                )
                self._pubmed_context = context
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

    def _browser_fetch_bytes(self, page: Any, url: str) -> tuple[int, str, str, bytes]:
        request_headers = self._request_headers("pubmed")
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
                "PMC PDF verification page detected. Complete it in the opened browser "
                "window, then wait for the site to continue."
            )
            page.wait_for_url(
                lambda current_url: not any(
                    marker in current_url.lower() for marker in ("challengepage", "recaptcha")
                ),
                timeout=self.pubmed_browser_timeout_ms,
            )
            self._safe_page_content(page)
        time.sleep(2)

    def _navigate_to_pdf(self, page: Any, pdf_link: str) -> bytes:
        """Fetch PDF bytes through the live browser session.

        We deliberately use in-page fetch() here instead of page.goto() because
        Chromium navigations to PDF assets are intercepted by the built-in PDF
        viewer and return viewer HTML, not the raw file bytes.
        """
        current = pdf_link
        for _ in range(3):
            status, final_url, _, body = self._browser_fetch_bytes(page, current)
            if status >= 400:
                raise RuntimeError(f"PDF fetch returned HTTP {status} for {current}")
            if not body or len(body) < 5:
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
                return False, target_path, last_error
            finally:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass

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

    def _download_one(self, item: DownloadItem) -> tuple[bool, Path, str | None]:
        target_path = self._target_path(item)
        if target_path.exists() and self._has_pdf_signature(target_path):
            return True, target_path, None
        self._remove_if_invalid_pdf(target_path)

        if item.source == "pubmed" and self.pubmed_mode == "playwright":
            return self._download_one_pubmed_playwright(item, target_path)

        last_error: str | None = None
        for attempt in range(1, 4):
            try:
                with self.session.get(
                    item.pdf_url,
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
                                or item.pdf_url.lower().endswith(".pdf")
                                or final_url.lower().endswith("/pdf/")
                                or final_url.lower().endswith("/pdf")
                                or item.pdf_url.lower().endswith("/pdf/")
                                or item.pdf_url.lower().endswith("/pdf")
                            )
                        )
                    )

                    if is_binary_pdf:
                        self._write_streamed_response(response, target_path)
                    elif item.source == "pubmed" and is_html:
                        # The initial request is streamed for binary downloads.
                        # PMC article pages can arrive truncated through that
                        # path, so fetch the full HTML page before scraping the
                        # "PDF" action link or citation_pdf_url metadata.
                        html = self._fetch_full_html(final_url, item.source)
                        resolved_pdf_url = self._extract_pdf_url_from_html(html, final_url)
                        if not resolved_pdf_url:
                            raise ValueError(f"Could not resolve direct PMC PDF URL from {final_url}")
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
                if status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
                    wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 3.0 * attempt)
                    time.sleep(wait_seconds)
                else:
                    time.sleep(attempt)
            except Exception as exc:
                last_error = str(exc)
                time.sleep(attempt)

            temp_path = target_path.with_suffix(target_path.suffix + ".part")
            temp_path.unlink(missing_ok=True)
            if target_path.exists() and (target_path.stat().st_size == 0 or not self._has_pdf_signature(target_path)):
                target_path.unlink(missing_ok=True)

        return False, target_path, last_error

    def run(self) -> dict:
        items = self.plan_downloads()
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
    parser.add_argument("--arxiv_count", type=int, default=167, help="Number of arXiv PDFs to download.")
    parser.add_argument("--pubmed_count", type=int, default=167, help="Number of PubMed PDFs to download.")
    parser.add_argument("--medrxiv_count", type=int, default=166, help="Number of medRxiv PDFs to download.")
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
        "--retry_failed_only",
        action="store_true",
        help="Retry only the failed entries recorded in output_dir/download_manifest.jsonl.",
    )
    parser.add_argument(
        "--retry_source",
        choices=("arxiv", "pubmed", "medrxiv"),
        help="When retrying from the manifest, limit retries to one source.",
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
        retry_failed_only=args.retry_failed_only,
        retry_source=args.retry_source,
        pubmed_recycle_every=args.pubmed_recycle_every,
        retry_manifest_offset=args.retry_manifest_offset,
        retry_manifest_limit=args.retry_manifest_limit,
    )
    downloader.run()


if __name__ == "__main__":
    main()
