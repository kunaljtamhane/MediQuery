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
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urljoin

import requests
from requests import Response
from tqdm import tqdm


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


class CorpusPDFDownloader:
    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        arxiv_count: int,
        pubmed_count: int,
        medrxiv_count: int,
        timeout: int = 60,
    ) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "download_manifest.jsonl"
        self.arxiv_count = arxiv_count
        self.pubmed_count = pubmed_count
        self.medrxiv_count = medrxiv_count
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "capstone-pdf-downloader/1.0",
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            }
        )

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
            return 0.5
        return 0.2

    def _select_items(self, filename: str, source: str, count: int) -> List[DownloadItem]:
        path = self.raw_dir / filename
        selected: List[DownloadItem] = []
        seen_ids: set[str] = set()

        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")

        for paper in iter_jsonl(path):
            pdf_url = paper.get("pdf_url")
            source_id = paper.get("source_id")
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
        items = []
        items.extend(self._select_items("arxiv_papers.jsonl", "arxiv", self.arxiv_count))
        items.extend(self._select_items("pubmed_papers.jsonl", "pubmed", self.pubmed_count))
        items.extend(self._select_items("medrxiv_papers.jsonl", "medrxiv", self.medrxiv_count))
        return items

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

        href_match = re.search(
            r'href=["\']([^"\']+/pdf/[^"\']+\.pdf)["\']',
            html,
            flags=re.IGNORECASE,
        )
        if href_match:
            return urljoin(base_url, href_match.group(1))

        return None

    def _download_binary(self, url: str, target_path: Path) -> None:
        temp_path = target_path.with_suffix(target_path.suffix + ".part")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

        with self.session.get(url, timeout=self.timeout, stream=True, allow_redirects=True) as response:
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

    def _download_one(self, item: DownloadItem) -> tuple[bool, Path, str | None]:
        target_path = self._target_path(item)
        if target_path.exists() and self._has_pdf_signature(target_path):
            return True, target_path, None
        self._remove_if_invalid_pdf(target_path)

        last_error: str | None = None
        for attempt in range(1, 4):
            try:
                with self.session.get(item.pdf_url, timeout=self.timeout, stream=True, allow_redirects=True) as response:
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

                    if "pdf" in content_type or final_url.lower().endswith(".pdf") or item.pdf_url.lower().endswith(".pdf"):
                        self._write_streamed_response(response, target_path)
                    elif item.source == "pubmed" and "html" in content_type:
                        html = response.text
                        resolved_pdf_url = self._extract_pdf_url_from_html(html, final_url)
                        if not resolved_pdf_url:
                            raise ValueError(f"Could not resolve direct PMC PDF URL from {final_url}")
                        self._download_binary(resolved_pdf_url, target_path)
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
        print(
            f"Planned downloads: arXiv={self.arxiv_count}, PubMed={self.pubmed_count}, "
            f"medRxiv={self.medrxiv_count}, total={len(items)}"
        )
        print(f"Output directory: {self.output_dir}")

        success = 0
        failed = 0
        for item in tqdm(items, desc="Downloading PDFs"):
            ok, target_path, error = self._download_one(item)
            if ok:
                success += 1
                self._write_manifest_entry(item, target_path, "downloaded")
            else:
                failed += 1
                self._write_manifest_entry(item, target_path, "failed", error=error)

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
    )
    downloader.run()


if __name__ == "__main__":
    main()
