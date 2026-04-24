from __future__ import annotations

import html
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse


class CorpusPDFBrowserMixin:
    def _extract_pmc_id_from_url(self, url: str) -> str | None:
        match = re.search(r"/(PMC\d+)", str(url or ""), re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _get_pmc_oa_pdf_url(self, pmc_id: str) -> str | None:
        """Query NCBI's PMC Open-Access API to get a direct PDF download link."""
        try:
            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"
            response = self.session.get(
                oa_url,
                timeout=self.timeout,
                headers=self._request_headers("pubmed", html=True),
            )
            response.raise_for_status()
            xml_text = response.text
            match = re.search(
                r'<link[^>]+format=["\']pdf["\'][^>]+href=["\']([^"\']+)["\']',
                xml_text,
                flags=re.IGNORECASE,
            )
            if not match:
                match = re.search(
                    r'<link[^>]+href=["\']([^"\']+)["\'][^>]+format=["\']pdf["\']',
                    xml_text,
                    flags=re.IGNORECASE,
                )
            if match:
                href = match.group(1).strip()
                if href.startswith("ftp://"):
                    href = "https://" + href[6:]
                return href
        except Exception:
            pass
        return None

    def _extract_pdf_url_from_html(self, html_text: str, base_url: str) -> str | None:
        meta_match = re.search(
            r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
            html_text,
            flags=re.IGNORECASE,
        )
        if meta_match:
            return urljoin(base_url, meta_match.group(1))

        pmc_pdf_match = re.search(
            r'href=["\']([^"\']*?/articles/PMC\d+/pdf/[^"\']+\.pdf)["\']',
            html_text,
            flags=re.IGNORECASE,
        )
        if pmc_pdf_match:
            return urljoin(base_url, pmc_pdf_match.group(1))

        pdf_button_match = re.search(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>\s*PDF(?:\s*\([^<]+\))?\s*</a>',
            html_text,
            flags=re.IGNORECASE,
        )
        if pdf_button_match:
            return urljoin(base_url, pdf_button_match.group(1))

        pmc_asset_match = re.search(
            r'href=["\']([^"\']*(?:/pdf/|/bin/)[^"\']*\.pdf(?:\?[^"\']*)?)["\']',
            html_text,
            flags=re.IGNORECASE,
        )
        if pmc_asset_match:
            return urljoin(base_url, pmc_asset_match.group(1))

        href_match = re.search(
            r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
            html_text,
            flags=re.IGNORECASE,
        )
        if href_match:
            return urljoin(base_url, href_match.group(1))

        structured_href_patterns = [
            r'href=["\']([^"\']*/doi/(?:pdf|epdf)/[^"\']+)["\']',
            r'href=["\']([^"\']*/fulltext/[^"\']+)["\'][^>]*>\s*(?:download|view)?\s*pdf\s*</a>',
            r'href=["\']([^"\']*viewcontent\.cgi\?[^"\']+)["\']',
            r'href=["\']([^"\']*showPdf[^"\']+)["\']',
            r'href=["\']([^"\']*download[^"\']*pdf[^"\']*)["\']',
        ]
        for pattern in structured_href_patterns:
            match = re.search(pattern, html_text, flags=re.IGNORECASE)
            if match:
                return urljoin(base_url, match.group(1))

        for pattern in [
            r'<iframe[^>]+src=["\']([^"\']+)["\']',
            r'<embed[^>]+src=["\']([^"\']+)["\']',
            r'<object[^>]+data=["\']([^"\']+)["\']',
            r'(?:src|data|content)=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        ]:
            match = re.search(pattern, html_text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = urljoin(base_url, match.group(1))
            resolved = self._unwrap_pdf_viewer_url(candidate)
            if resolved:
                return resolved

        generic_url_match = re.search(
            r'https?://[^"\']+\.pdf(?:\?[^"\']*)?',
            html_text,
            flags=re.IGNORECASE,
        )
        if generic_url_match:
            resolved = self._unwrap_pdf_viewer_url(generic_url_match.group(0))
            if resolved:
                return resolved

        return None

    def _normalize_medrxiv_article_url(self, url: str) -> str:
        candidate = str(url).strip()
        if not candidate:
            return candidate
        if candidate.endswith(".full.pdf"):
            return candidate[: -len(".full.pdf")]
        if candidate.endswith(".pdf"):
            return candidate[: -len(".pdf")]
        return candidate

    def _resolve_medrxiv_urls_via_api(self, item: DownloadItem) -> tuple[str | None, str | None]:
        doi = str(item.doi or item.source_id or "").strip()
        if not doi:
            return None, None

        def request_collection(api_url: str) -> list:
            response = self.session.get(
                api_url,
                timeout=self.timeout,
                headers=self._request_headers("medrxiv", html=True),
            )
            response.raise_for_status()
            payload = response.json()
            collection = payload.get("collection") or []
            return collection if isinstance(collection, list) else []

        try:
            collection = request_collection(f"https://api.medrxiv.org/details/medrxiv/{quote(doi, safe='/')}/na/json")
            if not collection:
                date_match = re.search(r"\d{4}-\d{2}-\d{2}", item.published_date or "")
                if date_match:
                    day = date_match.group(0)
                    cursor = 0
                    while True:
                        page = request_collection(f"https://api.medrxiv.org/details/medrxiv/{day}/{day}/{cursor}/json")
                        for candidate in page:
                            if str(candidate.get("doi") or "").strip().lower() == doi.lower():
                                collection = [candidate]
                                break
                        if collection or len(page) < 100:
                            break
                        cursor += len(page)
        except Exception:
            return None, None

        if not collection:
            return None, None

        record = collection[0] or {}
        record_doi = str(record.get("doi") or doi).strip()
        version = str(record.get("version") or "").strip()
        if not record_doi:
            return None, None

        article_url = (
            f"https://www.medrxiv.org/content/{record_doi}v{version}"
            if version
            else f"https://www.medrxiv.org/content/{record_doi}"
        )
        return article_url, f"{article_url}.full.pdf"

    def _discover_pdf_link_on_page(self, page: Any) -> str | None:
        page_html = self._safe_page_content(page)
        discovered = self._extract_pdf_url_from_html(page_html, page.url)
        if discovered:
            return discovered

        try:
            anchors = page.locator("a").evaluate_all(
                """
                (elements) => elements.map((element) => ({
                  href: element.href || element.getAttribute('href') || '',
                  text: (element.innerText || element.textContent || '').trim(),
                  title: (element.getAttribute('title') || '').trim(),
                  aria: (element.getAttribute('aria-label') || '').trim()
                }))
                """
            )
        except Exception:
            anchors = []

        candidates: list[str] = []
        for anchor in anchors:
            href = str(anchor.get("href") or "").strip()
            label = " ".join(
                str(anchor.get(key) or "").strip().lower() for key in ("text", "title", "aria")
            )
            if not href:
                continue
            href_lower = href.lower()
            if (
                ".pdf" in href_lower
                or "/doi/pdf/" in href_lower
                or "/doi/epdf/" in href_lower
                or "/download" in href_lower
                or "showpdf" in href_lower
                or "viewcontent.cgi" in href_lower
                or ("pdf" in label and "doi/full/" in href_lower)
                or ("pdf" in label and "/fulltext/" in href_lower)
                or (
                    "download" in label
                    and ("/article/" in href_lower or "/medical/" in href_lower or "/download" in href_lower)
                )
            ):
                candidates.append(urljoin(page.url, href))

        current_url = page.url.lower()
        if "tandfonline.com/doi/full/" in current_url:
            candidates.append(page.url.replace("/doi/full/", "/doi/pdf/"))
            candidates.append(page.url.replace("/doi/full/", "/doi/epdf/"))
        if "journals.lww.com" in current_url and "/fulltext/" in current_url:
            candidates.append(page.url.replace("/fulltext/", "/pdf/"))
        if "brieflands.com" in current_url and not current_url.endswith(".pdf"):
            candidates.append(page.url.rstrip("/") + ".pdf")

        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            return normalized
        return None

    def _dismiss_cookie_banners(self, page: Any) -> None:
        selectors = [
            "button:has-text('Accept all cookies')",
            "button:has-text('Accept All Cookies')",
            "button:has-text('Accept cookies')",
            "button:has-text('Accept')",
            "button:has-text('I agree')",
            "button:has-text('Agree')",
            "a:has-text('Accept all cookies')",
            "a:has-text('Accept cookies')",
            "[id*='accept' i]",
            "[class*='accept' i]",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if not locator.count():
                    continue
                locator.click(timeout=2_000)
                time.sleep(0.5)
            except Exception:
                continue

    def _extract_locator_target_url(self, page: Any, locator: Any) -> str | None:
        attribute_names = ("href", "data-href", "data-download-url")
        for attribute_name in attribute_names:
            try:
                value = locator.get_attribute(attribute_name, timeout=2_000)
            except Exception:
                value = None
            if value:
                candidate = html.unescape(str(value).strip())
                if candidate and not candidate.lower().startswith("javascript:"):
                    return urljoin(page.url, candidate)

        try:
            onclick = locator.get_attribute("onclick", timeout=2_000)
        except Exception:
            onclick = None
        if onclick:
            onclick_text = html.unescape(str(onclick).strip())
            for pattern in (
                r"""window\.open\(\s*['"]([^'"]+)['"]""",
                r"""location(?:\.href)?\s*=\s*['"]([^'"]+)['"]""",
                r"""open\(\s*['"]([^'"]+)['"]""",
            ):
                match = re.search(pattern, onclick_text, flags=re.IGNORECASE)
                if match:
                    return urljoin(page.url, match.group(1).strip())
        return None

    def _is_forbidden_page(self, page_url: str | None, page_html: str, page_title: str | None = None) -> bool:
        title_text = str(page_title or "").strip().lower()
        html_lower = str(page_html or "").lower()
        url_text = str(page_url or "").lower()
        return (
            title_text == "403"
            or "403 forbidden" in title_text
            or "<h1>403 forbidden</h1>" in html_lower
            or "<title>403 forbidden</title>" in html_lower
            or ("forbidden" in html_lower[:2000] and "cgi/viewcontent.cgi" in url_text)
        )

    def _try_direct_link_target(self, page: Any, locator: Any, target_path: Path) -> tuple[bool, str | None]:
        direct_url = self._extract_locator_target_url(page, locator)
        if not direct_url:
            return False, None

        try:
            body = self._navigate_to_pdf(
                page,
                direct_url,
                headers={
                    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
                    "Referer": page.url,
                },
            )
            target_path.write_bytes(body)
            if self._has_pdf_signature(target_path):
                return True, None
            target_path.unlink(missing_ok=True)
        except Exception as fetch_exc:
            fetch_error = str(fetch_exc)
        else:
            fetch_error = "direct link did not return a valid PDF"

        try:
            page.goto(direct_url, wait_until="domcontentloaded", timeout=15_000)
            ok, page_error = self._handle_post_click_page(page, target_path)
            if ok:
                return True, None
            return False, page_error or fetch_error
        except Exception as nav_exc:
            return False, f"{fetch_error}; browser navigation to direct link failed: {nav_exc}"

    def _try_browser_download_click(self, page: Any, target_path: Path) -> tuple[bool, str | None]:
        selectors = [
            "a[title*='PDF' i]",
            "a[aria-label*='PDF' i]",
            "a[title*='Download' i]",
            "a[aria-label*='Download' i]",
            "a[href*='/doi/pdf/']",
            "a[href*='/doi/epdf/']",
            "a[href*='.pdf']",
            "a[href*='/download']",
            "a:has-text('Download PDF')",
            "a:has-text('Download')",
            "a:has-text('View PDF')",
            "a:has-text('PDF')",
            "button:has-text('Download PDF')",
            "button:has-text('Download')",
            "button:has-text('View PDF')",
            "button:has-text('PDF')",
        ]

        last_error: str | None = None
        for selector in selectors:
            try:
                self._dismiss_cookie_banners(page)
                locator = page.locator(selector).first
                if not locator.count():
                    continue
                temp_path = target_path.with_suffix(target_path.suffix + ".part")
                temp_path.unlink(missing_ok=True)
                with page.expect_download(timeout=15_000) as download_info:
                    locator.click(timeout=5_000)
                download = download_info.value
                download.save_as(str(temp_path))
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    temp_path.unlink(missing_ok=True)
                    last_error = f"download click produced an empty file via selector {selector}"
                    continue
                temp_path.replace(target_path)
                if not self._has_pdf_signature(target_path):
                    target_path.unlink(missing_ok=True)
                    last_error = f"download click did not return a valid PDF via selector {selector}"
                    continue
                return True, None
            except Exception as exc:
                last_error = str(exc)
                try:
                    self._dismiss_cookie_banners(page)
                    locator = page.locator(selector).first
                    if not locator.count():
                        continue
                    direct_ok, direct_error = self._try_direct_link_target(page, locator, target_path)
                    if direct_ok:
                        return True, None
                    if direct_error:
                        last_error = direct_error

                    before_files = {entry.name for entry in self.output_dir.glob("*.pdf")}
                    original_url = str(page.url or "")
                    before_page_count = len(page.context.pages)

                    popup_page = None
                    try:
                        with page.context.expect_page(timeout=5_000) as popup_info:
                            locator.click(timeout=5_000)
                        popup_page = popup_info.value
                    except Exception:
                        locator.click(timeout=5_000)

                    downloaded_path, wait_error = self._wait_for_new_downloaded_pdf(
                        before_files,
                        target_path,
                        timeout_seconds=20,
                    )
                    if downloaded_path is not None:
                        return True, None

                    candidate_pages = []
                    if popup_page is not None:
                        candidate_pages.append(popup_page)
                    current_pages = page.context.pages
                    if len(current_pages) > before_page_count:
                        for extra_page in current_pages[before_page_count:]:
                            if extra_page not in candidate_pages:
                                candidate_pages.append(extra_page)
                    if str(page.url or "") != original_url:
                        candidate_pages.append(page)

                    for candidate_page in candidate_pages:
                        ok, page_error = self._handle_post_click_page(candidate_page, target_path)
                        if ok:
                            return True, None
                        last_error = page_error or last_error

                    last_error = wait_error or last_error
                except Exception:
                    pass

        return False, last_error or "no browser download action found"

    def _open_full_text_article_view(self, page: Any) -> bool:
        selectors = [
            "a:has-text('Read this article')",
            "button:has-text('Read this article')",
            "a:has-text('Full Article')",
            "button:has-text('Full Article')",
            "a:has-text('View full text')",
            "button:has-text('View full text')",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if not locator.count():
                    continue
                locator.click(timeout=5_000)
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=10_000)
                except Exception:
                    pass
                return True
            except Exception:
                continue
        return False

    def _page_looks_like_full_text_article(self, page: Any) -> bool:
        try:
            html_text = self._safe_page_content(page).lower()
        except Exception:
            return False
        markers = (
            "<h1",
            "abstract",
            "references",
            "research article",
            "full article",
            "introduction",
            "materials and methods",
            "results",
            "discussion",
        )
        score = sum(1 for marker in markers if marker in html_text)
        return score >= 3

    def _page_looks_like_pdf_viewer(self, page: Any) -> bool:
        current_url = str(page.url or "").lower()
        if any(marker in current_url for marker in ("/doi/epdf/", "/doi/pdf/", ".full.pdf")):
            return True
        try:
            html_text = self._safe_page_content(page).lower()
        except Exception:
            return False
        return any(
            marker in html_text
            for marker in (
                "page 1 /",
                "aria-label=\"download\"",
                "pdf viewer",
                "viewer",
                "toolbarviewer",
                "download",
            )
        )

    def _save_page_as_pdf(self, page: Any, target_path: Path) -> tuple[bool, str | None]:
        temp_path = target_path.with_suffix(target_path.suffix + ".part")
        temp_path.unlink(missing_ok=True)
        try:
            page.pdf(
                path=str(temp_path),
                print_background=True,
                format="A4",
                margin={
                    "top": "0.4in",
                    "right": "0.35in",
                    "bottom": "0.4in",
                    "left": "0.35in",
                },
            )
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            return False, str(exc)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            temp_path.unlink(missing_ok=True)
            return False, "page.pdf() produced an empty file"

        temp_path.replace(target_path)
        if not self._has_pdf_signature(target_path):
            target_path.unlink(missing_ok=True)
            return False, "page.pdf() output was not a valid PDF"
        return True, None

    def _handle_post_click_page(self, page: Any, target_path: Path) -> tuple[bool, str | None]:
        try:
            self._dismiss_cookie_banners(page)
        except Exception:
            pass

        try:
            page_html = self._safe_page_content(page)
        except Exception as exc:
            return False, str(exc)

        try:
            page_title = page.title()
        except Exception:
            page_title = ""

        if self._is_forbidden_page(page.url, page_html, page_title):
            return False, f"Post-click page returned 403 Forbidden at {page.url}"

        if self._is_verification_page(page.url, page_html):
            try:
                self._handle_pdf_verification_challenge(page, page.url)
            except Exception as exc:
                return False, str(exc)

        if self._page_looks_like_pdf_viewer(page):
            viewer_ok, viewer_error = self._try_pdf_viewer_download(page, target_path)
            if viewer_ok:
                return True, None
            saved_ok, saved_error = self._save_page_as_pdf(page, target_path)
            if saved_ok:
                return True, None
            return False, viewer_error or saved_error

        if self._page_looks_like_full_text_article(page):
            saved_ok, saved_error = self._save_page_as_pdf(page, target_path)
            if saved_ok:
                return True, None
            return False, saved_error

        current_url = str(page.url or "").lower()
        if current_url.endswith(".pdf") or "/pdf/" in current_url or "/download" in current_url:
            try:
                body = self._navigate_to_pdf(page, page.url)
                target_path.write_bytes(body)
                if self._has_pdf_signature(target_path):
                    return True, None
                target_path.unlink(missing_ok=True)
            except Exception as exc:
                return False, str(exc)

        return False, f"Post-click page did not expose a downloadable PDF at {page.url}"

    def _wait_for_new_downloaded_pdf(
        self,
        before_files: set[str],
        target_path: Path,
        timeout_seconds: int | None = None,
    ) -> tuple[Path | None, str | None]:
        deadline = time.monotonic() + float(timeout_seconds or (self.pubmed_browser_timeout_ms / 1000))
        while time.monotonic() < deadline:
            partials = {
                entry.name
                for entry in self.output_dir.glob("*")
                if entry.is_file() and entry.suffix.lower() in {".crdownload", ".tmp", ".part"}
            }
            for candidate in self.output_dir.glob("*.pdf"):
                if candidate.name in before_files:
                    continue
                if any(candidate.name + suffix in partials for suffix in ("", ".crdownload", ".tmp", ".part")):
                    continue
                if not self._has_pdf_signature(candidate):
                    continue
                if candidate.resolve() != target_path.resolve():
                    target_path.unlink(missing_ok=True)
                    candidate.replace(target_path)
                return target_path, None
            time.sleep(1)
        return None, "Timed out waiting for browser PDF download"

    def _try_pdf_viewer_download(self, page: Any, target_path: Path) -> tuple[bool, str | None]:
        before_files = {entry.name for entry in self.output_dir.glob("*.pdf")}

        selectors = [
            "[aria-label*='download' i]",
            "[title*='download' i]",
            "button[aria-label*='download' i]",
            "button[title*='download' i]",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if not locator.count():
                    continue
                with page.expect_download(timeout=10_000) as download_info:
                    locator.click(timeout=5_000)
                download = download_info.value
                temp_path = target_path.with_suffix(target_path.suffix + ".part")
                temp_path.unlink(missing_ok=True)
                download.save_as(str(temp_path))
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    temp_path.unlink(missing_ok=True)
                    continue
                temp_path.replace(target_path)
                if self._has_pdf_signature(target_path):
                    return True, None
                target_path.unlink(missing_ok=True)
            except Exception:
                continue

        for combo in ("Control+Shift+S", "Control+S"):
            try:
                page.bring_to_front()
            except Exception:
                pass
            try:
                page.keyboard.press(combo)
            except Exception:
                continue
            downloaded_path, error = self._wait_for_new_downloaded_pdf(before_files, target_path, timeout_seconds=20)
            if downloaded_path is not None:
                return True, None
        return False, "PDF viewer download controls did not produce a file"

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
