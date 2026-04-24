import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from download_corpus_pdfs import CorpusPDFDownloader, DownloadItem, normalize_pmc_article_url
from env_loader import configure_requests_session


class DownloadCorpusPDFTests(unittest.TestCase):
    def _scratch_dir(self) -> Path:
        scratch_dir = Path(__file__).resolve().parents[2] / "test_data" / f"download_corpus_pdfs_{uuid4().hex}"
        scratch_dir.mkdir(parents=True, exist_ok=False)
        return scratch_dir

    def test_configure_requests_session_disables_env_proxy_by_default(self):
        session = configure_requests_session(requests.Session())
        self.assertFalse(session.trust_env)

    def test_normalize_pmc_article_url_handles_direct_asset_links(self):
        self.assertEqual(
            normalize_pmc_article_url("https://pmc.ncbi.nlm.nih.gov/articles/PMC13003866/pdf/ZGHA_19_2611693.pdf"),
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC13003866/",
        )

    def test_pubmed_prefers_pmc_open_access_pdf_url_when_available(self):
        temp_dir = self._scratch_dir()
        try:
            downloader = CorpusPDFDownloader(
                raw_dir=temp_dir,
                output_dir=temp_dir,
                arxiv_count=0,
                pubmed_count=0,
                medrxiv_count=0,
            )
            item = DownloadItem(
                source="pubmed",
                source_id="123",
                title="Example",
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/",
                published_date="2026-01-01",
            )
            with patch.object(downloader, "_get_pmc_oa_pdf_url", return_value="https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/main.pdf"):
                request_url, fallback_pdf_url = downloader._resolve_pubmed_request_url(item)

            self.assertEqual(request_url, "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/main.pdf")
            self.assertEqual(fallback_pdf_url, item.pdf_url)
            downloader.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_medrxiv_download_tries_http_before_browser_fallbacks(self):
        temp_dir = self._scratch_dir()
        try:
            downloader = CorpusPDFDownloader(
                raw_dir=temp_dir,
                output_dir=temp_dir,
                arxiv_count=0,
                pubmed_count=0,
                medrxiv_count=0,
            )
            item = DownloadItem(
                source="medrxiv",
                source_id="10.1101/2026.01.01.123456",
                title="Example",
                pdf_url="https://www.medrxiv.org/content/10.1101/2026.01.01.123456v1.full.pdf",
                published_date="2026-01-01",
            )
            target_path = downloader._target_path(item)

            with patch.object(
                downloader,
                "_download_one_via_http",
                return_value=(True, target_path, None),
            ) as http_download, patch.object(
                downloader,
                "_download_one_medrxiv_playwright",
                return_value=(False, target_path, "should not run"),
            ) as playwright_download, patch.object(
                downloader,
                "_download_one_medrxiv_selenium",
                return_value=(False, target_path, "should not run"),
            ) as selenium_download:
                ok, resolved_path, error = downloader._download_one_medrxiv(item, target_path)

            self.assertTrue(ok)
            self.assertEqual(resolved_path, target_path)
            self.assertIsNone(error)
            http_download.assert_called_once_with(item, target_path)
            playwright_download.assert_not_called()
            selenium_download.assert_not_called()
            downloader.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_extract_locator_target_url_uses_href_and_unescapes_entities(self):
        temp_dir = self._scratch_dir()
        try:
            downloader = CorpusPDFDownloader(
                raw_dir=temp_dir,
                output_dir=temp_dir,
                arxiv_count=0,
                pubmed_count=0,
                medrxiv_count=0,
            )

            class FakePage:
                url = "https://journals.tubitak.gov.tr/medical/vol56/iss1/28/"

            class FakeLocator:
                def get_attribute(self, name, timeout=None):
                    values = {
                        "href": "https://journals.tubitak.gov.tr/cgi/viewcontent.cgi?article=6157&amp;context=medical",
                    }
                    return values.get(name)

            resolved = downloader._extract_locator_target_url(FakePage(), FakeLocator())
            self.assertEqual(
                resolved,
                "https://journals.tubitak.gov.tr/cgi/viewcontent.cgi?article=6157&context=medical",
            )
            downloader.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_is_forbidden_page_detects_tubitak_403_html(self):
        temp_dir = self._scratch_dir()
        try:
            downloader = CorpusPDFDownloader(
                raw_dir=temp_dir,
                output_dir=temp_dir,
                arxiv_count=0,
                pubmed_count=0,
                medrxiv_count=0,
            )
            html = "<html><head><title>403 Forbidden</title></head><body><h1>403 Forbidden</h1></body></html>"
            self.assertTrue(
                downloader._is_forbidden_page(
                    "https://journals.tubitak.gov.tr/cgi/viewcontent.cgi?article=6157&context=medical",
                    html,
                    "403 Forbidden",
                )
            )
            downloader.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_select_items_can_be_restricted_to_specific_source_ids(self):
        temp_dir = self._scratch_dir()
        try:
            raw_dir = temp_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "pubmed_papers.jsonl").write_text(
                "\n".join(
                    [
                        '{"source_id":"111","title":"One","pdf_url":"https://example.com/111.pdf","published_date":"2026-01-01"}',
                        '{"source_id":"222","title":"Two","pdf_url":"https://example.com/222.pdf","published_date":"2026-01-02"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            downloader = CorpusPDFDownloader(
                raw_dir=raw_dir,
                output_dir=temp_dir / "pdfs",
                arxiv_count=0,
                pubmed_count=0,
                medrxiv_count=0,
                only_source_ids={"222"},
            )

            items = downloader._select_items("pubmed_papers.jsonl", "pubmed", 10)
            self.assertEqual([item.source_id for item in items], ["222"])
            downloader.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
