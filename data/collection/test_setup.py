#!/usr/bin/env python3
"""
Quick smoke test for the capstone data collection pipeline.

Target repo path:
    data/collection/test_setup.py
"""

from arxiv_scraper import ArxivScraper
from pdf_extractor import PDFExtractor


def test_collection() -> bool:
    print("=" * 70)
    print("QUICK TEST - DATA COLLECTION")
    print("=" * 70)

    scraper = ArxivScraper(output_dir="./test_data")
    extractor = PDFExtractor(data_dir="./test_data")

    # Try a few fallback categories because arXiv can be flaky
    test_categories = ["cs.AI", "cs.LG", "cs.CL"]

    print("\nTest 1: Collect a few sample papers")
    loaded = []
    for category in test_categories:
        try:
            print(f"\nTrying category: {category}")
            saved_count = scraper.search_papers(
                category=category,
                max_results=3,
                start_year=2022,
                max_attempts=3,
            )
            loaded = scraper.load_papers()
            print(f"Saved papers after {category}: {len(loaded)}")

            if saved_count > 0 and loaded:
                print(f"Success with category: {category}")
                break

        except Exception as exc:
            print(f"Test 1 failed for {category}: {exc}")

    if not loaded:
        print("No papers saved from any test category.")
        return False

    print("\nTest 2: Download and extract up to 2 PDFs")
    try:
        extractor.process_pdfs(num_pdfs=2)
        updated = extractor.load_papers()
        extracted = [p for p in updated if p.get("full_text_extracted")]
        print(f"Papers with full text: {len(extracted)}")
    except Exception as exc:
        print(f"Test 2 failed: {exc}")
        return False

    print("\nTest 3: Show one sample record")
    sample = scraper.load_papers()[0]
    print(f"Title: {sample.get('title', 'N/A')[:80]}")
    print(f"Primary category: {sample.get('primary_category', 'N/A')}")
    print(f"PDF URL: {sample.get('pdf_url', 'N/A')}")
    print(f"Full text extracted: {sample.get('full_text_extracted', False)}")

    if sample.get("abstract"):
        print(f"Abstract preview: {sample['abstract'][:150]}...")

    print("\nAll basic tests completed.")
    return True


def main() -> None:
    print("This creates ./test_data with a small sample.\n")
    choice = input("Continue? (y/n): ").strip().lower()
    if choice != "y":
        print("Cancelled.")
        return

    ok = test_collection()
    if ok:
        print("\nNext:")
        print("1. Review ./test_data")
        print("2. Run python data/collection/arxiv_scraper.py")
        print("3. Run python data/collection/pdf_extractor.py")
        print("4. Run python data/annotation/validate_data.py")
    else:
        print("\nTests failed.")
        print("Possible reasons:")
        print("1. Temporary arXiv API outage or HTTP 503")
        print("2. Network instability")
        print("3. Dependencies not installed correctly")
        print("4. PDF download block or timeout")


if __name__ == "__main__":
    main()