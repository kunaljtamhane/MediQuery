#!/usr/bin/env python3
"""
Audit downloaded PDFs and optionally remove files that are not real PDFs.

Usage:
    python data/collection/audit_pdfs.py
    python data/collection/audit_pdfs.py --delete-invalid
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def has_pdf_signature(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 5:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Audit the corpus PDF directory.")
    parser.add_argument(
        "--pdf_dir",
        type=Path,
        default=project_root / "data" / "raw" / "pdfs",
        help="Directory containing downloaded PDF files.",
    )
    parser.add_argument("--delete-invalid", action="store_true", help="Delete files that do not have a PDF signature.")
    args = parser.parse_args()

    counts = Counter()
    invalid: list[Path] = []

    for path in sorted(args.pdf_dir.glob("*.pdf")):
        source = "other"
        lower_name = path.name.lower()
        if lower_name.startswith("arxiv_"):
            source = "arxiv"
        elif lower_name.startswith("pubmed_"):
            source = "pubmed"
        elif lower_name.startswith("medrxiv_"):
            source = "medrxiv"

        counts[f"{source}_total"] += 1
        if has_pdf_signature(path):
            counts[f"{source}_valid"] += 1
        else:
            counts[f"{source}_invalid"] += 1
            invalid.append(path)

    print("PDF audit summary")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")

    if invalid:
        print("\nInvalid files")
        for path in invalid[:25]:
            print(f"  {path.name}")
        if len(invalid) > 25:
            print(f"  ... and {len(invalid) - 25} more")

    if args.delete_invalid and invalid:
        for path in invalid:
            path.unlink(missing_ok=True)
        print(f"\nDeleted {len(invalid)} invalid files")


if __name__ == "__main__":
    main()
