from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_COLLECTION_SOURCES: tuple[str, ...] = (
    "pubmed_papers.jsonl",
    "arxiv_papers.jsonl",
    "medrxiv_papers.jsonl",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rebuild_papers_jsonl(
    raw_dir: Path,
    *,
    papers_filename: str = "papers.jsonl",
    source_filenames: Iterable[str] = DEFAULT_COLLECTION_SOURCES,
) -> int:
    papers_path = raw_dir / papers_filename
    total = 0
    with papers_path.open("w", encoding="utf-8") as out:
        for source_file in source_filenames:
            path = raw_dir / source_file
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    # Normalize each line while preserving valid JSONL.
                    json.loads(line)
                    out.write(line + "\n")
                    total += 1
    return total
