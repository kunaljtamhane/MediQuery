#!/usr/bin/env python3
"""
Fetch additional PubMed papers using 25 diverse medical topic queries.

Loads existing IDs from pubmed_papers.jsonl to skip duplicates, appends
new records, then rebuilds papers.jsonl from all three source files.

Usage:
    python data/collection/fetch_more_pubmed.py
    python data/collection/fetch_more_pubmed.py --max_per_query 200 --start_date 2022/01/01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

COLLECTION_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = COLLECTION_DIR.parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

sys.path.insert(0, str(COLLECTION_DIR))

from collection_utils import rebuild_papers_jsonl
from env_loader import load_env_file
from pubmed_scraper import PubMedCollector

# 25 queries spanning major medical specialties and AI-in-medicine topics.
# Broad enough to yield thousands of unique PMIDs when combined.
MEDICAL_QUERIES: List[str] = [
    # Cardiovascular
    '("Heart Failure"[MeSH Terms] OR "Coronary Artery Disease"[MeSH Terms]'
    ' OR "Atrial Fibrillation"[MeSH Terms] OR "Myocardial Infarction"[MeSH Terms])'
    " AND (diagnosis[Title/Abstract] OR treatment[Title/Abstract] OR prognosis[Title/Abstract])",

    # Oncology / cancer
    '(Neoplasms[MeSH Terms] OR cancer[Title/Abstract])'
    " AND (immunotherapy[Title/Abstract] OR chemotherapy[Title/Abstract]"
    ' OR "targeted therapy"[Title/Abstract] OR biomarker[Title/Abstract])',

    # Neurology / neurodegeneration
    '("Alzheimer Disease"[MeSH Terms] OR "Parkinson Disease"[MeSH Terms]'
    " OR stroke[MeSH Terms] OR epilepsy[MeSH Terms])"
    " AND (treatment[Title/Abstract] OR diagnosis[Title/Abstract] OR biomarker[Title/Abstract])",

    # Infectious disease / COVID-19 / sepsis
    '("COVID-19"[MeSH Terms] OR "SARS-CoV-2"[MeSH Terms]'
    " OR Influenza[MeSH Terms] OR Sepsis[MeSH Terms])"
    " AND (clinical[Title/Abstract] OR outcomes[Title/Abstract] OR treatment[Title/Abstract])",

    # Diabetes / endocrinology / obesity
    '("Diabetes Mellitus"[MeSH Terms] OR "insulin resistance"[Title/Abstract]'
    " OR Obesity[MeSH Terms] OR Thyroid[MeSH Terms])"
    " AND (treatment[Title/Abstract] OR management[Title/Abstract] OR complication[Title/Abstract])",

    # Respiratory / pulmonology
    '("Pulmonary Disease, Chronic Obstructive"[MeSH Terms] OR Asthma[MeSH Terms]'
    " OR Pneumonia[MeSH Terms] OR lung[Title/Abstract])"
    " AND (diagnosis[Title/Abstract] OR therapy[Title/Abstract] OR outcome[Title/Abstract])",

    # Psychiatry / mental health
    "(Depression[MeSH Terms] OR \"Anxiety Disorders\"[MeSH Terms]"
    " OR Schizophrenia[MeSH Terms] OR \"mental health\"[Title/Abstract])"
    " AND (treatment[Title/Abstract] OR outcome[Title/Abstract] OR pharmacotherapy[Title/Abstract])",

    # Genomics / precision medicine
    "(Genomics[MeSH Terms] OR \"precision medicine\"[Title/Abstract]"
    " OR pharmacogenomics[Title/Abstract] OR CRISPR[Title/Abstract])"
    " AND (clinical[Title/Abstract] OR therapeutic[Title/Abstract] OR outcome[Title/Abstract])",

    # Medical imaging + AI
    '("Magnetic Resonance Imaging"[MeSH Terms] OR "Tomography, X-Ray Computed"[MeSH Terms]'
    ' OR Ultrasonography[MeSH Terms] OR "medical imaging"[Title/Abstract])'
    ' AND ("deep learning"[Title/Abstract] OR "neural network"[Title/Abstract]'
    ' OR "image segmentation"[Title/Abstract] OR "convolutional"[Title/Abstract])',

    # Surgery / minimally invasive / robotics
    '("Robotic Surgical Procedures"[MeSH Terms] OR "minimally invasive surgery"[Title/Abstract]'
    " OR laparoscopic[Title/Abstract] OR endoscopy[Title/Abstract])"
    " AND (outcome[Title/Abstract] OR complication[Title/Abstract] OR safety[Title/Abstract])",

    # Electronic health records / clinical informatics
    '("Electronic Health Records"[MeSH Terms] OR "clinical informatics"[Title/Abstract]'
    ' OR "health information technology"[Title/Abstract] OR "clinical decision support"[Title/Abstract])'
    " AND (outcome[Title/Abstract] OR prediction[Title/Abstract] OR quality[Title/Abstract])",

    # Drug discovery / AI-assisted pharmacology
    '("Drug Discovery"[MeSH Terms] OR "drug repurposing"[Title/Abstract]'
    ' OR "drug target"[Title/Abstract] OR pharmacology[MeSH Terms])'
    ' AND ("machine learning"[Title/Abstract] OR "artificial intelligence"[Title/Abstract]'
    ' OR "deep learning"[Title/Abstract])',

    # Pediatrics / neonatology
    "(Pediatrics[MeSH Terms] OR child[MeSH Terms] OR neonatal[Title/Abstract]"
    " OR adolescent[Title/Abstract])"
    " AND (disease[Title/Abstract] OR outcome[Title/Abstract] OR management[Title/Abstract])",

    # Chronic kidney disease / nephrology
    '("Renal Insufficiency, Chronic"[MeSH Terms] OR "kidney disease"[Title/Abstract]'
    " OR dialysis[Title/Abstract] OR transplantation[MeSH Terms])"
    " AND (outcome[Title/Abstract] OR treatment[Title/Abstract] OR progression[Title/Abstract])",

    # Gastroenterology / hepatology
    '("Inflammatory Bowel Diseases"[MeSH Terms] OR "Colorectal Neoplasms"[MeSH Terms]'
    ' OR "Liver Cirrhosis"[MeSH Terms] OR "Hepatocellular Carcinoma"[MeSH Terms])'
    " AND (diagnosis[Title/Abstract] OR treatment[Title/Abstract] OR prognosis[Title/Abstract])",

    # Immunology / autoimmune / rheumatology
    '("Autoimmune Diseases"[MeSH Terms] OR "rheumatoid arthritis"[MeSH Terms]'
    " OR Lupus[MeSH Terms] OR immunology[MeSH Terms])"
    " AND (treatment[Title/Abstract] OR biomarker[Title/Abstract] OR pathogenesis[Title/Abstract])",

    # Epidemiology / cohort / RCT / mortality
    "(Epidemiology[MeSH Terms] OR \"cohort study\"[Title/Abstract]"
    ' OR "randomized controlled trial"[Publication Type] OR "meta-analysis"[Publication Type])'
    ' AND ("risk factor"[Title/Abstract] OR mortality[Title/Abstract] OR incidence[Title/Abstract])',

    # Maternal / reproductive / obstetric
    '("Pregnancy Complications"[MeSH Terms] OR Preeclampsia[MeSH Terms]'
    ' OR "maternal mortality"[Title/Abstract] OR "reproductive health"[Title/Abstract])'
    " AND (outcome[Title/Abstract] OR management[Title/Abstract] OR risk[Title/Abstract])",

    # Intensive care / critical care / sepsis prediction
    '("Intensive Care Units"[MeSH Terms] OR "Critical Care"[MeSH Terms]'
    ' OR "mechanical ventilation"[Title/Abstract] OR "hospital mortality"[Title/Abstract])'
    " AND (outcome[Title/Abstract] OR prediction[Title/Abstract] OR management[Title/Abstract])",

    # LLMs / NLP / GPT in clinical settings
    '("large language model"[Title/Abstract] OR "natural language processing"[Title/Abstract]'
    " OR GPT[Title/Abstract] OR BERT[Title/Abstract] OR \"foundation model\"[Title/Abstract])"
    " AND (clinical[Title/Abstract] OR medical[Title/Abstract] OR health[Title/Abstract])",

    # Federated learning / privacy-preserving ML in healthcare
    '("federated learning"[Title/Abstract] OR "differential privacy"[Title/Abstract]'
    ' OR "privacy-preserving"[Title/Abstract])'
    " AND (medical[Title/Abstract] OR health[Title/Abstract] OR clinical[Title/Abstract])",

    # Biomarkers / liquid biopsy / proteomics
    "(Biomarkers[MeSH Terms] OR \"liquid biopsy\"[Title/Abstract]"
    " OR Proteomics[MeSH Terms] OR metabolomics[Title/Abstract])"
    " AND (cancer[Title/Abstract] OR disease[Title/Abstract] OR prognosis[Title/Abstract])",

    # Vaccines / vaccination / immunization
    "(Vaccines[MeSH Terms] OR vaccination[Title/Abstract] OR immunization[Title/Abstract])"
    " AND (efficacy[Title/Abstract] OR safety[Title/Abstract] OR outcome[Title/Abstract]"
    " OR immunogenicity[Title/Abstract])",

    # Digital health / telemedicine / wearables / mHealth
    "(Telemedicine[MeSH Terms] OR \"digital health\"[Title/Abstract]"
    " OR wearable[Title/Abstract] OR mHealth[Title/Abstract])"
    " AND (patient[Title/Abstract] OR outcome[Title/Abstract] OR adherence[Title/Abstract])",

    # Broad AI / machine learning in clinical medicine
    '("Artificial Intelligence"[MeSH Terms] OR "Machine Learning"[MeSH Terms]'
    " OR \"deep learning\"[Title/Abstract] OR \"neural network\"[Title/Abstract])"
    " AND (clinical[Title/Abstract] OR medical[Title/Abstract] OR hospital[Title/Abstract]"
    " OR patient[Title/Abstract])",
]


def load_existing_ids(jsonl_path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not jsonl_path.exists():
        return ids
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                source_id = record.get("source_id") or record.get("pubmed_id")
                if source_id:
                    ids.add(str(source_id))
            except json.JSONDecodeError:
                continue
    return ids

def main() -> None:
    load_env_file(PROJECT_ROOT / ".env")

    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    parser = argparse.ArgumentParser(
        description="Fetch additional PubMed papers with diverse medical topic queries."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DIR / "pubmed_papers.jsonl",
        help="Path to pubmed_papers.jsonl (new records are appended).",
    )
    parser.add_argument(
        "--max_per_query",
        type=int,
        default=200,
        help="Maximum new IDs to collect per query (default 200).",
    )
    parser.add_argument(
        "--start_date",
        default="2022/01/01",
        help="Earliest publication date (YYYY/MM/DD).",
    )
    parser.add_argument(
        "--end_date",
        default=today,
        help="Latest publication date (YYYY/MM/DD).",
    )
    parser.add_argument(
        "--no_rebuild",
        action="store_true",
        help="Skip rebuilding papers.jsonl after fetching.",
    )
    args = parser.parse_args()

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_ids(output_path)
    print(f"Existing PubMed IDs already on disk: {len(existing_ids)}")

    # One reusable collector — we'll swap the query per iteration.
    collector = PubMedCollector(
        output_path=output_path,
        query=MEDICAL_QUERIES[0],
        max_results=args.max_per_query,
        start_date=args.start_date,
        end_date=args.end_date,
        page_size=200,
        fetch_batch_size=100,
    )

    all_new_ids: List[str] = []
    seen_new: Set[str] = set()

    print(f"\nSearching {len(MEDICAL_QUERIES)} medical topic queries …")
    for i, query in enumerate(MEDICAL_QUERIES, 1):
        collector.query = query
        collector.max_results = args.max_per_query
        skip = existing_ids | seen_new
        ids = collector.search_ids(existing_ids=skip)
        new_ids = [pid for pid in ids if pid not in skip]
        all_new_ids.extend(new_ids)
        seen_new.update(new_ids)
        print(
            f"  [{i:2d}/{len(MEDICAL_QUERIES)}] {len(new_ids):4d} new IDs  "
            f"(cumulative new: {len(all_new_ids):,})"
        )

    if not all_new_ids:
        print("\nNo new PubMed IDs found — nothing to fetch.")
        return

    print(f"\nFetching {len(all_new_ids):,} new PubMed records …")
    collector.max_results = len(all_new_ids)
    records = collector.fetch_records(all_new_ids)
    print(f"Valid records with title + abstract: {len(records):,}")

    collector._append_records(records)
    total_on_disk = len(existing_ids) + len(records)
    print(f"Appended {len(records):,} records. pubmed_papers.jsonl now has ~{total_on_disk:,} records.")

    if not args.no_rebuild:
        total = rebuild_papers_jsonl(output_path.parent)
        print(f"Rebuilt papers.jsonl: {total:,} total records.")


if __name__ == "__main__":
    main()
