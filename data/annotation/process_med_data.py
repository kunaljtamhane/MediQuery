import json
import os
import re
from pathlib import Path

# Updated Regex to find:
# 1. Standard citations: [12345], [123, 456]
# 2. PMID citations: [PMID: 12345], [PMID: 123, 456]
# 3. Bare PMIDs: PMID: 12345, PMID 12345
CITATION_PATTERN = re.compile(r'\[(?:PMID:\s*)?\d+(?:,\s*\d+)*\]|PMID:?\s*\d+', re.IGNORECASE)

def remove_citations(text):
    if not text:
        return ""
    # Remove the patterns (PMID and [123])
    text = CITATION_PATTERN.sub('', text)
    
    # Remove any remaining "garbage" brackets like [], [, ], [ , ], [,,]
    # This matches brackets containing only whitespace, commas, or semicolons
    text = re.sub(r'\[[\s,;]*\]', '', text)
    
    # Clean up multiple spaces and punctuation artifacts
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.?;:])', r'\1', text)
    return text.strip()

# Paths
source_file = "End-to-End-Research-Tool-with-Multi-Agent-System-and-Document-Vectorization/data/annotation/medaesqa_v1.json"
output_dir = Path("End-to-End-Research-Tool-with-Multi-Agent-System-and-Document-Vectorization/data/annotation/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Output File Handlers
files = {
    "expert": open(output_dir / "expert_answers.jsonl", "w"),
    "accurate_machine": open(output_dir / "accurate_machine_answers.jsonl", "w"),
    "inaccurate_machine": open(output_dir / "inaccurate_machine_answers.jsonl", "w"),
    "contradicting_evidence": open(output_dir / "contradicting_evidence.jsonl", "w"),
    "neutral_evidence": open(output_dir / "neutral_evidence.jsonl", "w"),
}

print("Processing medaesqa_v1.json and performing deep cleaning of brackets...")

try:
    with open(source_file, "r") as f:
        data = json.load(f)

    for entry in data:
        question = entry.get("question", "")

        # 1. Expert Answers
        if entry.get("expert_curated_answer"):
            files["expert"].write(json.dumps({
                "question": question,
                "answer": remove_citations(entry["expert_curated_answer"])
            }) + "\n")

        # 2 & 3. Machine Answers (Accurate vs Inaccurate)
        machine_answers = entry.get("machine_generated_answers", {})
        for m_id, m_data in machine_answers.items():
            out_key = None
            if m_data.get("is_answer_accurate") == "yes":
                out_key = "accurate_machine"
            elif m_data.get("is_answer_accurate") == "no":
                out_key = "inaccurate_machine"
            
            if out_key:
                files[out_key].write(json.dumps({
                    "question": question,
                    "machine_id": m_id,
                    "answer": remove_citations(m_data.get("answer", ""))
                }) + "\n")

            # 4 & 5. Evidence Support (Contradicting vs Neutral)
            for sentence in m_data.get("answer_sentences", []):
                citations = sentence.get("citation_assessment") or []
                for cite in citations:
                    rel = cite.get("evidence_relation", "").lower()
                    evidence = cite.get("evidence_support")
                    
                    if not evidence:
                        continue
                        
                    cleaned_evidence = remove_citations(evidence)
                    if rel == "contradicting":
                        files["contradicting_evidence"].write(json.dumps({
                            "question": question,
                            "evidence": cleaned_evidence
                        }) + "\n")
                    elif rel == "neutral":
                        files["neutral_evidence"].write(json.dumps({
                            "question": question,
                            "evidence": cleaned_evidence
                        }) + "\n")

    print(f"Done! Deep-cleaned files created in: {output_dir}")

finally:
    for f in files.values():
        f.close()
