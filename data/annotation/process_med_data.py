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
    "joint_accurate": open(output_dir / "joint_accurate_answers.jsonl", "w"),
}

print("Processing medaesqa_v1.json and generating joint accurate file...")

try:
    with open(source_file, "r") as f:
        data = json.load(f)

    for entry in data:
        question = entry.get("question", "")

        # 1. Expert Answers
        if entry.get("expert_curated_answer"):
            cleaned_expert = remove_citations(entry["expert_curated_answer"])
            expert_obj = {
                "question": question,
                "answer": cleaned_expert,
                "source": "expert"
            }
            files["expert"].write(json.dumps(expert_obj) + "\n")
            files["joint_accurate"].write(json.dumps(expert_obj) + "\n")

        # 2 & 3. Machine Answers (Accurate vs Inaccurate)
        machine_answers = entry.get("machine_generated_answers", {})
        for m_id, m_data in machine_answers.items():
            out_key = None
            is_accurate = False
            if m_data.get("is_answer_accurate") == "yes":
                out_key = "accurate_machine"
                is_accurate = True
            elif m_data.get("is_answer_accurate") == "no":
                out_key = "inaccurate_machine"
            
            if out_key:
                cleaned_answer = remove_citations(m_data.get("answer", ""))
                machine_obj = {
                    "question": question,
                    "machine_id": m_id,
                    "answer": cleaned_answer,
                    "source": "machine"
                }
                files[out_key].write(json.dumps(machine_obj) + "\n")
                
                if is_accurate:
                    files["joint_accurate"].write(json.dumps(machine_obj) + "\n")

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

    print(f"Done! All files (including joint_accurate_answers.jsonl) created in: {output_dir}")

finally:
    for f in files.values():
        f.close()
