# Data Annotation & Processing Progress

## Date: 4/8/2026

### 1. Medical Dataset Inspection
- Explored `medaesqa_v1.json`, a rich medical QA dataset containing expert answers, machine-generated answers (M1-M30), and citation-level evidence assessments.
- Identified key fields for reward model training: `is_answer_accurate`, `answer_sentence_relevance`, and `evidence_relation` (supporting, contradicting, neutral, invalid citation).

### 2. Specialized Data Extraction
- Created `process_med_data.py` to partition the raw dataset into 6 high-quality subsets for training and evaluation:
    - `expert_answers.jsonl`: Gold-standard human-curated answers.
    - `accurate_machine_answers.jsonl`: Machine answers manually verified as correct.
    - `inaccurate_machine_answers.jsonl`: Machine answers verified as incorrect (Hard Negatives).
    - `contradicting_evidence.jsonl`: Clinical snippets that directly contradict a claim.
    - `neutral_evidence.jsonl`: Snippets that mention the topic but lack definitive evidence.
    - `joint_accurate_answers.jsonl`: Combined expert and accurate machine answers for a robust "Positive" source.

### 3. Deep Text Cleaning
- Implemented robust regex cleaning to remove:
    - Standard citations (e.g., `[12345]`).
    - PMID references (e.g., `[PMID: 12345]`, `PMID 12345`).
    - "Garbage" brackets left behind after citation removal (e.g., `[]`, `[, ]`, `[ , ]`).
- Ensured clinical terminology remains intact while removing academic metadata.

### 4. Triple Generation Strategy
- Created `generate_triples.py` using a **Multi-Level Negative Strategy**:
    - **Hard Negatives:** Pairing positive answers with contradicting/neutral evidence for the *same* question.
    - **Fact-Check Negatives:** Pairing positive answers with inaccurate machine answers for the *same* question.
    - **Easy Negatives:** Pairing positive answers with accurate answers from *different* questions to maintain broad relevance.
- **Result:** Generated **28,136 triples** in `triples.jsonl`.

### 5. Repository Updates
- Updated `.gitignore` to track specific processed datasets and the final `triples.jsonl`.
- Pushed processing scripts, documentation, and cleaned datasets to the main repository.
