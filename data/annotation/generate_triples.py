import json
import random
from collections import defaultdict
from pathlib import Path

# Paths
processed_dir = Path("End-to-End-Research-Tool-with-Multi-Agent-System-and-Document-Vectorization/data/annotation/processed")
output_file = Path("End-to-End-Research-Tool-with-Multi-Agent-System-and-Document-Vectorization/data/annotation/triples.jsonl")

# Data structures to group by question
positives = defaultdict(list)
negatives = defaultdict(list)

def load_jsonl(filename, target_dict, is_evidence=False):
    path = processed_dir / filename
    if not path.exists():
        print(f"Warning: {filename} not found.")
        return
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            q = data["question"]
            # For evidence files, the key is "evidence", for others it's "answer"
            text = data.get("evidence") if is_evidence else data.get("answer")
            if text:
                target_dict[q].append(text)

print("Loading datasets...")
# Positives
load_jsonl("expert_answers.jsonl", positives)
load_jsonl("accurate_machine_answers.jsonl", positives)

# Negatives
load_jsonl("inaccurate_machine_answers.jsonl", negatives)
load_jsonl("contradicting_evidence.jsonl", negatives, is_evidence=True)
load_jsonl("neutral_evidence.jsonl", negatives, is_evidence=True)

# Generate Triples
triples = []
all_questions = list(positives.keys())

print("Generating triples...")
for q in all_questions:
    q_pos = list(set(positives[q])) # De-duplicate
    q_neg = list(set(negatives[q])) # De-duplicate
    
    # 1. Hard Negatives (Directly related to the question but wrong/neutral)
    for pos_text in q_pos:
        for neg_text in q_neg:
            triples.append({
                "query": q,
                "positive": pos_text,
                "negative": neg_text
            })
            
    # 2. Easy Negatives (Random answer from a DIFFERENT question)
    # This helps the model maintain broad relevance
    if q_pos:
        for _ in range(2): # Add 2 random negatives per question
            random_q = random.choice(all_questions)
            while random_q == q:
                random_q = random.choice(all_questions)
            
            # Pick a random positive from that other question as our negative
            if positives[random_q]:
                random_neg = random.choice(positives[random_q])
                triples.append({
                    "query": q,
                    "positive": random.choice(q_pos),
                    "negative": random_neg
                })

# Shuffle and save
random.shuffle(triples)

print(f"Saving {len(triples)} triples to {output_file}...")
with open(output_file, "w") as f:
    for t in triples:
        f.write(json.dumps(t) + "\n")

print("Done!")
