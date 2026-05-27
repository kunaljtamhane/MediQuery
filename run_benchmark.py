import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

import pandas as pd
import evaluate  # This must say evaluate, not run_benchmark!
from supervisor import SupervisorAgent
import json
import time

def run_evaluation():
    print("--- MEDIQUERY AUTOMATED EVALUATION PIPELINE ---")
    
    # 1. Load the Evaluation Metrics
    print("Loading scoring models (this may take a moment)...")
    bleu = evaluate.load("bleu")             # Restore evaluate here
    rouge = evaluate.load("rouge")           # Restore evaluate here
    meteor = evaluate.load("meteor")         # Restore evaluate here
    bertscore = evaluate.load("bertscore")   # Restore evaluate here
    
    # 2. Load the Test Dataset
    # Assuming your test.csv has 'question' and 'ground_truth_answer' columns
    print("Loading test.csv...")
    try:
        test_df = pd.read_csv("data/test.csv")
        # For the first run, let's just test the first 5 rows to ensure it works
        test_data = test_df.head(5).to_dict(orient="records") 
    except FileNotFoundError:
        print("Error: data/test.csv not found. Using a mock dataset for testing.")
        test_data = [
            {
                "question": "What is the sensitivity of rapid prescreening for detecting glandular cell abnormalities?",
                "ground_truth_answer": "The sensitivity of rapid prescreening for detecting glandular cell abnormalities is 0.92."
            }
        ]

    # 3. Initialize the Supervisor Agent
    supervisor = SupervisorAgent()
    
    predictions = []
    references = []
    
    # 4. Run the Pipeline
    print(f"\nEvaluating {len(test_data)} samples...\n")
    for i, row in enumerate(test_data):
        print(f"Processing Sample {i+1}/{len(test_data)}")
        query = row["question"]
        ground_truth = row["ground_truth_answer"]
        
        # Execute the LangGraph workflow
        start_time = time.time()
        result = supervisor.execute(query)
        end_time = time.time()
        
        generated_answer = result["final_answer"]
        
        predictions.append(generated_answer)
        references.append(ground_truth)
        
        print(f"Generation Time: {round(end_time - start_time, 2)}s")
        print("-" * 50)

    # 5. Calculate Final Scores
    print("\nCalculating Final Metrics...")
    
    # BLEU-1 (max_order=1 restricts it to unigram matches for lexical accuracy)
    bleu_results = bleu.compute(predictions=predictions, references=references, max_order=1)
    
    # ROUGE
    rouge_results = rouge.compute(predictions=predictions, references=references)
    
    # METEOR
    meteor_results = meteor.compute(predictions=predictions, references=references)
    
    # BERTScore (Uses a pre-trained model to check semantic similarity)
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    avg_bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])

    # 6. Save and Display Results
    final_report = {
        "BLEU-1": round(bleu_results["bleu"], 4),
        "ROUGE-1": round(rouge_results["rouge1"], 4),
        "METEOR": round(meteor_results["meteor"], 4),
        "BERTScore_F1": round(avg_bert_f1, 4)
    }

    print("\n=== FINAL EVALUATION REPORT ===")
    print(json.dumps(final_report, indent=4))
    
    with open("evaluation_results.json", "w") as f:
        json.dump(final_report, f, indent=4)
    print("\nResults saved to evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()