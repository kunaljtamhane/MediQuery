import sys
import os
# Ensure we can find your agents
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

import pandas as pd
import evaluate
from supervisor import SupervisorAgent
import json

class AblationSupervisor(SupervisorAgent):
    """A wrapper that allows us to toggle retrieval nodes on/off."""
    def __init__(self, mode="full"):
        super().__init__()
        self.mode = mode

    def execute(self, query):
        # We override the execute method to force empty evidence lists
        # based on the mode provided to the wrapper
        initial_state = {
            "query": query, 
            "rag_evidence": [], 
            "pubmed_evidence": [], 
            "web_evidence": [], 
            "synthesis_prompt": "",
            "final_answer": ""
        }
        
        # Manually run the retrieval nodes conditionally
        if self.mode in ["partial", "full"]:
            initial_state["rag_evidence"] = self.rag_worker.retrieve(query)
            
        if self.mode == "full":
            initial_state["pubmed_evidence"] = self.pubmed_worker.retrieve_literature(query, top_k=1)
            initial_state["web_evidence"] = self.web_worker.search_definitions(query, top_k=1)
            
        # Synthesize and Generate
        final_state = self.graph.invoke(initial_state)
        return final_state

def run_ablation():
    print("--- RUNNING ABLATION STUDY (ISOLATED FILE) ---")
    
    # Load Metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    test_data = pd.read_csv("data/processed/test.csv").head(5).to_dict(orient="records")
    modes = ["baseline", "partial", "full"]
    results = []

    for mode in modes:
        print(f"\nTesting mode: {mode}")
        supervisor = AblationSupervisor(mode=mode)
        preds, refs = [], []
        
        for row in test_data:
            res = supervisor.execute(row["query"])
            preds.append(res["final_answer"])
            refs.append(row["answer"])
            
        # Score
        b = bleu.compute(predictions=preds, references=refs, max_order=1)
        r = rouge.compute(predictions=preds, references=refs)
        ber = bertscore.compute(predictions=preds, references=refs, lang="en")
        
        results.append({
            "Mode": mode,
            "BLEU-1": round(b["bleu"], 4),
            "ROUGE-1": round(r["rouge1"], 4),
            "BERTScore": round(sum(ber["f1"])/len(ber["f1"]), 4)
        })

    # Use this block to print the results without requiring the 'tabulate' library
    print("\n=== FINAL ABLATION RESULTS ===")
    for row in results:
        print(row)

if __name__ == "__main__":
    run_ablation()