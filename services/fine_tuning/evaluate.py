"""
Fine-Tuning Evaluation — Stage 4 (MediQuery Proposal)
Evaluates the LoRA fine-tuned LLaMA-3 against a zero-shot frontier LLM baseline
on the full PubMedQA pqa_labeled set (1,000 records, all decisions).

Decision scoring (applied to all samples):
  yes   = +1  (model should affirm the research finding)
  no    = -1  (model should contradict the research finding)
  maybe =  0  (model should express uncertainty)

The model is asked to predict a yes/no/maybe decision alongside its long answer.
Decision accuracy and weighted decision score are reported alongside text metrics.

Metrics computed:
  - ROUGE-L            : text overlap with ground truth long answer
  - Token F1           : SQuAD-style token-level F1
  - Exact Match (EM)   : normalized string equality
  - NDCG@5             : ranking quality of top-5 beam candidates
  - MRR                : reciprocal rank of first matching beam candidate
  - Decision Accuracy  : predicted yes/no/maybe == ground truth decision
  - Decision Score     : mean of (+1 / 0 / -1) for correct / maybe / wrong decisions
  - LLM-as-a-Judge     : GPT-4 scores accuracy, hallucination, grounding (1-5)

Usage:
    # Evaluate fine-tuned model vs. GPT-4 baseline (all 1,000 samples)
    python evaluate.py \\
        --adapter_path ./lora_adapter \\
        --base_model meta-llama/Meta-Llama-3-8B \\
        --num_samples 1000 \\
        --output_path ./eval_results.json

    # Baseline only (no local model needed)
    python evaluate.py --baseline_only --num_samples 1000
"""
import argparse
import json
import logging
import os
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from openai import OpenAI
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
BASELINE_MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")

# Decision scoring scheme: yes=+1, maybe=0, no=-1
DECISION_SCORES = {"yes": 1, "maybe": 0, "no": -1}


# ── Text normalisation (SQuAD-style) ──────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize(prediction) == _normalize(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, ground_truth: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ground_truth, prediction)["rougeL"].fmeasure


# ── Decision extraction ───────────────────────────────────────────────────────

def extract_decision(text: str) -> str | None:
    """
    Parse a yes/no/maybe decision from generated text.
    Looks for an explicit DECISION: tag first, then scans the last sentence,
    then falls back to keyword presence.
    Returns 'yes', 'no', 'maybe', or None if unparseable.
    """
    text_lower = text.lower().strip()

    # Explicit tag: "DECISION: yes"
    tag_match = re.search(r"decision\s*[:=]\s*(yes|no|maybe)", text_lower)
    if tag_match:
        return tag_match.group(1)

    # Last sentence keyword scan
    last_sentence = text_lower.split(".")[-2] if "." in text_lower else text_lower
    for keyword in ("yes", "no", "maybe"):
        if re.search(rf"\b{keyword}\b", last_sentence):
            return keyword

    # Whole-text fallback — take first keyword found
    for keyword in ("yes", "no", "maybe"):
        if re.search(rf"\b{keyword}\b", text_lower):
            return keyword

    return None


def decision_score(predicted: str | None, ground_truth: str) -> int:
    """
    Score a predicted decision against ground truth.
      Correct prediction → DECISION_SCORES[ground_truth]  (+1, 0, or -1)
      Wrong prediction   → -DECISION_SCORES[ground_truth] (inverted)
      Unparseable        → 0
    """
    if predicted is None:
        return 0
    gt_score = DECISION_SCORES.get(ground_truth, 0)
    if predicted == ground_truth:
        return gt_score          # correct: +1, 0, or -1
    return -gt_score             # wrong: inverted


# ── Ranking metrics over beam candidates ──────────────────────────────────────

def _candidate_relevance(candidate: str, ground_truth: str) -> float:
    return token_f1(candidate, ground_truth)


def ndcg_at_k(candidates: list[str], ground_truth: str, k: int = 5) -> float:
    relevances = [_candidate_relevance(c, ground_truth) for c in candidates[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(candidates: list[str], ground_truth: str, threshold: float = 0.3) -> float:
    for i, candidate in enumerate(candidates):
        if _candidate_relevance(candidate, ground_truth) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


# ── Model loading ─────────────────────────────────────────────────────────────

def load_finetuned_model(adapter_path: str, base_model: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    log.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    log.info("Fine-tuned model loaded.")
    return model, tokenizer


def generate_candidates(
    model, tokenizer, query: str, num_candidates: int = 5, max_new_tokens: int = 300
) -> list[str]:
    """
    Generate top-N candidates via beam search.
    Prompt instructs the model to end with a DECISION: yes/no/maybe tag
    so decision extraction is reliable.
    """
    prompt = (
        "<|system|>You are a medical research assistant. "
        "Answer the question based on clinical evidence. "
        "At the end of your answer write exactly: DECISION: yes, DECISION: no, or DECISION: maybe.\n"
        f"<|user|>{query}\n<|assistant|>"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_candidates,
            num_return_sequences=num_candidates,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]


# ── Zero-shot baseline ────────────────────────────────────────────────────────

def baseline_answer(query: str, client: OpenAI) -> str:
    """Zero-shot GPT-4 — no RAG, no fine-tuning, no context."""
    resp = client.chat.completions.create(
        model=BASELINE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a general-purpose medical assistant. "
                    "Answer the medical question as accurately as possible. "
                    "End your answer with exactly: DECISION: yes, DECISION: no, or DECISION: maybe."
                ),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=300,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ── LLM-as-a-Judge ────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a medical QA system answer.

Question: {question}
Ground Truth Decision: {ground_truth_decision}
Ground Truth Answer: {ground_truth}

Source Passages (from PubMedQA):
{source_passages}

System Answer: {answer}

Score the system answer and reply in valid JSON only, no extra text.

{{
  "accuracy": <1-5>,         // 5 = fully correct vs ground truth, 1 = completely wrong
  "hallucination": <0 or 1>, // 1 = contains claims NOT supported by any source passage
  "grounding": <1-5>,        // 5 = every claim traceable to a source passage, 1 = pure speculation
  "predicted_decision": "<yes|no|maybe>"  // decision implied by the system answer
}}"""


def llm_judge(
    question: str,
    ground_truth: str,
    ground_truth_decision: str,
    answer: str,
    client: OpenAI,
    context: dict = None,
) -> dict:
    source_passages = "Not provided."
    if context and context.get("contexts"):
        passages = context["contexts"]
        source_passages = "\n".join(
            f"[{i+1}] {p[:300]}..." for i, p in enumerate(passages[:5])
        )

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    ground_truth_decision=ground_truth_decision,
                    ground_truth=ground_truth,
                    source_passages=source_passages,
                    answer=answer,
                ),
            }],
            max_tokens=120,
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as e:
        log.warning(f"LLM-as-a-Judge failed: {e}")
        return {"accuracy": 0, "hallucination": -1, "grounding": 0, "predicted_decision": None}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_pubmedqa(num_samples: int, local_path: str = None) -> list[dict]:
    """
    Load all PubMedQA pqa_labeled records (yes + no + maybe).
    Prefers local data/pqa_labeled.jsonl; falls back to HuggingFace.

    Distribution: 552 yes / 338 no / 110 maybe (1,000 total).
    Decision scores applied during evaluation: yes=+1, no=-1, maybe=0.

    Source: https://huggingface.co/datasets/qiaojin/PubMedQA
    """
    default_local = Path(__file__).resolve().parents[2] / "data" / "pqa_labeled.jsonl"
    path = Path(local_path) if local_path else default_local

    if path.exists():
        log.info(f"Loading PubMedQA from local file: {path}")
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line.strip()))
    else:
        log.info("Local pqa_labeled.jsonl not found — downloading from HuggingFace...")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        records = list(ds)

    samples = [
        {
            "question": row["question"],
            "long_answer": row["long_answer"],
            "final_decision": row["final_decision"],
            "decision_score": DECISION_SCORES.get(row["final_decision"], 0),
            "context": row.get("context", {}),
        }
        for row in records
    ]

    counts = {d: sum(1 for s in samples if s["final_decision"] == d) for d in ("yes", "no", "maybe")}
    log.info(f"PubMedQA loaded: {len(samples)} total | yes={counts['yes']} no={counts['no']} maybe={counts['maybe']}")
    log.info(f"Using first {num_samples} samples.")
    return samples[:num_samples]


# ── Main evaluation loop ──────────────────────────────────────────────────────

def _empty_metrics() -> dict:
    return {
        "rouge_l": [], "f1": [], "em": [], "ndcg5": [], "mrr": [],
        "decision_correct": [], "decision_score": [],
        "judge_accuracy": [], "judge_hallucination": [], "judge_grounding": [],
    }


def evaluate(
    adapter_path: str,
    base_model: str,
    num_samples: int,
    output_path: str,
    local_data_path: str = None,
    baseline_only: bool = False,
):
    client = OpenAI(api_key=OPENAI_API_KEY)
    samples = load_pubmedqa(num_samples, local_path=local_data_path)

    ft_model, ft_tokenizer = None, None
    if not baseline_only:
        ft_model, ft_tokenizer = load_finetuned_model(adapter_path, base_model)

    ft_m = _empty_metrics()
    bl_m = _empty_metrics()
    results = []

    for i, sample in enumerate(samples):
        question       = sample["question"]
        ground_truth   = sample["long_answer"]
        gt_decision    = sample["final_decision"]
        gt_score       = sample["decision_score"]
        context        = sample["context"]

        log.info(f"[{i+1}/{len(samples)}] [{gt_decision.upper()}] {question[:75]}...")
        row: dict = {
            "question": question,
            "ground_truth": ground_truth,
            "ground_truth_decision": gt_decision,
            "ground_truth_decision_score": gt_score,
        }

        # ── Fine-tuned model ────────────────────────────────────────────────
        if ft_model is not None:
            candidates = generate_candidates(ft_model, ft_tokenizer, question, num_candidates=5)
            best = candidates[0]
            pred_decision = extract_decision(best)
            ds = decision_score(pred_decision, gt_decision)

            row.update({
                "ft_answer": best,
                "ft_candidates": candidates,
                "ft_predicted_decision": pred_decision,
                "ft_decision_score": ds,
            })

            ft_m["rouge_l"].append(rouge_l(best, ground_truth))
            ft_m["f1"].append(token_f1(best, ground_truth))
            ft_m["em"].append(exact_match(best, ground_truth))
            ft_m["ndcg5"].append(ndcg_at_k(candidates, ground_truth, k=5))
            ft_m["mrr"].append(mean_reciprocal_rank(candidates, ground_truth))
            ft_m["decision_correct"].append(float(pred_decision == gt_decision))
            ft_m["decision_score"].append(ds)

            judge = llm_judge(question, ground_truth, gt_decision, best, client, context)
            row["ft_judge"] = judge
            if judge.get("accuracy", 0) > 0:
                ft_m["judge_accuracy"].append(judge["accuracy"])
                ft_m["judge_hallucination"].append(judge["hallucination"])
                ft_m["judge_grounding"].append(judge["grounding"])

        # ── Zero-shot baseline ───────────────────────────────────────────────
        base_ans = baseline_answer(question, client)
        base_pred_decision = extract_decision(base_ans)
        base_ds = decision_score(base_pred_decision, gt_decision)

        row.update({
            "baseline_answer": base_ans,
            "baseline_predicted_decision": base_pred_decision,
            "baseline_decision_score": base_ds,
        })

        bl_m["rouge_l"].append(rouge_l(base_ans, ground_truth))
        bl_m["f1"].append(token_f1(base_ans, ground_truth))
        bl_m["em"].append(exact_match(base_ans, ground_truth))
        bl_m["decision_correct"].append(float(base_pred_decision == gt_decision))
        bl_m["decision_score"].append(base_ds)

        judge_base = llm_judge(question, ground_truth, gt_decision, base_ans, client, context)
        row["baseline_judge"] = judge_base
        if judge_base.get("accuracy", 0) > 0:
            bl_m["judge_accuracy"].append(judge_base["accuracy"])
            bl_m["judge_hallucination"].append(judge_base["hallucination"])
            bl_m["judge_grounding"].append(judge_base["grounding"])

        results.append(row)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _mean(lst): return round(float(np.mean(lst)), 4) if lst else None

    summary = {
        "num_samples": len(samples),
        "decision_scoring": {"yes": +1, "maybe": 0, "no": -1},
        "fine_tuned_model": {
            "ROUGE-L":              _mean(ft_m["rouge_l"]),
            "F1":                   _mean(ft_m["f1"]),
            "EM":                   _mean(ft_m["em"]),
            "NDCG@5":               _mean(ft_m["ndcg5"]),
            "MRR":                  _mean(ft_m["mrr"]),
            "Decision_Accuracy":    _mean(ft_m["decision_correct"]),
            "Decision_Score_Mean":  _mean(ft_m["decision_score"]),
            "Judge_Accuracy":       _mean(ft_m["judge_accuracy"]),
            "Judge_Hallucination":  _mean(ft_m["judge_hallucination"]),
            "Judge_Grounding":      _mean(ft_m["judge_grounding"]),
        } if not baseline_only else None,
        "baseline_zero_shot": {
            "ROUGE-L":              _mean(bl_m["rouge_l"]),
            "F1":                   _mean(bl_m["f1"]),
            "EM":                   _mean(bl_m["em"]),
            "Decision_Accuracy":    _mean(bl_m["decision_correct"]),
            "Decision_Score_Mean":  _mean(bl_m["decision_score"]),
            "Judge_Accuracy":       _mean(bl_m["judge_accuracy"]),
            "Judge_Hallucination":  _mean(bl_m["judge_hallucination"]),
            "Judge_Grounding":      _mean(bl_m["judge_grounding"]),
        },
        "target_thresholds": {"ROUGE-L": "> 0.45", "F1": "> 0.60"},
    }

    log.info("\n" + "=" * 65)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 65)
    if summary["fine_tuned_model"]:
        log.info("Fine-Tuned LLaMA-3 (LoRA):")
        for k, v in summary["fine_tuned_model"].items():
            log.info(f"  {k:30s}: {v}")
    log.info("\nZero-Shot Baseline:")
    for k, v in summary["baseline_zero_shot"].items():
        log.info(f"  {k:30s}: {v}")
    log.info("=" * 65)

    output_data = {"summary": summary, "per_sample_results": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Results saved to {output_path}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="./lora_adapter")
    parser.add_argument("--base_model", default=os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B"))
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to evaluate (default: all 1,000)")
    parser.add_argument("--data_path", default=None,
                        help="Path to local pqa_labeled.jsonl (default: data/pqa_labeled.jsonl)")
    parser.add_argument("--output_path", default="./eval_results.json")
    parser.add_argument("--baseline_only", action="store_true")
    args = parser.parse_args()

    evaluate(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        num_samples=args.num_samples,
        output_path=args.output_path,
        local_data_path=args.data_path,
        baseline_only=args.baseline_only,
    )
