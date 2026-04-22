"""
Fine-Tuning Evaluation — Stage 4 (MediQuery Proposal)
Evaluates the LoRA fine-tuned LLaMA-3 against a zero-shot frontier LLM baseline
on the PubMedQA filtered evaluation set (final_decision = "yes").

Metrics computed:
  - ROUGE-L          : longest common subsequence overlap with ground truth
  - Token F1         : token-level precision/recall (SQuAD-style)
  - Exact Match (EM) : normalized string equality
  - NDCG@5           : ranking quality of top-5 beam candidates vs. ground truth
  - MRR              : rank of first candidate that matches ground truth
  - LLM-as-a-Judge   : GPT-4 scores each answer for accuracy, hallucination, grounding

Usage:
    # Evaluate fine-tuned model vs. GPT-4 baseline
    python evaluate.py \
        --adapter_path ./lora_adapter \
        --base_model meta-llama/Meta-Llama-3-8B \
        --num_samples 100 \
        --output_path ./eval_results.json

    # Evaluate baseline only (no local model)
    python evaluate.py --baseline_only --num_samples 100
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


# ── Ranking metrics over beam candidates ──────────────────────────────────────

def _candidate_relevance(candidate: str, ground_truth: str) -> float:
    """Score a candidate answer against ground truth using token F1."""
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
    model, tokenizer, query: str, num_candidates: int = 5, max_new_tokens: int = 256
) -> list[str]:
    """Generate top-N candidates via beam search for NDCG@5 / MRR."""
    prompt = (
        "<|system|>You are a medical research assistant. "
        "Answer the question accurately based on clinical evidence.\n"
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


# ── Zero-shot baseline (GPT-4 / Claude) ──────────────────────────────────────

def baseline_answer(query: str, client: OpenAI) -> str:
    """Query the frontier LLM with zero-shot prompting — no RAG, no fine-tuning."""
    resp = client.chat.completions.create(
        model=BASELINE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a general-purpose medical assistant. "
                    "Answer the following medical question as accurately as possible."
                ),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=256,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ── LLM-as-a-Judge ────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a medical QA system answer against a ground-truth reference.

Question: {question}
Ground Truth: {ground_truth}
System Answer: {answer}

Score the system answer on three dimensions. Reply in valid JSON only, no extra text.

{{
  "accuracy": <1-5>,        // 5 = fully correct, 1 = completely wrong
  "hallucination": <0 or 1>,// 1 = contains claims not supported by ground truth
  "grounding": <1-5>        // 5 = every claim traceable to evidence, 1 = pure speculation
}}"""


def llm_judge(question: str, ground_truth: str, answer: str, client: OpenAI) -> dict:
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question,
                        ground_truth=ground_truth,
                        answer=answer,
                    ),
                }
            ],
            max_tokens=100,
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as e:
        log.warning(f"LLM-as-a-Judge failed: {e}")
        return {"accuracy": 0, "hallucination": -1, "grounding": 0}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_pubmedqa(num_samples: int) -> list[dict]:
    """
    Load PubMedQA labeled split filtered to final_decision = 'yes'.
    Returns list of {question, long_answer, final_decision}.
    """
    log.info("Loading PubMedQA labeled split...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)
    filtered = [
        {
            "question": row["question"],
            "long_answer": row["long_answer"],
            "final_decision": row["final_decision"],
        }
        for row in ds
        if row["final_decision"] == "yes"
    ]
    log.info(f"PubMedQA filtered to {len(filtered)} 'yes' samples. Using {num_samples}.")
    return filtered[:num_samples]


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    adapter_path: str,
    base_model: str,
    num_samples: int,
    output_path: str,
    baseline_only: bool = False,
):
    client = OpenAI(api_key=OPENAI_API_KEY)
    samples = load_pubmedqa(num_samples)

    ft_model, ft_tokenizer = (None, None)
    if not baseline_only:
        ft_model, ft_tokenizer = load_finetuned_model(adapter_path, base_model)

    ft_metrics: dict = {
        "rouge_l": [], "f1": [], "em": [], "ndcg5": [], "mrr": [],
        "judge_accuracy": [], "judge_hallucination": [], "judge_grounding": [],
    }
    baseline_metrics: dict = {
        "rouge_l": [], "f1": [], "em": [],
        "judge_accuracy": [], "judge_hallucination": [], "judge_grounding": [],
    }

    results = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        ground_truth = sample["long_answer"]
        log.info(f"[{i+1}/{len(samples)}] Evaluating: {question[:80]}...")

        row: dict = {"question": question, "ground_truth": ground_truth}

        # ── Fine-tuned model ────────────────────────────────────────────────
        if ft_model is not None:
            candidates = generate_candidates(ft_model, ft_tokenizer, question, num_candidates=5)
            best = candidates[0]
            row["ft_answer"] = best
            row["ft_candidates"] = candidates

            ft_metrics["rouge_l"].append(rouge_l(best, ground_truth))
            ft_metrics["f1"].append(token_f1(best, ground_truth))
            ft_metrics["em"].append(exact_match(best, ground_truth))
            ft_metrics["ndcg5"].append(ndcg_at_k(candidates, ground_truth, k=5))
            ft_metrics["mrr"].append(mean_reciprocal_rank(candidates, ground_truth))

            judge = llm_judge(question, ground_truth, best, client)
            row["ft_judge"] = judge
            if judge["accuracy"] > 0:
                ft_metrics["judge_accuracy"].append(judge["accuracy"])
                ft_metrics["judge_hallucination"].append(judge["hallucination"])
                ft_metrics["judge_grounding"].append(judge["grounding"])

        # ── Zero-shot baseline ───────────────────────────────────────────────
        base_ans = baseline_answer(question, client)
        row["baseline_answer"] = base_ans

        baseline_metrics["rouge_l"].append(rouge_l(base_ans, ground_truth))
        baseline_metrics["f1"].append(token_f1(base_ans, ground_truth))
        baseline_metrics["em"].append(exact_match(base_ans, ground_truth))

        judge_base = llm_judge(question, ground_truth, base_ans, client)
        row["baseline_judge"] = judge_base
        if judge_base["accuracy"] > 0:
            baseline_metrics["judge_accuracy"].append(judge_base["accuracy"])
            baseline_metrics["judge_hallucination"].append(judge_base["hallucination"])
            baseline_metrics["judge_grounding"].append(judge_base["grounding"])

        results.append(row)

    # ── Aggregate and report ─────────────────────────────────────────────────
    def _mean(lst): return round(float(np.mean(lst)), 4) if lst else None

    summary = {
        "num_samples": len(samples),
        "fine_tuned_model": {
            "ROUGE-L":            _mean(ft_metrics["rouge_l"]),
            "F1":                 _mean(ft_metrics["f1"]),
            "EM":                 _mean(ft_metrics["em"]),
            "NDCG@5":             _mean(ft_metrics["ndcg5"]),
            "MRR":                _mean(ft_metrics["mrr"]),
            "Judge_Accuracy":     _mean(ft_metrics["judge_accuracy"]),
            "Judge_Hallucination":_mean(ft_metrics["judge_hallucination"]),
            "Judge_Grounding":    _mean(ft_metrics["judge_grounding"]),
        } if not baseline_only else None,
        "baseline_zero_shot": {
            "ROUGE-L":            _mean(baseline_metrics["rouge_l"]),
            "F1":                 _mean(baseline_metrics["f1"]),
            "EM":                 _mean(baseline_metrics["em"]),
            "Judge_Accuracy":     _mean(baseline_metrics["judge_accuracy"]),
            "Judge_Hallucination":_mean(baseline_metrics["judge_hallucination"]),
            "Judge_Grounding":    _mean(baseline_metrics["judge_grounding"]),
        },
        "target_thresholds": {
            "ROUGE-L": "> 0.45",
            "F1": "> 0.60",
        },
    }

    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    if summary["fine_tuned_model"]:
        log.info("Fine-Tuned LLaMA-3 (LoRA):")
        for k, v in summary["fine_tuned_model"].items():
            log.info(f"  {k:30s}: {v}")
    log.info("\nZero-Shot Baseline (GPT-4o-mini):")
    for k, v in summary["baseline_zero_shot"].items():
        log.info(f"  {k:30s}: {v}")
    log.info("=" * 60)

    output = {"summary": summary, "per_sample_results": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Full results saved to {output_path}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="./lora_adapter",
                        help="Path to the LoRA adapter directory produced by train.py")
    parser.add_argument("--base_model", default=os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B"),
                        help="Base model ID (same as used in train.py)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of PubMedQA 'yes' samples to evaluate")
    parser.add_argument("--output_path", default="./eval_results.json",
                        help="Path to write JSON results")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Skip fine-tuned model; evaluate zero-shot baseline only")
    args = parser.parse_args()

    evaluate(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        num_samples=args.num_samples,
        output_path=args.output_path,
        baseline_only=args.baseline_only,
    )
