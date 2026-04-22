"""
Fine-Tuning Pipeline — Stage 3b (Option A: Supervised Fine-Tuning)
LoRA/PEFT fine-tuning of LLaMA-3 (8B) directly on MedAESQA triples.

The MedAESQA triples (query, positive, negative) serve as the supervised training
signal. The model is trained to generate the positive (accurate, evidence-grounded)
answer given the query. Negatives informed data curation quality; SFT trains only
on positive completions.

LoRA config: r=16, alpha=32, applied to attention weight matrices only.
All base LLaMA-3 weights are frozen — only adapter weights are trained.
Early stopping on held-out 15% MedAESQA validation split.

Fallback model hierarchy (if LLaMA-3 is unavailable due to GPU constraints):
  1. meta-llama/Meta-Llama-3-8B  (primary)
  2. stanford-crfm/BioMedLM       (2.7B, medically pre-trained decoder)
  3. epfl-llm/meditron-7b         (LLaMA-2 variant, medical domain)

Usage:
    python train.py \
        --triples_path ../../data/annotation/triples.jsonl \
        --output_dir ./lora_adapter \
        --base_model meta-llama/Meta-Llama-3-8B \
        --epochs 3
"""
import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FALLBACK_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
    "stanford-crfm/BioMedLM",
    "epfl-llm/meditron-7b",
]

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)


def load_triples(path: str) -> tuple[list[dict], list[dict]]:
    """Load triples and split 85% train / 15% validation."""
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            records.append(row)

    split = int(len(records) * 0.85)
    return records[:split], records[split:]


def format_for_sft(row: dict, tokenizer) -> dict:
    """
    Convert a triple into an instruction-tuning format.
    We train the model to generate the positive (accurate) answer given the query.
    The negative answer is not used during training — it informed data quality filtering.
    """
    prompt = (
        f"<|system|>You are a medical research assistant. "
        f"Answer the question accurately based on clinical evidence.\n"
        f"<|user|>{row['query']}\n"
        f"<|assistant|>{row['positive']}"
    )
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_base_model(model_name: str):
    """Load model with 8-bit quantization if available, else full precision."""
    log.info(f"Loading base model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
        )
    except Exception:
        log.warning("8-bit loading failed — loading in fp16")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    return model


def train(
    triples_path: str,
    output_dir: str,
    base_model: str = "meta-llama/Meta-Llama-3-8B",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
):
    log.info(f"Loading triples from {triples_path}")
    train_records, val_records = load_triples(triples_path)
    log.info(f"Train: {len(train_records)} | Val: {len(val_records)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_model(base_model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    train_dataset = Dataset.from_list(train_records).map(
        lambda x: format_for_sft(x, tokenizer),
        remove_columns=["query", "positive", "negative"],
    )
    val_dataset = Dataset.from_list(val_records).map(
        lambda x: format_for_sft(x, tokenizer),
        remove_columns=["query", "positive", "negative"],
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    log.info("Starting LoRA fine-tuning...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"LoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_path", default="../../data/annotation/triples.jsonl")
    parser.add_argument("--output_dir", default="./lora_adapter")
    parser.add_argument(
        "--base_model",
        default=os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B"),
        help="HuggingFace model ID. Fallbacks: stanford-crfm/BioMedLM, epfl-llm/meditron-7b",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(
        triples_path=args.triples_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
