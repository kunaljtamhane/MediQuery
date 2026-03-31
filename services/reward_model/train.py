"""
Person A — Reward Model Training (Weeks 5-6)
Fine-tunes ms-marco-MiniLM cross-encoder on labeled (query, passage, label) triples.

Usage:
    python train.py --data_path ../../data/annotation/triples.jsonl --output_dir ./model
"""
import argparse
import json
import logging
from pathlib import Path

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import Dataset
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TriplesDataset(Dataset):
    """
    Expects a JSONL file where each line is:
    {"query": "...", "positive": "...", "negative": "..."}
    """
    def __init__(self, path: str):
        self.samples = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                # Positive pair (label=1) and negative pair (label=0)
                self.samples.append({"texts": [row["query"], row["positive"]], "label": 1})
                self.samples.append({"texts": [row["query"], row["negative"]], "label": 0})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train(data_path: str, output_dir: str, epochs: int = 3, batch_size: int = 16):
    log.info(f"Loading triples from {data_path}")
    dataset = TriplesDataset(data_path)
    log.info(f"Training on {len(dataset)} pairs ({len(dataset)//2} triples)")

    # TODO Week 5: Start with this checkpoint — it was pre-trained on MS MARCO
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

    # Split 80/20 for train/eval
    split = int(len(dataset) * 0.8)
    train_samples = dataset.samples[:split]
    eval_samples = dataset.samples[split:]

    model.fit(
        train_dataloader=train_samples,
        epochs=epochs,
        warmup_steps=10,
        output_path=output_dir,
        show_progress_bar=True,
    )

    log.info(f"Model saved to {output_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../../data/annotation/triples.jsonl")
    parser.add_argument("--output_dir", default="./model")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args.data_path, args.output_dir, args.epochs)
