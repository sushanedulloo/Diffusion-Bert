"""
SWAG — Situations With Adversarial Generations.

Multiple-choice commonsense inference: given a sentence and 4 candidate
continuations, pick the most plausible one.

Input format: [CLS] sentence [SEP] ending [SEP] — encoded separately for
each of the 4 choices, then stacked to shape (batch, 4, seq_len).

Metric: Accuracy.

Reference: Section 4.4 of Devlin et al. (2019).

Usage:
    python -m evaluate.swag --checkpoint ./ckpt --output_dir ./results
"""

import argparse
import logging
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from evaluate.fine_tuning_heads   import BertForMultipleChoice, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics import compute_accuracy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class SWAGDataset(Dataset):
    """
    Tokenizes SWAG examples as 4 [CLS] ctx [SEP] ending [SEP] sequences.

    Output tensors have shape (4, max_seq_length) for the three input
    fields; labels are a single long scalar (0–3).
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 128,
    ):
        self.examples: List[Dict[str, torch.Tensor]] = []

        for ex in hf_dataset:
            ctx      = ex["sent1"] + " " + ex["sent2"].rstrip()
            endings  = [ex["ending0"], ex["ending1"], ex["ending2"], ex["ending3"]]
            label    = int(ex["label"])

            encs = tokenizer(
                [ctx] * 4,
                endings,
                max_length=max_seq_length,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )

            self.examples.append({
                "input_ids":      encs["input_ids"],           # (4, S)
                "attention_mask": encs["attention_mask"],       # (4, S)
                "token_type_ids": encs.get(
                    "token_type_ids",
                    torch.zeros_like(encs["input_ids"])
                ),
                "labels": torch.tensor(label, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------ #
# Metric helper
# ------------------------------------------------------------------ #

def _compute_metrics(eval_output: Dict) -> Dict[str, float]:
    logits = eval_output["logits"]           # (N, 4)
    labels = eval_output["labels"]           # (N,)
    preds  = logits.argmax(dim=-1).tolist()
    labs   = labels.tolist()
    return compute_accuracy(preds, labs)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def run(
    checkpoint_dir: str,
    output_dir:     str,
    max_seq_length: int   = 128,
    batch_size:     int   = 16,
    num_epochs:     int   = 2,
    learning_rate:  float = 2e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, float]:
    """Fine-tune and evaluate on SWAG."""
    logging.basicConfig(level=logging.INFO)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    bert, config = load_bert_from_checkpoint(ckpt_file)
    model = BertForMultipleChoice(config)
    model.bert = bert

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    raw = load_dataset("swag", "regular")

    train_ds = SWAGDataset(raw["train"],      tokenizer, max_seq_length)
    eval_ds  = SWAGDataset(raw["validation"], tokenizer, max_seq_length)

    task_output = os.path.join(output_dir, "swag")
    args = TrainingArgs(
        output_dir=task_output,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        use_amp=use_amp,
        metric_for_best="accuracy",
        greater_is_better=True,
    )

    trainer = FineTuningTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    final = trainer.evaluate()

    print(f"\n=== SWAG Results ===")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")
    return final


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SWAG multiple-choice evaluation")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--max_seq_length", type=int,   default=128)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--epochs",         type=int,   default=2)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_amp",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
