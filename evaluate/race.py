"""
RACE — Large-scale Reading Comprehension Dataset From Examinations.

4-way multiple-choice reading comprehension: given a passage and a question,
pick the correct answer from four options (A/B/C/D).  The dataset is sourced
from Chinese middle and high school English exams (Lai et al., 2017).

Evaluated in the original GPT paper (Radford et al., 2018) as the QA /
commonsense reasoning task, and adopted here to measure reading comprehension
of our diffusion-pretrained BERT.

Input format:
    [CLS] question [SEP] passage + " " + option_i [SEP]   ×  4
Output shape: (batch, 4, seq_len)  →  BertForMultipleChoice
Metric: Accuracy

Reference:
  Lai et al., "RACE: Large-scale ReAding Comprehension Dataset From
  Examinations", EMNLP 2017.

Usage:
    python -m evaluate.race --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.race --checkpoint ./ckpt --split high --output_dir ./results
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
from evaluate.metrics             import compute_accuracy

logger = logging.getLogger(__name__)

# Maps the letter answer to a 0-indexed integer label
_ANSWER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class RACEDataset(Dataset):
    """
    Tokenises RACE examples into 4 sequences of the form:
        [CLS] question [SEP] passage + option [SEP]

    Output tensors have shape (4, max_seq_length) for input_ids,
    attention_mask, and token_type_ids; label is a scalar long (0–3).
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 512,
    ):
        self.examples: List[Dict[str, torch.Tensor]] = []

        for ex in hf_dataset:
            question = ex["question"]
            article  = ex["article"]
            options  = ex["options"]           # list of 4 strings (A, B, C, D)
            label    = _ANSWER_TO_IDX.get(ex["answer"], 0)

            # Encode: question as first sequence, passage+option as second
            encs = tokenizer(
                [question] * 4,
                [article + " " + opt for opt in options],
                max_length=max_seq_length,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )

            self.examples.append({
                "input_ids":      encs["input_ids"],          # (4, S)
                "attention_mask": encs["attention_mask"],      # (4, S)
                "token_type_ids": encs.get(
                    "token_type_ids",
                    torch.zeros_like(encs["input_ids"]),
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
# Main entry point
# ------------------------------------------------------------------ #

def run(
    checkpoint_dir: str,
    output_dir:     str,
    split:          str   = "all",    # "all" | "middle" | "high"
    max_seq_length: int   = 512,
    batch_size:     int   = 8,
    num_epochs:     int   = 3,
    learning_rate:  float = 2e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, float]:
    """Fine-tune and evaluate on RACE."""
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

    logger.info(f"Loading RACE dataset (split='{split}') …")
    raw = load_dataset("race", split)

    train_ds = RACEDataset(raw["train"],      tokenizer, max_seq_length)
    eval_ds  = RACEDataset(raw["validation"], tokenizer, max_seq_length)
    logger.info(f"  train={len(train_ds):,}  val={len(eval_ds):,}")

    task_output = os.path.join(output_dir, f"race_{split}")
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

    print(f"\n=== RACE ({split}) Results ===")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")
    return final


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RACE reading comprehension evaluation")
    p.add_argument("--checkpoint",     required=True,
                   help="Path to checkpoint.pt or its parent directory")
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--split",          default="all",
                   choices=["all", "middle", "high"],
                   help="RACE sub-split: all (default), middle, or high")
    p.add_argument("--max_seq_length", type=int,   default=512)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_amp",         action="store_true",
                   help="Disable automatic mixed precision")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
