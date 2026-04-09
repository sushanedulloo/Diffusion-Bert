"""
GLUE Benchmark fine-tuning and evaluation.

Covers all 9 GLUE tasks evaluated in Devlin et al. (2019):
  CoLA  — linguistic acceptability    (Matthews correlation)
  SST-2 — sentiment classification    (Accuracy)
  MRPC  — paraphrase detection        (Accuracy + F1)
  STS-B — semantic textual similarity (Pearson + Spearman)
  QQP   — question-pair paraphrase    (Accuracy + F1)
  MNLI  — NLI multi-genre             (Matched + Mismatched Accuracy)
  QNLI  — question NLI                (Accuracy)
  RTE   — textual entailment          (Accuracy)
  WNLI  — Winograd NLI                (Accuracy)

Usage:
    python -m evaluate.glue --task cola --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.glue --task mnli --checkpoint ./ckpt --output_dir ./results
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from config import BertConfig
from evaluate.fine_tuning_heads   import BertForSequenceClassification, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics import (
    compute_matthews,
    compute_accuracy,
    compute_accuracy_and_f1,
    compute_pearson_spearman,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Task metadata
# ------------------------------------------------------------------ #

@dataclass
class _TaskMeta:
    keys:        Tuple[str, Optional[str]]   # (sentence_key_a, sentence_key_b or None)
    num_labels:  int
    metric_name: str                          # key returned by compute_* functions
    greater_is_better: bool


GLUE_TASKS: Dict[str, _TaskMeta] = {
    "cola":  _TaskMeta(("sentence", None),          2, "matthews_corrcoef", True),
    "sst2":  _TaskMeta(("sentence", None),          2, "accuracy",          True),
    "mrpc":  _TaskMeta(("sentence1", "sentence2"),  2, "acc_and_f1",        True),
    "stsb":  _TaskMeta(("sentence1", "sentence2"),  1, "corr",              True),
    "qqp":   _TaskMeta(("question1", "question2"),  2, "acc_and_f1",        True),
    "mnli":  _TaskMeta(("premise", "hypothesis"),   3, "accuracy",          True),
    "qnli":  _TaskMeta(("question", "sentence"),    2, "accuracy",          True),
    "rte":   _TaskMeta(("sentence1", "sentence2"),  2, "accuracy",          True),
    "wnli":  _TaskMeta(("sentence1", "sentence2"),  2, "accuracy",          True),
}

# HuggingFace dataset split names (MNLI has two validation splits)
_EVAL_SPLITS = {
    "mnli":  ["validation_matched", "validation_mismatched"],
}
_DEFAULT_EVAL_SPLIT = "validation"


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class GlueDataset(Dataset):
    """Tokenizes a HuggingFace GLUE split into BERT input tensors."""

    def __init__(
        self,
        hf_dataset,
        tokenizer:    BertTokenizerFast,
        task_meta:    _TaskMeta,
        max_seq_length: int = 128,
    ):
        self.task_meta = task_meta
        key_a, key_b   = task_meta.keys

        texts_a = hf_dataset[key_a]
        texts_b = hf_dataset[key_b] if key_b else None

        enc = tokenizer(
            texts_a,
            texts_b,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))

        # Labels: float for regression (STS-B), long for classification
        if task_meta.num_labels == 1:
            self.labels = torch.tensor(hf_dataset["label"], dtype=torch.float)
        else:
            self.labels = torch.tensor(hf_dataset["label"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels":         self.labels[idx],
        }


# ------------------------------------------------------------------ #
# Metric helper
# ------------------------------------------------------------------ #

def _make_compute_metrics(task_name: str, task_meta: _TaskMeta):
    """Return a compute_metrics callable for the given task."""
    def compute_metrics(eval_output: Dict) -> Dict[str, float]:
        logits = eval_output["logits"]
        labels = eval_output["labels"]
        if labels is None:
            return {}

        if task_meta.num_labels == 1:                    # STS-B regression
            preds = logits.squeeze(-1).tolist()
            labs  = labels.float().tolist()
            return compute_pearson_spearman(preds, labs)
        else:
            preds = logits.argmax(dim=-1).tolist()
            labs  = labels.long().tolist()
            if task_name == "cola":
                return compute_matthews(preds, labs)
            elif task_name in ("mrpc", "qqp"):
                return compute_accuracy_and_f1(preds, labs)
            else:
                return compute_accuracy(preds, labs)

    return compute_metrics


# ------------------------------------------------------------------ #
# Main fine-tuning + evaluation function
# ------------------------------------------------------------------ #

def run(
    task_name:      str,
    checkpoint_dir: str,
    output_dir:     str,
    max_seq_length: int   = 128,
    batch_size:     int   = 32,
    num_epochs:     int   = 3,
    learning_rate:  float = 2e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, float]:
    """
    Fine-tune and evaluate on a single GLUE task.

    Returns the final evaluation metrics dict.
    """
    task_name = task_name.lower().replace("-", "")
    assert task_name in GLUE_TASKS, f"Unknown task '{task_name}'. Choose from {list(GLUE_TASKS)}"
    task_meta = GLUE_TASKS[task_name]

    logging.basicConfig(level=logging.INFO)

    # ── Load tokenizer ──────────────────────────────────────────────
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ── Load pre-trained BERT backbone ──────────────────────────────
    ckpt_file = os.path.join(checkpoint_dir, "checkpoint.pt") if os.path.isdir(checkpoint_dir) else checkpoint_dir
    bert, config = load_bert_from_checkpoint(ckpt_file)

    model = BertForSequenceClassification(config, num_labels=task_meta.num_labels)
    model.bert = bert

    # ── Load datasets ────────────────────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    hf_name = "sst2" if task_name == "sst2" else task_name
    raw = load_dataset("glue", hf_name)

    train_ds = GlueDataset(raw["train"], tokenizer, task_meta, max_seq_length)

    eval_splits = _EVAL_SPLITS.get(task_name, [_DEFAULT_EVAL_SPLIT])
    eval_datasets = {
        split: GlueDataset(raw[split], tokenizer, task_meta, max_seq_length)
        for split in eval_splits
    }
    # Use first eval split as the primary one for early stopping
    primary_eval_split = eval_splits[0]

    # ── Trainer ──────────────────────────────────────────────────────
    task_output_dir = os.path.join(output_dir, f"glue_{task_name}")
    args = TrainingArgs(
        output_dir=task_output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        use_amp=use_amp,
        metric_for_best=task_meta.metric_name,
        greater_is_better=task_meta.greater_is_better,
    )

    trainer = FineTuningTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_datasets[primary_eval_split],
        args=args,
        compute_metrics=_make_compute_metrics(task_name, task_meta),
    )
    trainer.train()

    # ── Final evaluation on all eval splits ──────────────────────────
    results: Dict[str, float] = {}
    for split, ds in eval_datasets.items():
        trainer.eval_loader = torch.utils.data.DataLoader(
            ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
        )
        split_metrics = trainer.evaluate()
        for k, v in split_metrics.items():
            results[f"{split}/{k}"] = v
        logger.info(f"  {split}: {split_metrics}")

    print(f"\n=== GLUE {task_name.upper()} Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GLUE fine-tuning and evaluation")
    p.add_argument("--task",           required=True,  choices=list(GLUE_TASKS))
    p.add_argument("--checkpoint",     required=True,  help="path to checkpoint.pt or checkpoint dir")
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--max_seq_length", type=int,   default=128)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_amp",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        task_name=args.task,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
