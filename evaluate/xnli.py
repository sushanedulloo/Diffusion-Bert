"""
XNLI — Cross-lingual Natural Language Inference.

Protocol (zero-shot cross-lingual transfer):
  1. Fine-tune on English MultiNLI (mnli training split).
  2. Evaluate on 15 XNLI test languages without any target-language data.

The 15 XNLI languages:
  ar, bg, de, el, en, es, fr, hi, ru, sw, th, tr, ur, vi, zh

Metric: accuracy per language (3-way classification: entailment /
        contradiction / neutral).

Reference: Wu & Dredze (2019) — "Beto, Bentz, Becas:
           The Surprising Cross-Lingual Effectiveness of BERT".

Usage:
    python -m evaluate.xnli --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.xnli --checkpoint ./ckpt --languages en de fr --output_dir ./results
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from evaluate.fine_tuning_heads   import BertForSequenceClassification, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics import compute_accuracy

logger = logging.getLogger(__name__)

XNLI_LANGUAGES = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
XNLI_LABELS    = ["entailment", "neutral", "contradiction"]
LABEL2ID       = {l: i for i, l in enumerate(XNLI_LABELS)}
NUM_LABELS     = 3


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class NLIDataset(Dataset):
    """Tokenizes premise/hypothesis pairs for 3-way NLI classification."""

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 128,
        label_field:    str = "label",
    ):
        premises    = hf_dataset["premise"]
        hypotheses  = hf_dataset["hypothesis"]

        # XNLI hypothesis field may be a dict {'language': ..., 'translation': ...}
        if isinstance(hypotheses[0], dict):
            hypotheses = hypotheses["translation"]

        enc = tokenizer(
            premises,
            hypotheses,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
        self.labels         = torch.tensor(hf_dataset[label_field], dtype=torch.long)

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

def _compute_metrics(eval_output: Dict) -> Dict[str, float]:
    logits = eval_output["logits"]
    labels = eval_output["labels"]
    preds  = logits.argmax(dim=-1).tolist()
    return compute_accuracy(preds, labels.tolist())


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def run(
    checkpoint_dir: str,
    output_dir:     str,
    languages:      Optional[List[str]] = None,
    max_seq_length: int   = 128,
    batch_size:     int   = 32,
    num_epochs:     int   = 2,
    learning_rate:  float = 2e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, float]:
    """
    Fine-tune on English MNLI, then evaluate zero-shot on XNLI languages.

    Returns per-language accuracy dict.
    """
    if languages is None:
        languages = XNLI_LANGUAGES
    logging.basicConfig(level=logging.INFO)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    bert, config = load_bert_from_checkpoint(ckpt_file)
    model = BertForSequenceClassification(config, num_labels=NUM_LABELS)
    model.bert = bert

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    # ── Step 1: Fine-tune on English MNLI ───────────────────────────
    logger.info("Loading English MNLI for fine-tuning …")
    mnli = load_dataset("glue", "mnli")

    train_ds = NLIDataset(mnli["train"],              tokenizer, max_seq_length)
    eval_ds  = NLIDataset(mnli["validation_matched"], tokenizer, max_seq_length)

    task_output = os.path.join(output_dir, "xnli")
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

    # ── Step 2: Zero-shot evaluation on each XNLI language ──────────
    raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    raw_model.eval()

    results: Dict[str, float] = {}
    for lang in languages:
        logger.info(f"Evaluating on XNLI language: {lang} …")
        try:
            xnli_lang = load_dataset("xnli", lang)
            test_ds   = NLIDataset(xnli_lang["test"], tokenizer, max_seq_length)
        except Exception as e:
            logger.warning(f"  Could not load XNLI {lang}: {e}")
            continue

        loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(trainer.device) for k, v in batch.items()}
                out   = raw_model(**batch)
                all_logits.append(out["logits"].cpu())
                all_labels.append(batch["labels"].cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = _compute_metrics({"logits": logits, "labels": labels})
        acc = metrics["accuracy"]
        results[lang] = acc
        logger.info(f"  {lang}: accuracy={acc:.4f}")

    print(f"\n=== XNLI Zero-Shot Results ===")
    for lang, acc in sorted(results.items()):
        print(f"  {lang}: {acc:.4f}")
    if results:
        avg = sum(results.values()) / len(results)
        print(f"  Average: {avg:.4f}")
        results["average"] = avg

    return results


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XNLI cross-lingual NLI evaluation")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--languages",      nargs="+",  default=None,
                   help=f"Languages to evaluate. Default: all 15 ({' '.join(XNLI_LANGUAGES)})")
    p.add_argument("--max_seq_length", type=int,   default=128)
    p.add_argument("--batch_size",     type=int,   default=32)
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
        languages=args.languages,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
