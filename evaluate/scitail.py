"""
SciTail — Science Question Answering as Textual Entailment.

Binary NLI task: given a science premise and a hypothesis, predict whether
the premise entails the hypothesis ("entails") or not ("neutral").
Dataset sourced from science exam questions (Khot et al., 2018).

Evaluated in the original GPT paper (Radford et al., 2018) as one of five
NLI benchmarks; adopted here as a science-domain NLI evaluation.

Input format: [CLS] premise [SEP] hypothesis [SEP]
Metric: Accuracy

Reference:
  Khot et al., "SciTaiL: A Textual Entailment Dataset from Science Question
  Answering", AAAI 2018.

Usage:
    python -m evaluate.scitail --checkpoint ./ckpt --output_dir ./results
"""

import argparse
import logging
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from evaluate.fine_tuning_heads   import BertForSequenceClassification, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics             import compute_accuracy

logger = logging.getLogger(__name__)

# Label mapping: HuggingFace SciTail uses string labels
_LABEL_MAP = {"entails": 0, "neutral": 1}


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class SciTailDataset(Dataset):
    """
    Tokenises SciTail examples as:
        [CLS] premise [SEP] hypothesis [SEP]
    Label: 0 = entails, 1 = neutral.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 128,
    ):
        self.examples: List[Dict[str, torch.Tensor]] = []

        for ex in hf_dataset:
            premise    = ex["premise"]
            hypothesis = ex["hypothesis"]
            label      = _LABEL_MAP.get(ex["label"], 1)

            enc = tokenizer(
                premise,
                hypothesis,
                max_length=max_seq_length,
                truncation="longest_first",
                padding="max_length",
                return_tensors="pt",
            )

            self.examples.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": enc.get(
                    "token_type_ids",
                    torch.zeros(max_seq_length, dtype=torch.long),
                ).squeeze(0) if "token_type_ids" in enc else
                    torch.zeros(max_seq_length, dtype=torch.long),
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
    logits = eval_output["logits"]           # (N, 2)
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
    max_seq_length: int   = 128,
    batch_size:     int   = 32,
    num_epochs:     int   = 3,
    learning_rate:  float = 2e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, float]:
    """Fine-tune and evaluate on SciTail."""
    logging.basicConfig(level=logging.INFO)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    bert, config = load_bert_from_checkpoint(ckpt_file)
    model = BertForSequenceClassification(config, num_labels=2)
    model.bert = bert

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    logger.info("Loading SciTail dataset …")
    # "tsv_format" config provides premise/hypothesis/label columns directly
    raw = load_dataset("scitail", "tsv_format")

    train_ds = SciTailDataset(raw["train"],      tokenizer, max_seq_length)
    eval_ds  = SciTailDataset(raw["validation"], tokenizer, max_seq_length)
    logger.info(f"  train={len(train_ds):,}  val={len(eval_ds):,}")

    task_output = os.path.join(output_dir, "scitail")
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

    print(f"\n=== SciTail Results ===")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")
    return final


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SciTail NLI evaluation")
    p.add_argument("--checkpoint",     required=True,
                   help="Path to checkpoint.pt or its parent directory")
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
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
