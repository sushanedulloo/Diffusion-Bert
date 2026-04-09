"""
MLDoc — Multilingual Document Classification.

Task: classify Reuters news documents into 4 topic categories:
  CCAT (Corporate/Industrial), ECAT (Economics),
  GCAT (Government/Social),   MCAT (Markets)

Protocol (zero-shot cross-lingual transfer):
  1. Fine-tune on English MLDoc training split.
  2. Evaluate on 7 other languages without target-language training data.

Languages: de, en, es, fr, it, ja, ru, zh

Metric: accuracy per language.

Reference: Schwenk & Li (2018) — "A corpus for multilingual document
           classification in eight languages"; Wu & Dredze (2019).

Data format (MLDoc TSV):
  Each line: <label>\t<text>   where label ∈ {CCAT, ECAT, GCAT, MCAT}

Data download:
  MLDoc is NOT available on HuggingFace Hub. Download from:
    https://github.com/facebookresearch/MLDoc
  Place the files under --data_dir with structure:
    <data_dir>/
      english.train.1000
      english.test.1000
      german.train.1000
      german.test.1000
      ... (one train + test file per language)

Usage:
    python -m evaluate.mldoc --checkpoint ./ckpt --data_dir ./mldoc --output_dir ./results
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

# ------------------------------------------------------------------ #
# Label and language metadata
# ------------------------------------------------------------------ #

MLDOC_LABELS  = ["CCAT", "ECAT", "GCAT", "MCAT"]
LABEL2ID      = {l: i for i, l in enumerate(MLDOC_LABELS)}
NUM_LABELS    = len(MLDOC_LABELS)

# Map language code → MLDoc file prefix
LANG_TO_FILE_PREFIX: Dict[str, str] = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "es": "spanish",
    "ru": "russian",
    "zh": "chinese",
    "ja": "japanese",
}
DEFAULT_LANGUAGES = list(LANG_TO_FILE_PREFIX.keys())

# Train size variants available in MLDoc (1000, 2000, 5000, 10000)
_DEFAULT_TRAIN_SIZE = 1000


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

def _load_mldoc_file(path: str) -> List[Dict]:
    """Parse a MLDoc TSV file → list of {'label': str, 'text': str}."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            label, text = parts
            if label not in LABEL2ID:
                continue
            examples.append({"label": label, "text": text})
    return examples


class MLDocDataset(Dataset):
    """Tokenizes MLDoc text examples for 4-class classification."""

    def __init__(
        self,
        examples:       List[Dict],
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 128,
    ):
        texts  = [ex["text"]  for ex in examples]
        labels = [LABEL2ID[ex["label"]] for ex in examples]

        enc = tokenizer(
            texts,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

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
    data_dir:       str,
    output_dir:     str,
    languages:      Optional[List[str]] = None,
    max_seq_length: int   = 128,
    batch_size:     int   = 32,
    num_epochs:     int   = 3,
    learning_rate:  float = 2e-5,
    train_size:     int   = _DEFAULT_TRAIN_SIZE,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, float]:
    """
    Fine-tune on English MLDoc training split, then evaluate zero-shot on other languages.

    Args:
        data_dir: directory containing MLDoc TSV files
                  (e.g. english.train.1000, german.test.1000)

    Returns per-language accuracy dict.
    """
    if languages is None:
        languages = DEFAULT_LANGUAGES
    logging.basicConfig(level=logging.INFO)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    bert, config = load_bert_from_checkpoint(ckpt_file)
    model = BertForSequenceClassification(config, num_labels=NUM_LABELS)
    model.bert = bert

    # ── Load English training split ─────────────────────────────────
    en_train_path = os.path.join(data_dir, f"english.train.{train_size}")
    en_test_path  = os.path.join(data_dir, "english.test.1000")

    if not os.path.exists(en_train_path):
        raise FileNotFoundError(
            f"MLDoc English train file not found: {en_train_path}\n"
            "Download MLDoc from https://github.com/facebookresearch/MLDoc "
            "and pass --data_dir pointing to the directory with the TSV files."
        )

    en_train = _load_mldoc_file(en_train_path)
    en_test  = _load_mldoc_file(en_test_path) if os.path.exists(en_test_path) else []

    train_ds = MLDocDataset(en_train, tokenizer, max_seq_length)
    eval_ds  = MLDocDataset(en_test,  tokenizer, max_seq_length) if en_test else None

    task_output = os.path.join(output_dir, "mldoc")
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

    # ── Zero-shot evaluation per language ───────────────────────────
    raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    raw_model.eval()

    results: Dict[str, float] = {}
    for lang in languages:
        prefix    = LANG_TO_FILE_PREFIX.get(lang)
        if prefix is None:
            logger.warning(f"  Unknown language code '{lang}'. Skipping.")
            continue
        test_path = os.path.join(data_dir, f"{prefix}.test.1000")
        if not os.path.exists(test_path):
            logger.warning(f"  File not found: {test_path}. Skipping {lang}.")
            continue

        logger.info(f"Evaluating MLDoc on: {lang} …")
        test_examples = _load_mldoc_file(test_path)
        test_ds       = MLDocDataset(test_examples, tokenizer, max_seq_length)
        loader        = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(trainer.device) for k, v in batch.items()}
                out   = raw_model(**batch)
                all_logits.append(out["logits"].cpu())
                all_labels.append(batch["labels"].cpu())

        logits  = torch.cat(all_logits, dim=0)
        labels  = torch.cat(all_labels, dim=0)
        metrics = _compute_metrics({"logits": logits, "labels": labels})
        acc = metrics["accuracy"]
        results[lang] = acc
        logger.info(f"  {lang}: accuracy={acc:.4f}")

    print(f"\n=== MLDoc Cross-Lingual Results ===")
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
    p = argparse.ArgumentParser(description="MLDoc multilingual document classification")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--data_dir",       required=True,
                   help="Path to directory with MLDoc TSV files")
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--languages",      nargs="+", default=None,
                   help=f"Languages to evaluate. Default: {' '.join(DEFAULT_LANGUAGES)}")
    p.add_argument("--max_seq_length", type=int,   default=128)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--train_size",     type=int,   default=_DEFAULT_TRAIN_SIZE,
                   choices=[1000, 2000, 5000, 10000])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_amp",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint_dir=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        languages=args.languages,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        train_size=args.train_size,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
