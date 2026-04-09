"""
Cross-lingual Named Entity Recognition.

Protocol (zero-shot cross-lingual transfer):
  1. Fine-tune on English CoNLL-2003 NER.
  2. Evaluate zero-shot on other languages using the WikiANN dataset
     (Pan et al., 2017) which provides PER/ORG/LOC annotations in
     176 languages in BIO format.

Metric: span-level F1 per language (seqeval).

Reference: Wu & Dredze (2019) Table 4.

Default target languages: ar, de, es, fr, hi, ko, nl, pt, ru, tr, zh
(subset of WikiANN with high-quality annotations).

Usage:
    python -m evaluate.cross_lingual_ner --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.cross_lingual_ner --checkpoint ./ckpt --languages de fr zh
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from evaluate.ner import NERDataset, CONLL_LABELS, ID2LABEL
from evaluate.fine_tuning_heads   import BertForTokenClassification, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics import compute_ner_f1

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ["ar", "de", "es", "fr", "hi", "ko", "nl", "pt", "ru", "tr", "zh"]

# WikiANN uses the same 3+O label set but may differ slightly from CoNLL
WIKIANN_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
WIKIANN_LABEL2ID = {l: i for i, l in enumerate(WIKIANN_LABELS)}
WIKIANN_ID2LABEL = {i: l for l, i in WIKIANN_LABEL2ID.items()}


# ------------------------------------------------------------------ #
# WikiANN Dataset
# ------------------------------------------------------------------ #

class WikiANNDataset(Dataset):
    """
    Tokenizes WikiANN (pan_x) word-level NER sequences into BERT input tensors.

    WikiANN provides tokens / ner_tags fields (ints: 0=O, 1=B-PER, 2=I-PER,
    3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC).

    The label mapping is aligned to the same scheme used for CoNLL fine-tuning.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 128,
    ):
        self.encodings: List[Dict[str, torch.Tensor]] = []

        for example in hf_dataset:
            words    = example["tokens"]
            ner_tags = example["ner_tags"]

            enc = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            word_ids = enc.word_ids(batch_index=0)
            labels   = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)
                elif wid != prev_wid:
                    labels.append(ner_tags[wid])
                else:
                    labels.append(-100)
                prev_wid = wid

            self.encodings.append({
                "input_ids":      enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "token_type_ids": enc.get(
                    "token_type_ids", torch.zeros_like(enc["input_ids"])
                )[0],
                "labels":         torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encodings[idx]


# ------------------------------------------------------------------ #
# Metric helper
# ------------------------------------------------------------------ #

def _compute_metrics_wikiann(eval_output: Dict, id2label: Dict[int, str]) -> Dict[str, float]:
    logits = eval_output["logits"]
    labels = eval_output["labels"]
    preds  = logits.argmax(dim=-1)

    true_seq: List[List[str]] = []
    pred_seq: List[List[str]] = []
    for pred_row, label_row in zip(preds.tolist(), labels.tolist()):
        t, p = [], []
        for pi, li in zip(pred_row, label_row):
            if li == -100:
                continue
            t.append(id2label.get(li, "O"))
            p.append(id2label.get(pi, "O"))
        true_seq.append(t)
        pred_seq.append(p)

    return compute_ner_f1(true_seq, pred_seq)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def run(
    checkpoint_dir: str,
    output_dir:     str,
    languages:      Optional[List[str]] = None,
    max_seq_length: int   = 128,
    batch_size:     int   = 32,
    num_epochs:     int   = 3,
    learning_rate:  float = 5e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, float]:
    """
    Fine-tune on English CoNLL-2003 NER, then evaluate zero-shot on target languages.

    Returns per-language F1 dict.
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
    # Use CoNLL label count (9 labels) for fine-tuning
    model = BertForTokenClassification(config, num_labels=len(CONLL_LABELS))
    model.bert = bert

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    # ── Step 1: Fine-tune on English CoNLL-2003 ─────────────────────
    logger.info("Loading English CoNLL-2003 for fine-tuning …")
    conll = load_dataset("conll2003", trust_remote_code=True)
    train_ds = NERDataset(conll["train"],      tokenizer, max_seq_length)
    eval_ds  = NERDataset(conll["validation"], tokenizer, max_seq_length)

    task_output = os.path.join(output_dir, "cross_lingual_ner")
    args = TrainingArgs(
        output_dir=task_output,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        use_amp=use_amp,
        metric_for_best="f1",
        greater_is_better=True,
    )

    from evaluate.ner import _compute_metrics as _conll_metrics
    trainer = FineTuningTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        compute_metrics=_conll_metrics,
    )
    trainer.train()

    # ── Step 2: Zero-shot evaluation on each target language ────────
    raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    raw_model.eval()

    results: Dict[str, float] = {}
    for lang in languages:
        logger.info(f"Evaluating zero-shot NER on: {lang} …")
        try:
            wikiann = load_dataset("wikiann", lang)
            test_ds = WikiANNDataset(wikiann["test"], tokenizer, max_seq_length)
        except Exception as e:
            logger.warning(f"  Could not load WikiANN/{lang}: {e}")
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
        metrics = _compute_metrics_wikiann(
            {"logits": logits, "labels": labels},
            WIKIANN_ID2LABEL,
        )
        f1 = metrics.get("f1", 0.0)
        results[lang] = f1
        logger.info(f"  {lang}: F1={f1:.4f}")

    print(f"\n=== Cross-Lingual NER (WikiANN) Results ===")
    for lang, f1 in sorted(results.items()):
        print(f"  {lang}: {f1:.4f}")
    if results:
        avg = sum(results.values()) / len(results)
        print(f"  Average: {avg:.4f}")
        results["average"] = avg

    return results


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-lingual NER (WikiANN)")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--languages",      nargs="+", default=None,
                   help=f"Target languages. Default: {' '.join(DEFAULT_LANGUAGES)}")
    p.add_argument("--max_seq_length", type=int,   default=128)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr",             type=float, default=5e-5)
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
