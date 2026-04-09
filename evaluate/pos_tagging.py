"""
Cross-lingual POS tagging on Universal Dependencies treebanks.

Protocol (zero-shot cross-lingual transfer):
  1. Fine-tune on English UD (en_ewt) with 17 UPOS tags.
  2. Evaluate zero-shot on other language UD treebanks.

Metric: UPOS accuracy (ignoring padding / special tokens).

Reference: Wu & Dredze (2019) Table 5.

Universal UPOS tags (17 categories):
  ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART,
  PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X

Dataset: HuggingFace `universal_dependencies` (config per treebank).

Usage:
    python -m evaluate.pos_tagging --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.pos_tagging --checkpoint ./ckpt --languages de fr zh
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from evaluate.fine_tuning_heads   import BertForTokenClassification, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics import compute_pos_accuracy

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# UPOS label set (CoNLL-U universal tags)
# ------------------------------------------------------------------ #

UPOS_LABELS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
]
NUM_UPOS  = len(UPOS_LABELS)          # 17
UPOS2ID   = {t: i for i, t in enumerate(UPOS_LABELS)}
ID2UPOS   = {i: t for t, i in UPOS2ID.items()}

# UD treebank config names on HuggingFace universal_dependencies
UD_CONFIGS: Dict[str, str] = {
    "en":  "en_ewt",
    "de":  "de_gsd",
    "fr":  "fr_gsd",
    "es":  "es_gsd",
    "ru":  "ru_syntagrus",
    "ar":  "ar_padt",
    "zh":  "zh_gsd",
    "ja":  "ja_gsd",
    "ko":  "ko_gsd",
    "hi":  "hi_hdtb",
    "tr":  "tr_imst",
    "nl":  "nl_alpino",
    "pt":  "pt_bosque",
    "it":  "it_isdt",
    "pl":  "pl_pdb",
    "fi":  "fi_tdt",
    "sv":  "sv_talbanken",
}
DEFAULT_LANGUAGES = list(UD_CONFIGS.keys())


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class UDPOSDataset(Dataset):
    """
    Tokenizes UD word sequences with UPOS integer labels.

    universal_dependencies uses integer upos tags aligned to the UPOS_LABELS
    list above.  Only the first sub-token of each word gets the real label;
    others get -100.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 128,
    ):
        self.encodings: List[Dict[str, torch.Tensor]] = []

        for example in hf_dataset:
            words = example["tokens"]
            upos  = example["upos"]       # list of ints (0-indexed UPOS)

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
                    labels.append(upos[wid])
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

def _compute_metrics(eval_output: Dict) -> Dict[str, float]:
    logits = eval_output["logits"]
    labels = eval_output["labels"]
    preds  = logits.argmax(dim=-1).view(-1).tolist()
    labs   = labels.view(-1).tolist()
    return compute_pos_accuracy(preds, labs)


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
    Fine-tune on English UD POS, then evaluate zero-shot on target languages.

    Returns per-language UPOS accuracy dict.
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
    model = BertForTokenClassification(config, num_labels=NUM_UPOS)
    model.bert = bert

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    # ── Step 1: Fine-tune on English UD (en_ewt) ───────────────────
    logger.info("Loading English UD (en_ewt) for fine-tuning …")
    en_ud    = load_dataset("universal_dependencies", "en_ewt", trust_remote_code=True)
    train_ds = UDPOSDataset(en_ud["train"],      tokenizer, max_seq_length)
    eval_ds  = UDPOSDataset(en_ud["validation"], tokenizer, max_seq_length)

    task_output = os.path.join(output_dir, "pos_tagging")
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

    # ── Step 2: Zero-shot evaluation on target languages ───────────
    raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    raw_model.eval()

    results: Dict[str, float] = {}
    for lang in languages:
        ud_cfg = UD_CONFIGS.get(lang)
        if ud_cfg is None:
            logger.warning(f"  No UD config known for language '{lang}'. Skipping.")
            continue
        logger.info(f"Evaluating zero-shot POS on: {lang} ({ud_cfg}) …")
        try:
            ud_lang  = load_dataset("universal_dependencies", ud_cfg, trust_remote_code=True)
            test_ds  = UDPOSDataset(ud_lang["test"], tokenizer, max_seq_length)
        except Exception as e:
            logger.warning(f"  Could not load {ud_cfg}: {e}")
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
        logger.info(f"  {lang}: UPOS accuracy={acc:.4f}")

    print(f"\n=== Cross-Lingual POS Tagging (UD) Results ===")
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
    p = argparse.ArgumentParser(description="Cross-lingual POS tagging (Universal Dependencies)")
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
