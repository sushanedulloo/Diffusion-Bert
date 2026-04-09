"""
Cross-lingual Dependency Parsing on Universal Dependencies treebanks.

Protocol (zero-shot cross-lingual transfer):
  1. Fine-tune on English UD (en_ewt) using a biaffine dependency parser
     head on top of BERT (Dozat & Manning 2017 architecture).
  2. Evaluate zero-shot on other language UD treebanks.

Metrics:
  UAS — Unlabeled Attachment Score (% correct head assignments)
  LAS — Labeled Attachment Score   (% correct head + label assignments)

The parser uses:
  - Separate MLP projections to arc-head / arc-dependent spaces
  - Biaffine attention for arc scoring  (BertForDependencyParsing)
  - Separate biaffine for label scoring
  - Greedy decoding (argmax per dependent token)

Reference: Wu & Dredze (2019); Dozat & Manning (2017).

Usage:
    python -m evaluate.dependency_parsing --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.dependency_parsing --checkpoint ./ckpt --languages de fr zh
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from evaluate.fine_tuning_heads   import BertForDependencyParsing, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics import compute_dep_parsing_metrics
from evaluate.pos_tagging import UD_CONFIGS, DEFAULT_LANGUAGES as UD_DEFAULT_LANGS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Universal Dependency relation labels (UD v2 core set, 37 labels)
# ------------------------------------------------------------------ #

UD_DEPREL_LABELS = [
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc",
    "ccomp", "clf", "compound", "conj", "cop", "csubj", "dep", "det",
    "discourse", "dislocated", "expl", "fixed", "flat", "goeswith",
    "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl",
    "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp",
]
NUM_DEPREL = len(UD_DEPREL_LABELS)
DEPREL2ID  = {l: i for i, l in enumerate(UD_DEPREL_LABELS)}
ID2DEPREL  = {i: l for l, i in DEPREL2ID.items()}


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class UDDepDataset(Dataset):
    """
    Tokenizes UD sentences for dependency parsing.

    Each word is aligned to its first sub-token; others get label -100.
    Head indices are adjusted to token indices (word offset → first sub-token offset).

    Special tokens ([CLS], [SEP], [PAD]) receive head=-100, label=-100.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 128,
    ):
        self.encodings: List[Dict[str, torch.Tensor]] = []

        for example in hf_dataset:
            words   = example["tokens"]
            heads   = example["head"]      # list of ints (0 = root, 1-indexed to words)
            deprels = example["deprel"]    # list of dep label strings

            enc = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            word_ids = enc.word_ids(batch_index=0)

            # Build word → first-subtoken mapping
            word_to_tok: Dict[int, int] = {}
            for tok_i, wid in enumerate(word_ids):
                if wid is not None and wid not in word_to_tok:
                    word_to_tok[wid] = tok_i

            # Build aligned head and label tensors
            head_labels:  List[int] = []
            deprel_labels: List[int] = []
            prev_wid = None

            for tok_i, wid in enumerate(word_ids):
                if wid is None:                            # [CLS] / [SEP] / [PAD]
                    head_labels.append(-100)
                    deprel_labels.append(-100)
                elif wid != prev_wid:                      # first sub-token of word
                    gold_head = heads[wid]                 # 0 = root
                    if gold_head == 0:
                        # Root attaches to [CLS] (position 0)
                        head_tok = 0
                    else:
                        head_tok = word_to_tok.get(gold_head - 1, 0)  # convert 1-indexed → 0-indexed word
                    head_labels.append(head_tok)

                    deprel_str = deprels[wid].lower().split(":")[0]    # strip subtypes
                    deprel_labels.append(DEPREL2ID.get(deprel_str, DEPREL2ID.get("dep", 14)))
                else:                                      # continuation sub-token
                    head_labels.append(-100)
                    deprel_labels.append(-100)
                prev_wid = wid

            self.encodings.append({
                "input_ids":      enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "token_type_ids": enc.get(
                    "token_type_ids", torch.zeros_like(enc["input_ids"])
                )[0],
                "head_ids":   torch.tensor(head_labels,   dtype=torch.long),
                "label_ids":  torch.tensor(deprel_labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encodings[idx]


# ------------------------------------------------------------------ #
# Metric helper
# ------------------------------------------------------------------ #

def _compute_dep_metrics(
    pred_heads_list:  List[List[int]],
    pred_labels_list: List[List[int]],
    gold_heads_list:  List[List[int]],
    gold_labels_list: List[List[int]],
    masks_list:       List[List[bool]],
) -> Dict[str, float]:
    return compute_dep_parsing_metrics(
        pred_heads_list,
        pred_labels_list,
        gold_heads_list,
        gold_labels_list,
        masks_list,
    )


def _evaluate_dep(model, loader, device) -> Dict[str, float]:
    model.eval()
    pred_heads_all, pred_labels_all = [], []
    gold_heads_all, gold_labels_all = [], []
    masks_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)

            ph = out["pred_heads"].cpu().tolist()
            pl = out["pred_labels"].cpu().tolist()
            gh = batch["head_ids"].cpu().tolist()
            gl = batch["label_ids"].cpu().tolist()

            for i in range(len(ph)):
                mask = [g != -100 for g in gh[i]]
                pred_heads_all.append(ph[i])
                pred_labels_all.append(pl[i])
                gold_heads_all.append([max(g, 0) for g in gh[i]])
                gold_labels_all.append([max(g, 0) for g in gl[i]])
                masks_all.append(mask)

    return _compute_dep_metrics(
        pred_heads_all, pred_labels_all,
        gold_heads_all, gold_labels_all,
        masks_all,
    )


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
    learning_rate:  float = 2e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, Dict[str, float]]:
    """
    Fine-tune biaffine parser on English UD, then evaluate zero-shot on other languages.

    Returns {lang: {"uas": float, "las": float}} per language.
    """
    if languages is None:
        languages = UD_DEFAULT_LANGS
    logging.basicConfig(level=logging.INFO)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    bert, config = load_bert_from_checkpoint(ckpt_file)
    model = BertForDependencyParsing(config, num_labels=NUM_DEPREL)
    model.bert = bert

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    # ── Step 1: Fine-tune on English UD (en_ewt) ───────────────────
    logger.info("Loading English UD (en_ewt) for fine-tuning …")
    en_ud    = load_dataset("universal_dependencies", "en_ewt", trust_remote_code=True)
    train_ds = UDDepDataset(en_ud["train"],      tokenizer, max_seq_length)
    eval_ds  = UDDepDataset(en_ud["validation"], tokenizer, max_seq_length)

    task_output = os.path.join(output_dir, "dependency_parsing")

    # Custom training loop — the dependency model has non-standard keys
    # (head_ids, label_ids instead of labels), so we configure TrainingArgs
    # with metric_for_best="loss" and save based on loss.
    args = TrainingArgs(
        output_dir=task_output,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        use_amp=use_amp,
        metric_for_best="loss",
        greater_is_better=False,
    )

    # FineTuningTrainer expects "labels" key for compute_metrics; dependency
    # parsing uses "head_ids"/"label_ids".  We pass compute_metrics=None and
    # run custom evaluation after training.
    trainer = FineTuningTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=None,
        args=args,
        compute_metrics=None,
    )
    trainer.train()

    # Quick validation evaluation
    device    = trainer.device
    raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    val_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0)
    val_metrics = _evaluate_dep(raw_model, val_loader, device)
    logger.info(f"English UD val: UAS={val_metrics['uas']:.2f}  LAS={val_metrics['las']:.2f}")

    # ── Step 2: Zero-shot evaluation on target languages ───────────
    raw_model.eval()
    results: Dict[str, Dict[str, float]] = {}

    for lang in languages:
        ud_cfg = UD_CONFIGS.get(lang)
        if ud_cfg is None:
            logger.warning(f"  No UD config for '{lang}'. Skipping.")
            continue
        logger.info(f"Evaluating zero-shot parsing: {lang} ({ud_cfg}) …")
        try:
            ud_lang  = load_dataset("universal_dependencies", ud_cfg, trust_remote_code=True)
            test_ds  = UDDepDataset(ud_lang["test"], tokenizer, max_seq_length)
        except Exception as e:
            logger.warning(f"  Could not load {ud_cfg}: {e}")
            continue

        loader      = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
        lang_metrics = _evaluate_dep(raw_model, loader, device)
        results[lang] = lang_metrics
        logger.info(f"  {lang}: UAS={lang_metrics['uas']:.2f}  LAS={lang_metrics['las']:.2f}")

    print(f"\n=== Cross-Lingual Dependency Parsing (UD) Results ===")
    print(f"  {'Language':<8}  {'UAS':>6}  {'LAS':>6}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}")
    for lang, m in sorted(results.items()):
        print(f"  {lang:<8}  {m['uas']:>6.2f}  {m['las']:>6.2f}")
    if results:
        avg_uas = sum(m["uas"] for m in results.values()) / len(results)
        avg_las = sum(m["las"] for m in results.values()) / len(results)
        print(f"  {'Average':<8}  {avg_uas:>6.2f}  {avg_las:>6.2f}")

    return results


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-lingual dependency parsing (Universal Dependencies)")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--languages",      nargs="+", default=None,
                   help=f"Target languages. Default: all UD configs ({' '.join(UD_DEFAULT_LANGS)})")
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
        languages=args.languages,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
