"""
Named Entity Recognition on CoNLL-2003 (English).

The model assigns one of 9 IOB2 labels to each token:
  O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC

Wordpiece alignment: only the first sub-token of each word receives
the true label; all other sub-tokens (and special tokens) are set to
-100 so they are ignored by the loss.

Metric: span-level F1 (seqeval).

Reference: Section 4.3 of Devlin et al. (2019).

Usage:
    python -m evaluate.ner --checkpoint ./ckpt --output_dir ./results
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from evaluate.fine_tuning_heads   import BertForTokenClassification, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics import compute_ner_f1

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Label vocabulary (CoNLL-2003)
# ------------------------------------------------------------------ #

CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
LABEL2ID     = {l: i for i, l in enumerate(CONLL_LABELS)}
ID2LABEL     = {i: l for l, i in LABEL2ID.items()}


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class NERDataset(Dataset):
    """
    Tokenizes CoNLL-2003 word-label sequences into BERT sub-token inputs.

    Only the first sub-token of each word keeps the original label;
    subsequent sub-tokens and special tokens receive label -100.
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
            ner_tags = example["ner_tags"]           # list of ints

            enc = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Align labels to sub-tokens
            word_ids = enc.word_ids(batch_index=0)
            labels   = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:                     # [CLS] / [SEP] / [PAD]
                    labels.append(-100)
                elif wid != prev_wid:               # first sub-token of the word
                    labels.append(ner_tags[wid])
                else:                               # continuation sub-token
                    labels.append(-100)
                prev_wid = wid

            self.encodings.append({
                "input_ids":      enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))[0],
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
    """Convert flat logits / labels back to span-level F1 via seqeval."""
    logits = eval_output["logits"]           # (N, S, num_labels)
    labels = eval_output["labels"]           # (N, S)
    preds  = logits.argmax(dim=-1)           # (N, S)

    true_seq: List[List[str]] = []
    pred_seq: List[List[str]] = []
    for pred_row, label_row in zip(preds.tolist(), labels.tolist()):
        t, p = [], []
        for pi, li in zip(pred_row, label_row):
            if li == -100:
                continue
            t.append(ID2LABEL[li])
            p.append(ID2LABEL.get(pi, "O"))
        true_seq.append(t)
        pred_seq.append(p)

    return compute_ner_f1(true_seq, pred_seq)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def run(
    checkpoint_dir: str,
    output_dir:     str,
    max_seq_length: int   = 128,
    batch_size:     int   = 32,
    num_epochs:     int   = 3,
    learning_rate:  float = 5e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, float]:
    """Fine-tune and evaluate on CoNLL-2003 NER."""
    logging.basicConfig(level=logging.INFO)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    bert, config = load_bert_from_checkpoint(ckpt_file)
    model = BertForTokenClassification(config, num_labels=len(CONLL_LABELS))
    model.bert = bert

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    raw = load_dataset("conll2003", trust_remote_code=True)

    train_ds = NERDataset(raw["train"],      tokenizer, max_seq_length)
    eval_ds  = NERDataset(raw["validation"], tokenizer, max_seq_length)
    test_ds  = NERDataset(raw["test"],       tokenizer, max_seq_length)

    task_output = os.path.join(output_dir, "ner_conll2003")
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

    trainer = FineTuningTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    # Final test evaluation
    import torch.utils.data as tud
    test_loader = tud.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)
    trainer.eval_loader = test_loader
    test_metrics = trainer.evaluate()

    print(f"\n=== CoNLL-2003 NER Results (test) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    return test_metrics


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoNLL-2003 NER fine-tuning")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--output_dir",     default="./results")
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
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
