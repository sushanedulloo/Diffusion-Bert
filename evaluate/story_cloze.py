"""
Story Cloze Test — commonsense story completion.

2-way multiple-choice: given a 4-sentence story context, select the correct
5th sentence (ending) from two candidates.  Tests commonsense narrative
reasoning evaluated in the original GPT paper (Radford et al., 2018).

Input format:
    [CLS] story_context [SEP] ending_i [SEP]   ×  2
Output shape: (batch, 2, seq_len)  →  BertForMultipleChoice
Metric: Accuracy

The dataset (Winter 2018 version) is available on HuggingFace as
"story_cloze" but requires accepting dataset terms at:
    https://cs.rochester.edu/nlp/rocstories/

Run the script with --data_dir pointing to the downloaded TSV files if the
HuggingFace automatic download fails.

Reference:
  Mostafazadeh et al., "A Corpus and Evaluation Framework for Deeper
  Understanding of Commonsense Stories", NAACL 2016.

Usage:
    python -m evaluate.story_cloze --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.story_cloze --checkpoint ./ckpt --data_dir ./story_cloze_data
"""

import argparse
import csv
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from evaluate.fine_tuning_heads   import BertForMultipleChoice, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics             import compute_accuracy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class StoryClozeDataset(Dataset):
    """
    Tokenises Story Cloze examples into 2 sequences of the form:
        [CLS] story_context [SEP] ending_i [SEP]

    Output tensors have shape (2, max_seq_length); label is 0 or 1
    (0-indexed: answer_right_ending - 1).
    """

    def __init__(
        self,
        examples:       List[Dict],
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 128,
    ):
        self.data: List[Dict[str, torch.Tensor]] = []

        for ex in examples:
            context  = " ".join([ex["s1"], ex["s2"], ex["s3"], ex["s4"]])
            endings  = [ex["ending1"], ex["ending2"]]
            label    = int(ex["label"])  # 0 or 1

            encs = tokenizer(
                [context] * 2,
                endings,
                max_length=max_seq_length,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )

            self.data.append({
                "input_ids":      encs["input_ids"],          # (2, S)
                "attention_mask": encs["attention_mask"],      # (2, S)
                "token_type_ids": encs.get(
                    "token_type_ids",
                    torch.zeros_like(encs["input_ids"]),
                ),
                "labels": torch.tensor(label, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


# ------------------------------------------------------------------ #
# Data loading helpers
# ------------------------------------------------------------------ #

def _load_from_huggingface(split: str = "validation") -> List[Dict]:
    """Try loading from HuggingFace datasets (requires accepted terms)."""
    from datasets import load_dataset
    ds = load_dataset("story_cloze", "2016", split=split, trust_remote_code=True)
    out = []
    for ex in ds:
        out.append({
            "s1":      ex["input_sentence_1"],
            "s2":      ex["input_sentence_2"],
            "s3":      ex["input_sentence_3"],
            "s4":      ex["input_sentence_4"],
            "ending1": ex["sentence_quiz1"],
            "ending2": ex["sentence_quiz2"],
            "label":   int(ex["answer_right_ending"]) - 1,  # 1/2 → 0/1
        })
    return out


def _load_from_tsv(tsv_path: str) -> List[Dict]:
    """
    Load from the official TSV file downloaded from the ROCStories portal.

    Expected columns (Winter 2018):
        InputStoryid, InputSentence1..4, RandomFifthSentenceQuiz1,
        RandomFifthSentenceQuiz2, AnswerRightEnding
    """
    examples = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            examples.append({
                "s1":      row["InputSentence1"],
                "s2":      row["InputSentence2"],
                "s3":      row["InputSentence3"],
                "s4":      row["InputSentence4"],
                "ending1": row["RandomFifthSentenceQuiz1"],
                "ending2": row["RandomFifthSentenceQuiz2"],
                "label":   int(row["AnswerRightEnding"]) - 1,
            })
    return examples


def _load_examples(data_dir: Optional[str], split: str) -> List[Dict]:
    """
    Try HuggingFace first; fall back to local TSV files.
    TSV file is expected at: <data_dir>/cloze_test_val__spring2016-cloze_test_ALL_val.tsv
    (or similar; any .tsv in data_dir will be tried).
    """
    if data_dir is None:
        try:
            logger.info("Loading Story Cloze from HuggingFace …")
            return _load_from_huggingface(split)
        except Exception as e:
            raise RuntimeError(
                f"Could not load Story Cloze from HuggingFace ({e}). "
                "Download the dataset from https://cs.rochester.edu/nlp/rocstories/ "
                "and pass --data_dir pointing to the TSV files."
            )

    # Find first .tsv file in data_dir
    tsv_files = [f for f in os.listdir(data_dir) if f.endswith(".tsv")]
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in {data_dir}")
    tsv_path = os.path.join(data_dir, tsv_files[0])
    logger.info(f"Loading Story Cloze from {tsv_path} …")
    return _load_from_tsv(tsv_path)


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
    data_dir:       Optional[str] = None,
    max_seq_length: int           = 128,
    batch_size:     int           = 32,
    num_epochs:     int           = 3,
    learning_rate:  float         = 2e-5,
    seed:           int           = 42,
    use_amp:        bool          = True,
) -> Dict[str, float]:
    """Fine-tune and evaluate on Story Cloze."""
    logging.basicConfig(level=logging.INFO)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    bert, config = load_bert_from_checkpoint(ckpt_file)
    model = BertForMultipleChoice(config)
    model.bert = bert

    # Story Cloze only has a validation split; we split it 80/20 for train/eval
    all_examples = _load_examples(data_dir, split="validation")
    split_idx    = int(0.8 * len(all_examples))
    train_raw    = all_examples[:split_idx]
    eval_raw     = all_examples[split_idx:]
    logger.info(f"  train={len(train_raw):,}  val={len(eval_raw):,}")

    train_ds = StoryClozeDataset(train_raw, tokenizer, max_seq_length)
    eval_ds  = StoryClozeDataset(eval_raw,  tokenizer, max_seq_length)

    task_output = os.path.join(output_dir, "story_cloze")
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

    print(f"\n=== Story Cloze Results ===")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")
    return final


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Story Cloze commonsense evaluation")
    p.add_argument("--checkpoint",     required=True,
                   help="Path to checkpoint.pt or its parent directory")
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--data_dir",       default=None,
                   help="Directory containing Story Cloze TSV files "
                        "(optional; uses HuggingFace if omitted)")
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
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
