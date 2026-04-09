"""
SNLI — Stanford Natural Language Inference.

3-way NLI task: given a premise and hypothesis drawn from image captions,
predict the relationship as entailment, neutral, or contradiction.

Evaluated in the original GPT paper (Radford et al., 2018) as the largest
NLI benchmark (570K training pairs).  The five NLI benchmarks from GPT-1 are:
  SNLI (this file), MNLI (GLUE), QNLI (GLUE), SciTail, RTE (GLUE).

Input format: [CLS] premise [SEP] hypothesis [SEP]
Metric: Accuracy (3-way classification)

Reference:
  Bowman et al., "A large annotated corpus for learning natural language
  inference", EMNLP 2015.  https://nlp.stanford.edu/projects/snli/

Usage:
    python -m evaluate.snli --checkpoint ./ckpt --output_dir ./results
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from evaluate.fine_tuning_heads   import BertForSequenceClassification, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics             import compute_accuracy

logger = logging.getLogger(__name__)

# Label mapping: SNLI uses string labels
_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class SNLIDataset(Dataset):
    """
    Tokenises SNLI examples as:
        [CLS] premise [SEP] hypothesis [SEP]
    Label: 0 = entailment, 1 = neutral, 2 = contradiction.

    Examples with label "-" (no gold label / annotator disagreement)
    are filtered out automatically.
    """

    def __init__(
        self,
        split:          str,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 128,
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install the 'datasets' package: pip install datasets")

        logger.info(f"Loading SNLI split={split} …")
        ds = load_dataset("snli", split=split, trust_remote_code=True)

        self.examples: List[Dict[str, torch.Tensor]] = []

        for ex in ds:
            label = ex["label"]
            # SNLI uses integer labels: 0=entailment, 1=neutral, 2=contradiction, -1=no label
            if label == -1:
                continue

            enc = tokenizer(
                ex["premise"],
                ex["hypothesis"],
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": enc["token_type_ids"].squeeze(0),
                "labels":         torch.tensor(label, dtype=torch.long),
            })

        logger.info(f"  {len(self.examples):,} examples after filtering.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------ #
# Evaluation loop
# ------------------------------------------------------------------ #

@torch.no_grad()
def _evaluate(
    model:     BertForSequenceClassification,
    dataset:   SNLIDataset,
    device:    torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch["labels"].tolist())

    return compute_accuracy(preds, labels)


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
    """
    Fine-tune and evaluate on SNLI (3-way NLI).

    Args:
        checkpoint_dir: BERT pre-training checkpoint
        output_dir:     directory to write results and fine-tuned model
        max_seq_length: max tokens (premise + hypothesis + specials)
        batch_size:     training batch size
        num_epochs:     fine-tuning epochs
        learning_rate:  peak learning rate (linear warmup + decay)
        seed:           RNG seed
        use_amp:        automatic mixed precision (FP16)

    Returns:
        {"accuracy": float}
    """
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_ds = SNLIDataset("train",      tokenizer, max_seq_length)
    val_ds   = SNLIDataset("validation", tokenizer, max_seq_length)
    test_ds  = SNLIDataset("test",       tokenizer, max_seq_length)

    from config    import BertConfig
    from model.bert import BertModel
    config = BertConfig()
    bert   = BertModel(config)
    model  = BertForSequenceClassification(config, bert, num_labels=3)

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    model = load_bert_from_checkpoint(model, ckpt_file)
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    trainer = FineTuningTrainer(
        model=model,
        train_dataset=train_ds,
        args=TrainingArgs(
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_amp=use_amp,
            seed=seed,
        ),
    )
    trainer.train()

    logger.info("Evaluating on validation set …")
    val_metrics  = _evaluate(model, val_ds,  device, batch_size)
    logger.info("Evaluating on test set …")
    test_metrics = _evaluate(model, test_ds, device, batch_size)

    results = {
        "val_accuracy":  val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
    }

    import json
    out_path = os.path.join(output_dir, "snli_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SNLI Results ===")
    print(f"  Val  Accuracy: {results['val_accuracy']:.4f}")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"Results written to {out_path}")

    return results


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SNLI fine-tuning and evaluation for diffusion BERT"
    )
    p.add_argument("--checkpoint",     required=True,
                   help="Path to BERT checkpoint.pt or its parent directory")
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
