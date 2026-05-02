"""
Diffusion-style evaluation for all 9 GLUE tasks.

Replaces discriminative fine-tuning with instruction-prompted MDLM generation:
  - Each example is formatted as an instruction + [MASK] answer slot
  - MDLM fine-tuning trains the model to fill in the answer token
  - Inference runs the conditional reverse diffusion sampler

Tasks and label words:
  CoLA  — "yes" / "no"              (Matthews correlation)
  SST-2 — "positive" / "negative"   (Accuracy)
  MRPC  — "yes" / "no"              (Accuracy + F1)
  STS-B — "0" … "5"                 (Pearson + Spearman)
  QQP   — "yes" / "no"              (Accuracy + F1)
  MNLI  — "yes" / "maybe" / "no"    (Matched + Mismatched Accuracy)
  QNLI  — "yes" / "no"              (Accuracy)
  RTE   — "yes" / "no"              (Accuracy)
  WNLI  — "no" / "yes"              (Accuracy)

Usage:
    python evaluate.py glue --task cola --checkpoint ./ckpt --eval_mode generative
"""

import json
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from config                  import BertConfig
from model.bert              import BertModel
from model.diffusion_heads   import BertForDiffusion
from training.noise_schedule import LinearAlphaScheduler

from evaluate.diffusion_utils import (
    load_diffusion_model_from_checkpoint,
    resolve_checkpoint,
)
from evaluate.instruction_templates import (
    GLUE_TEMPLATES,
    build_instruction_sequence,
    build_instruction_labels,
    match_label,
)
from evaluate.diffusion_finetuner import DiffusionFineTuner, FineTuneArgs
from evaluate.metrics import (
    compute_matthews,
    compute_accuracy,
    compute_accuracy_and_f1,
    compute_pearson_spearman,
)

logger = logging.getLogger(__name__)

# HuggingFace dataset names for each task
_HF_NAMES: Dict[str, str] = {
    "cola": "glue/cola",
    "sst2": "glue/sst2",
    "mrpc": "glue/mrpc",
    "stsb": "glue/stsb",
    "qqp":  "glue/qqp",
    "mnli": "glue/mnli",
    "qnli": "glue/qnli",
    "rte":  "glue/rte",
    "wnli": "glue/wnli",
}

_EVAL_SPLITS: Dict[str, List[str]] = {
    "mnli": ["validation_matched", "validation_mismatched"],
}
_DEFAULT_EVAL_SPLIT = "validation"


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class GlueInstructionDataset(Dataset):
    """
    GLUE examples formatted as instruction-prompted diffusion sequences.

    Each item contains:
        input_ids:      (max_seq_length,) LongTensor
        condition_mask: (max_seq_length,) BoolTensor
        attention_mask: (max_seq_length,) LongTensor
        labels:         (max_seq_length,) LongTensor (for training)
        raw_label:      scalar float tensor (original task label)
    """

    def __init__(
        self,
        hf_split,
        task_name:      str,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int  = 128,
        is_train:       bool = True,
    ):
        self.examples: List[Dict[str, torch.Tensor]] = []

        template    = GLUE_TEMPLATES[task_name]
        label_words = template["label_words"]
        answer_len  = template["answer_len"]
        is_stsb     = (task_name == "stsb")

        for ex in hf_split:
            input_ids, condition_mask, attention_mask = build_instruction_sequence(
                instruction_fn=template["instruction_fn"],
                answer_prefix=template["answer_prefix"],
                answer_len=answer_len,
                example=ex,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )

            # Determine gold label and answer token IDs
            if is_stsb:
                raw_label  = float(ex["label"])
                disc_label = min(5, max(0, round(raw_label)))
                label_word = label_words[disc_label]
            else:
                raw_label  = float(int(ex["label"]))
                label_word = label_words.get(int(ex["label"]), "yes")

            true_answer_ids = tokenizer.encode(label_word, add_special_tokens=False)[:answer_len]
            # Pad to answer_len if the word tokenizes to fewer tokens
            while len(true_answer_ids) < answer_len:
                true_answer_ids.append(tokenizer.pad_token_id)

            if is_train:
                labels = build_instruction_labels(
                    input_ids, condition_mask, true_answer_ids, tokenizer
                )
            else:
                # Dummy labels during eval (not used for loss)
                labels = torch.full((max_seq_length,), -100, dtype=torch.long)

            self.examples.append({
                "input_ids":      input_ids,
                "condition_mask": condition_mask,
                "attention_mask": attention_mask,
                "labels":         labels,
                "raw_label":      torch.tensor(raw_label, dtype=torch.float),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------ #
# Per-task metric selector
# ------------------------------------------------------------------ #

def _get_metric_fn(task_name: str):
    if task_name == "cola":
        return compute_matthews
    if task_name in ("mrpc", "qqp"):
        return compute_accuracy_and_f1
    if task_name == "stsb":
        return compute_pearson_spearman
    return compute_accuracy


# ------------------------------------------------------------------ #
# Main entry point
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
    num_steps:      int   = 200,
    temperature:    float = 1.0,
) -> Dict[str, float]:
    """
    Instruction-prompted diffusion evaluation for a GLUE task.

    Args:
        task_name:      one of the 9 GLUE task names
        checkpoint_dir: MDLM pre-training checkpoint directory or file
        output_dir:     where to write JSON results
        max_seq_length: token budget (128 for GLUE)
        batch_size:     train and eval batch size
        num_epochs:     fine-tuning epochs
        learning_rate:  AdamW learning rate
        seed:           RNG seed
        use_amp:        enable AMP mixed precision
        num_steps:      reverse diffusion steps at inference
        temperature:    sampling temperature

    Returns:
        Dict with metric values (e.g. {"accuracy": 0.82})
    """
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = resolve_checkpoint(checkpoint_dir)

    logger.info(f"[GLUE-Diffusion] task={task_name}  device={device}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    logger.info(f"Loading diffusion model from {ckpt_file} …")
    model     = load_diffusion_model_from_checkpoint(ckpt_file)
    scheduler = LinearAlphaScheduler()

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    # Load HuggingFace splits
    ds_path   = _HF_NAMES[task_name].split("/")
    hf_ds     = hf_load(*ds_path, trust_remote_code=True)
    train_hf  = hf_ds["train"]

    eval_splits = _EVAL_SPLITS.get(task_name, [_DEFAULT_EVAL_SPLIT])

    logger.info(f"Building instruction datasets …")
    train_ds = GlueInstructionDataset(
        train_hf, task_name, tokenizer, max_seq_length, is_train=True
    )

    template    = GLUE_TEMPLATES[task_name]
    label_words = template["label_words"]
    is_stsb     = (task_name == "stsb")
    metric_fn   = _get_metric_fn(task_name)

    args = FineTuneArgs(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_amp=use_amp,
    )

    all_results: Dict[str, float] = {}
    os.makedirs(output_dir, exist_ok=True)

    for split_name in eval_splits:
        logger.info(f"Evaluating on {split_name} …")
        eval_hf = hf_ds[split_name]
        eval_ds = GlueInstructionDataset(
            eval_hf, task_name, tokenizer, max_seq_length, is_train=False
        )

        # Fine-tune (on first split only; reuse model for subsequent MNLI splits)
        if split_name == eval_splits[0]:
            finetuner = DiffusionFineTuner(
                model=model,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                scheduler=scheduler,
                args=args,
                tokenizer=tokenizer,
                device=device,
            )
            logger.info(f"Fine-tuning on {len(train_ds)} examples for {num_epochs} epoch(s) …")
            finetuner.train()
        else:
            finetuner.eval_dataset = eval_ds

        metrics = finetuner.evaluate(
            label_words=label_words,
            metric_fn=metric_fn,
            batch_size=batch_size,
            num_steps=num_steps,
            temperature=temperature,
            is_regression=is_stsb,
        )

        suffix  = f"_{split_name}" if len(eval_splits) > 1 else ""
        for k, v in metrics.items():
            all_results[f"{k}{suffix}"] = v
            logger.info(f"  {k}{suffix}: {v:.4f}")

    out_path = os.path.join(output_dir, f"glue_{task_name}_diffusion_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results written to {out_path}")

    return all_results
