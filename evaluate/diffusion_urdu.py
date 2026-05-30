"""
Diffusion-style evaluation for Urdu downstream tasks.

Mirrors `evaluate/diffusion_glue.py` but:
  * Uses a multilingual / Urdu tokenizer (default: jhu-clsp/mmBERT-base).
  * Loads Urdu HuggingFace datasets:
      urdu_sentiment → ai4bharat/IndicSentiment (Urdu split)
      urdu_nli       → facebook/xnli (config "ur")
  * Uses the URDU_TEMPLATES from instruction_templates.py.

The MDLM mechanics (condition mask, finetuner loop, sampler) are reused
unchanged — they are language-agnostic.

Usage:
    python evaluate.py urdu_sentiment --checkpoint ./ur_mdlm_ckpt --output_dir ./results
    python evaluate.py urdu_nli       --checkpoint ./ur_mdlm_ckpt --output_dir ./results
"""

import json
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from evaluate.diffusion_utils       import (
    load_diffusion_model_from_checkpoint,
    resolve_checkpoint,
)
from evaluate.instruction_templates import (
    URDU_TEMPLATES,
    build_instruction_sequence,
    build_instruction_labels,
)
from evaluate.diffusion_finetuner   import DiffusionFineTuner, FineTuneArgs
from evaluate.metrics               import compute_accuracy
from training.noise_schedule        import LinearAlphaScheduler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Per-task dataset loading
# ------------------------------------------------------------------ #
# Each task spec returns (train_split, eval_split, normalize_fn) where
# `normalize_fn(example) -> dict` rewrites the HF row into the schema
# expected by the URDU_TEMPLATES instruction_fn (must contain a "label"
# key for the gold integer label).

def _load_urdu_sentiment_splits():
    """
    AI4Bharat IndicSentiment — Urdu split.

    The dataset has columns "INDIC REVIEW" (text) and "LABEL" with values
    "Positive" / "Negative".  We normalise to {text, label} where
    label = 1 (positive) or 0 (negative).
    """
    from datasets import load_dataset

    # IndicSentiment configs are language tags; Urdu = "translation-ur"
    # (Indic translations of an English seed set, ~1k examples per lang).
    ds = load_dataset("ai4bharat/IndicSentiment", "translation-ur")

    def _normalize(ex):
        text  = ex.get("INDIC REVIEW") or ex.get("text") or ""
        lbl_s = (ex.get("LABEL") or ex.get("label") or "").strip().lower()
        label = 1 if lbl_s.startswith("pos") else 0
        return {"text": text, "label": label}

    # IndicSentiment provides only "validation" and "test" splits in the
    # public release — use validation for fine-tuning and test for eval.
    train_split = ds["validation"].map(_normalize)
    eval_split  = ds["test"].map(_normalize)
    return train_split, eval_split


def _load_urdu_nli_splits():
    """
    XNLI Urdu split (facebook/xnli, config 'ur').

    XNLI is already in the right schema: {premise, hypothesis, label}
    with label ∈ {0=entailment, 1=neutral, 2=contradiction}.
    """
    from datasets import load_dataset

    ds = load_dataset("facebook/xnli", "ur")
    return ds["train"], ds["validation"]


_TASK_LOADERS = {
    "urdu_sentiment": _load_urdu_sentiment_splits,
    "urdu_nli":       _load_urdu_nli_splits,
}


# ------------------------------------------------------------------ #
# Instruction dataset
# ------------------------------------------------------------------ #

class UrduInstructionDataset(Dataset):
    """
    Wrap a HuggingFace split in instruction-prompted MDLM examples.

    Yields dicts with: input_ids, condition_mask, attention_mask, labels,
    raw_label — same schema as GlueInstructionDataset.
    """

    def __init__(
        self,
        hf_split,
        task_name:      str,
        tokenizer,
        max_seq_length: int  = 128,
        is_train:       bool = True,
    ):
        self.examples: List[Dict[str, torch.Tensor]] = []

        template    = URDU_TEMPLATES[task_name]
        label_words = template["label_words"]
        answer_len  = template["answer_len"]

        for ex in hf_split:
            input_ids, condition_mask, attention_mask = build_instruction_sequence(
                instruction_fn=template["instruction_fn"],
                answer_prefix=template["answer_prefix"],
                answer_len=answer_len,
                example=ex,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )

            raw_label  = int(ex["label"])
            label_word = label_words.get(raw_label, label_words[0])

            true_answer_ids = tokenizer.encode(
                label_word, add_special_tokens=False
            )[:answer_len]
            while len(true_answer_ids) < answer_len:
                true_answer_ids.append(tokenizer.pad_token_id)

            if is_train:
                labels = build_instruction_labels(
                    input_ids, condition_mask, true_answer_ids, tokenizer
                )
            else:
                labels = torch.full((max_seq_length,), -100, dtype=torch.long)

            self.examples.append({
                "input_ids":      input_ids,
                "condition_mask": condition_mask,
                "attention_mask": attention_mask,
                "labels":         labels,
                "raw_label":      torch.tensor(float(raw_label), dtype=torch.float),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------ #
# Token-length sanity check
# ------------------------------------------------------------------ #

def _verify_label_token_lengths(tokenizer, task_name: str) -> None:
    """
    Warn if a label word tokenises into more pieces than answer_len under
    the current tokenizer.  Mismatch is the #1 source of zero accuracy in
    instruction-style classification.
    """
    template   = URDU_TEMPLATES[task_name]
    answer_len = template["answer_len"]
    for lid, word in template["label_words"].items():
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) > answer_len:
            logger.warning(
                f"  Label {lid} ('{word}') tokenises to {len(ids)} pieces "
                f"(answer_len={answer_len}) — will be truncated. "
                f"Consider bumping answer_len or picking a shorter synonym."
            )
        else:
            logger.info(f"  Label {lid} ('{word}') → {ids}  (len={len(ids)})")


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def run(
    task_name:      str,
    checkpoint_dir: str,
    output_dir:     str,
    tokenizer_name: str   = "jhu-clsp/mmBERT-base",
    max_seq_length: int   = 128,
    batch_size:     int   = 32,
    num_epochs:     int   = 3,
    learning_rate:  float = 2e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
    num_steps:      int   = 200,
    temperature:    float = 1.0,
    max_train:      Optional[int] = None,
    max_eval:       Optional[int] = None,
) -> Dict[str, float]:
    """
    Diffusion-style evaluation for one Urdu task.

    Args:
        task_name:      "urdu_sentiment" or "urdu_nli".
        checkpoint_dir: MDLM checkpoint file or its parent dir.
        output_dir:     where to write the results JSON.
        tokenizer_name: HF tokenizer id (default: jhu-clsp/mmBERT-base).
        max_train:      cap on train examples (None = all). Useful on Kaggle.
        max_eval:       cap on eval examples (None = all).
    """
    if task_name not in URDU_TEMPLATES:
        raise ValueError(
            f"Unknown Urdu task '{task_name}'. "
            f"Available: {list(URDU_TEMPLATES)}"
        )

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = resolve_checkpoint(checkpoint_dir)

    logger.info(f"[Urdu-Diffusion] task={task_name}  device={device}")
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info("Verifying label-word token lengths …")
    _verify_label_token_lengths(tokenizer, task_name)

    logger.info(f"Loading diffusion model from {ckpt_file} …")
    model     = load_diffusion_model_from_checkpoint(ckpt_file)
    scheduler = LinearAlphaScheduler()

    logger.info("Loading HuggingFace splits …")
    train_hf, eval_hf = _TASK_LOADERS[task_name]()

    if max_train is not None:
        train_hf = train_hf.select(range(min(max_train, len(train_hf))))
    if max_eval is not None:
        eval_hf = eval_hf.select(range(min(max_eval, len(eval_hf))))

    logger.info(
        f"Building instruction datasets "
        f"(train={len(train_hf)}, eval={len(eval_hf)}) …"
    )
    train_ds = UrduInstructionDataset(
        train_hf, task_name, tokenizer, max_seq_length, is_train=True,
    )
    eval_ds  = UrduInstructionDataset(
        eval_hf, task_name, tokenizer, max_seq_length, is_train=False,
    )

    args = FineTuneArgs(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_amp=use_amp,
    )

    finetuner = DiffusionFineTuner(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        scheduler=scheduler,
        args=args,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info(
        f"Fine-tuning on {len(train_ds)} examples for {num_epochs} epoch(s) …"
    )
    finetuner.train()

    logger.info("Running conditional diffusion eval …")
    metrics = finetuner.evaluate(
        label_words=URDU_TEMPLATES[task_name]["label_words"],
        metric_fn=compute_accuracy,
        batch_size=batch_size,
        num_steps=num_steps,
        temperature=temperature,
        is_regression=False,
    )

    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{task_name}_diffusion_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Results written to {out_path}")

    return metrics
