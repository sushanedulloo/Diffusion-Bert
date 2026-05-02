"""
Diffusion-style evaluation for SQuAD v1.1 and v2.0.

Replaces span-extraction fine-tuning with instruction-prompted generation:
  - Context + question form the instruction (condition)
  - 30 [MASK] tokens form the answer target
  - MDLM fine-tuning trains the model to generate the answer string
  - Inference runs the conditional reverse diffusion sampler

For SQuAD v2.0, unanswerable questions train the model to generate
"unanswerable"; predicted strings starting with "unanswerable" are
treated as no-answer predictions.

Metric: Exact Match (EM) + Token-level F1 (same formulas as the standard
SQuAD metric, applied to the generated string vs. gold answer strings).

Usage:
    python evaluate.py squad --version 1 --checkpoint ./ckpt --eval_mode generative
    python evaluate.py squad --version 2 --checkpoint ./ckpt --eval_mode generative
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from training.noise_schedule  import LinearAlphaScheduler
from evaluate.diffusion_utils import (
    load_diffusion_model_from_checkpoint,
    resolve_checkpoint,
    mdlm_sample_conditional,
)
from evaluate.instruction_templates import decode_answer
from evaluate.diffusion_finetuner   import DiffusionFineTuner, FineTuneArgs
from evaluate.metrics               import compute_squad_metrics

logger = logging.getLogger(__name__)

_ANSWER_LEN   = 30     # [MASK] tokens reserved for the answer
_MAX_SEQ_LEN  = 384    # wider budget for long contexts


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class SquadInstructionDataset(Dataset):
    """
    SQuAD examples formatted as instruction-prompted diffusion sequences.

    Template:
        [CLS] Context: {context} Question: {question} Answer: [MASK]×30 [SEP]

    The context is truncated to fit within max_seq_length, reserving room
    for the question, answer slots, and structural tokens.

    Each item:
        input_ids:      (max_seq_length,) LongTensor
        condition_mask: (max_seq_length,) BoolTensor
        attention_mask: (max_seq_length,) LongTensor
        labels:         (max_seq_length,) LongTensor  (training only)
        raw_label:      0.0 (placeholder, unused)
        qid:            question id string (stored separately)
        gold_answers:   list of gold answer strings (stored separately)
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int  = _MAX_SEQ_LEN,
        is_train:       bool = True,
        version_2:      bool = False,
    ):
        self.examples: List[Dict]        = []
        self.qids:     List[str]         = []
        self.golds:    List[List[str]]   = []

        cls_id  = tokenizer.cls_token_id
        sep_id  = tokenizer.sep_token_id
        pad_id  = tokenizer.pad_token_id
        mask_id = tokenizer.mask_token_id

        for ex in hf_dataset:
            qid        = ex["id"]
            question   = ex["question"].strip()
            context    = ex["context"].strip()
            answers    = ex["answers"]
            impossible = bool(ex.get("is_impossible", False))

            # ── Build gold answers for eval ──────────────────────────
            if version_2 and impossible:
                gold_texts = [""]
            else:
                gold_texts = answers["text"] if answers["text"] else [""]

            # ── Build instruction prefix tokens ──────────────────────
            question_tokens = tokenizer.encode(
                f"Question: {question} Answer:", add_special_tokens=False
            )
            context_prefix  = tokenizer.encode("Context: ", add_special_tokens=False)

            # Budget: CLS + "Context: " + ctx + question_tokens + MASK×30 + SEP
            ctx_budget = (
                max_seq_length
                - 1                          # [CLS]
                - len(context_prefix)
                - len(question_tokens)
                - _ANSWER_LEN
                - 1                          # [SEP]
            )
            ctx_budget = max(ctx_budget, 0)

            context_tokens = tokenizer.encode(context, add_special_tokens=False)[:ctx_budget]
            prefix_tokens  = context_prefix + context_tokens + question_tokens

            ids      = [cls_id] + prefix_tokens + [mask_id] * _ANSWER_LEN + [sep_id]
            real_len = len(ids)
            pad_len  = max_seq_length - real_len
            ids     += [pad_id] * pad_len

            attn = [1] * real_len + [0] * pad_len

            # condition_mask: False at the 30 answer positions
            ans_start = 1 + len(prefix_tokens)
            ans_end   = ans_start + _ANSWER_LEN

            cond = [True] * max_seq_length
            for i in range(ans_start, min(ans_end, max_seq_length)):
                cond[i] = False

            input_ids      = torch.tensor(ids,  dtype=torch.long)
            condition_mask = torch.tensor(cond, dtype=torch.bool)
            attention_mask = torch.tensor(attn, dtype=torch.long)

            # ── Build labels for training ────────────────────────────
            if is_train:
                if version_2 and impossible:
                    answer_text = "unanswerable"
                else:
                    answer_text = gold_texts[0] if gold_texts else ""

                true_ids = tokenizer.encode(answer_text, add_special_tokens=False)[:_ANSWER_LEN]

                labels = input_ids.clone()
                labels[condition_mask]      = -100
                labels[input_ids == pad_id] = -100

                target_positions = (~condition_mask).nonzero(as_tuple=True)[0]
                for i, pos in enumerate(target_positions):
                    if i < len(true_ids):
                        labels[pos] = true_ids[i]
                    else:
                        labels[pos] = -100
            else:
                labels = torch.full((max_seq_length,), -100, dtype=torch.long)

            self.examples.append({
                "input_ids":      input_ids,
                "condition_mask": condition_mask,
                "attention_mask": attention_mask,
                "labels":         labels,
                "raw_label":      torch.tensor(0.0, dtype=torch.float),
            })
            self.qids.append(qid)
            self.golds.append(gold_texts)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def run(
    version:        int,
    checkpoint_dir: str,
    output_dir:     str,
    max_seq_length: int   = _MAX_SEQ_LEN,
    batch_size:     int   = 12,
    num_epochs:     int   = 2,
    learning_rate:  float = 3e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
    num_steps:      int   = 200,
    temperature:    float = 1.0,
) -> Dict[str, float]:
    """
    Instruction-prompted diffusion evaluation for SQuAD v1 or v2.

    Args:
        version:        1 (answerable only) or 2 (includes unanswerable)
        checkpoint_dir: MDLM pre-training checkpoint directory or file
        output_dir:     where to write JSON results
        max_seq_length: 384 (wider budget for long contexts)
        batch_size:     batch size (12 recommended for 384-length sequences)
        num_epochs:     fine-tuning epochs
        learning_rate:  AdamW learning rate
        seed:           RNG seed
        use_amp:        enable AMP mixed precision
        num_steps:      reverse diffusion steps at inference
        temperature:    sampling temperature

    Returns:
        {"exact_match": float, "f1": float}
    """
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)

    version_2 = (version == 2)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = resolve_checkpoint(checkpoint_dir)

    logger.info(f"[SQuAD-Diffusion] version={version}  device={device}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    logger.info(f"Loading diffusion model from {ckpt_file} …")
    model     = load_diffusion_model_from_checkpoint(ckpt_file)
    scheduler = LinearAlphaScheduler()

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    ds_name = "rajpurkar/squad_v2" if version_2 else "rajpurkar/squad"
    hf_ds   = hf_load(ds_name, trust_remote_code=True)

    logger.info("Building instruction datasets …")
    train_ds = SquadInstructionDataset(
        hf_ds["train"], tokenizer, max_seq_length, is_train=True, version_2=version_2
    )
    eval_ds  = SquadInstructionDataset(
        hf_ds["validation"], tokenizer, max_seq_length, is_train=False, version_2=version_2
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

    logger.info(f"Fine-tuning on {len(train_ds)} examples for {num_epochs} epoch(s) …")
    finetuner.train()

    # ── Inference ────────────────────────────────────────────────────
    logger.info("Running conditional sampling on validation set …")
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    model.eval()

    predictions: Dict[str, str]        = {}
    references:  Dict[str, List[str]]  = {}

    example_idx = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        condition_mask = batch["condition_mask"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bsz            = input_ids.size(0)

        generated = mdlm_sample_conditional(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            condition_mask=condition_mask,
            scheduler=scheduler,
            attention_mask=attention_mask,
            num_steps=num_steps,
            temperature=temperature,
            device=device,
        )

        for i in range(bsz):
            decoded = decode_answer(generated[i], condition_mask[i], tokenizer)
            qid     = eval_ds.qids[example_idx]

            if version_2 and decoded.lower().startswith("unanswerable"):
                decoded = ""

            predictions[qid] = decoded
            references[qid]  = eval_ds.golds[example_idx]
            example_idx     += 1

    metrics = compute_squad_metrics(predictions, references)
    logger.info(f"  EM: {metrics['exact_match']:.2f}  F1: {metrics['f1']:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"squad_v{version}_diffusion_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results written to {out_path}")

    return metrics
