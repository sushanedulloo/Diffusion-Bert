"""
Diffusion-style evaluation for SWAG (zero-shot, no fine-tuning).

Uses PLL scoring instead of discriminative fine-tuning:
  - For each of the 4 candidate endings, build the full sequence
    [CLS] ctx [SEP] ending_i [SEP] and score the ending tokens
    using the PLL pattern (mask one token at a time, read log-prob)
  - Predict the ending with the highest total log-prob
  - Metric: Accuracy

This is analogous to the GPT-1 evaluation of SWAG: score each candidate
as a language model continuation, no task-specific training.

Usage:
    python evaluate.py swag --checkpoint ./ckpt --eval_mode generative
"""

import json
import logging
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from training.noise_schedule  import LinearAlphaScheduler
from evaluate.diffusion_utils import (
    load_diffusion_model_from_checkpoint,
    resolve_checkpoint,
    score_completion_pll,
)
from evaluate.metrics import compute_accuracy

logger = logging.getLogger(__name__)

_MAX_SEQ_LEN = 128


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class SwagPLLDataset(Dataset):
    """
    SWAG examples encoded for PLL scoring.

    For each of the 4 endings, stores:
        input_ids:      (4, max_seq_length) LongTensor
        attention_mask: (4, max_seq_length) LongTensor
        token_type_ids: (4, max_seq_length) LongTensor
        completion_starts: (4,) LongTensor  — first ending token position per choice
        completion_ends:   (4,) LongTensor  — last ending token position + 1 per choice
        label:          scalar LongTensor   — 0-3
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = _MAX_SEQ_LEN,
    ):
        self.examples: List[Dict[str, torch.Tensor]] = []

        sep_id = tokenizer.sep_token_id

        for ex in hf_dataset:
            ctx     = ex["sent1"] + " " + ex["sent2"].rstrip()
            endings = [ex["ending0"], ex["ending1"], ex["ending2"], ex["ending3"]]
            label   = int(ex["label"])

            encs = tokenizer(
                [ctx] * 4,
                endings,
                max_length=max_seq_length,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )

            input_ids      = encs["input_ids"]            # (4, L)
            attention_mask = encs["attention_mask"]        # (4, L)
            token_type_ids = encs.get(
                "token_type_ids",
                torch.zeros_like(input_ids)
            )                                              # (4, L)

            # Find ending boundaries using token_type_ids:
            # Segment A (type=0): [CLS] ctx [SEP]
            # Segment B (type=1): ending_i [SEP]
            # completion_start = first position where type_id=1
            # completion_end   = last position where type_id=1 AND not [SEP] + 1

            comp_starts = torch.zeros(4, dtype=torch.long)
            comp_ends   = torch.zeros(4, dtype=torch.long)

            for c in range(4):
                tids  = token_type_ids[c]      # (L,)
                ids_c = input_ids[c]           # (L,)
                attn  = attention_mask[c]      # (L,)

                # Positions belonging to the ending (segment B, real, not SEP)
                ending_mask = (tids == 1) & (attn == 1) & (ids_c != sep_id)
                positions   = ending_mask.nonzero(as_tuple=True)[0]

                if len(positions) > 0:
                    comp_starts[c] = positions[0].item()
                    comp_ends[c]   = positions[-1].item() + 1
                else:
                    # Empty ending: won't contribute to PLL
                    comp_starts[c] = 0
                    comp_ends[c]   = 0

            self.examples.append({
                "input_ids":        input_ids,
                "attention_mask":   attention_mask,
                "token_type_ids":   token_type_ids,
                "completion_starts": comp_starts,
                "completion_ends":   comp_ends,
                "label":            torch.tensor(label, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def run(
    checkpoint_dir: str,
    output_dir:     str,
    max_seq_length: int = _MAX_SEQ_LEN,
    batch_size:     int = 16,
    seed:           int = 42,
    max_batches:    int = None,
) -> Dict[str, float]:
    """
    Zero-shot PLL scoring evaluation for SWAG (no fine-tuning).

    For each SWAG example, scores all 4 candidate endings via PLL
    (sum of per-token log-probs conditioned on the context) and
    picks the highest-scoring ending.

    Args:
        checkpoint_dir: MDLM pre-training checkpoint directory or file
        output_dir:     where to write JSON results
        max_seq_length: sequence length (128 for SWAG)
        batch_size:     examples per PLL scoring batch
        seed:           RNG seed
        max_batches:    cap on batches (None = full evaluation)

    Returns:
        {"accuracy": float}
    """
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = resolve_checkpoint(checkpoint_dir)

    logger.info(f"[SWAG-Diffusion] device={device}  (zero-shot PLL scoring)")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    logger.info(f"Loading diffusion model from {ckpt_file} …")
    model = load_diffusion_model_from_checkpoint(ckpt_file)
    model.eval()
    model.to(device)

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    hf_ds = hf_load("swag", "regular", trust_remote_code=True)

    logger.info("Building PLL dataset …")
    eval_ds = SwagPLLDataset(hf_ds["validation"], tokenizer, max_seq_length)

    loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    preds:  List[int] = []
    labels: List[int] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # input_ids: (1, 4, L) — batch_size=1 here for simplicity
        input_ids_4        = batch["input_ids"][0].to(device)          # (4, L)
        attention_mask_4   = batch["attention_mask"][0].to(device)     # (4, L)
        completion_starts  = batch["completion_starts"][0]             # (4,)
        completion_ends    = batch["completion_ends"][0]               # (4,)
        gold_label         = int(batch["label"][0].item())

        scores = torch.zeros(4, device=device)

        for c in range(4):
            c_start = int(completion_starts[c].item())
            c_end   = int(completion_ends[c].item())

            if c_end <= c_start:
                scores[c] = -1e9   # empty ending: assign very low score
                continue

            # Score a single-example "batch" (B=1)
            ids_1  = input_ids_4[c].unsqueeze(0)      # (1, L)
            attn_1 = attention_mask_4[c].unsqueeze(0) # (1, L)

            score = score_completion_pll(
                model=model,
                full_ids=ids_1,
                completion_start=c_start,
                completion_end=c_end,
                attention_mask=attn_1,
                tokenizer=tokenizer,
                device=device,
            )
            scores[c] = score[0]

        pred = int(scores.argmax().item())
        preds.append(pred)
        labels.append(gold_label)

        if (batch_idx + 1) % 500 == 0:
            acc_so_far = sum(p == l for p, l in zip(preds, labels)) / len(preds)
            logger.info(f"  {batch_idx + 1} examples — running accuracy: {acc_so_far:.4f}")

    metrics = compute_accuracy(preds, labels)
    logger.info(f"  SWAG Accuracy: {metrics['accuracy']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "swag_diffusion_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results written to {out_path}")

    return metrics
