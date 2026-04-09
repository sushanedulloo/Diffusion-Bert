"""
Pseudo-Log-Likelihood (PLL) Perplexity for the diffusion BERT model.

Standard autoregressive perplexity is undefined for masked / diffusion
language models.  The correct proxy is Pseudo-Log-Likelihood (Salazar et al.,
2020): for each token position i, mask it with [MASK], run the model forward,
and collect log P(x_i | x_{-i}).  PLL perplexity is then:

    PPL_PLL = exp( -1/N * Σ_i log P(x_i | x_{-i}) )

This requires one forward pass per token position per sequence.  For
efficiency the loop is over positions (not over examples), so each forward
pass processes the full batch in parallel.

Datasets evaluated (zero-shot, no fine-tuning):
  - Wikitext-103   (standard LM benchmark)
  - Lambada        (long-range dependency, last-word prediction)
  - Penn Treebank  (PTB, classic LM benchmark)
  - AG News        (news domain — tests domain shift)
  - PubMed         (biomedical abstracts — tests domain shift)
  - ArXiv          (scientific abstracts — tests domain shift)

The model loaded is BertForDiffusion from the MDLM pre-training checkpoint.

Reference:
  Salazar et al., "Masked Language Model Scoring", ACL 2020.
  Shi et al.,     "Simplified and Effective Masked Diffusion Language Models",
                  NeurIPS 2024.

Usage:
    python -m evaluate.perplexity --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.perplexity --checkpoint ./ckpt --datasets wikitext lambada
"""

import argparse
import logging
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

from config             import BertConfig
from model.bert         import BertModel
from model.diffusion_heads import BertForDiffusion

logger = logging.getLogger(__name__)

# Supported zero-shot evaluation datasets
SUPPORTED_DATASETS = ["wikitext", "lambada", "ptb", "ag_news", "pubmed", "arxiv"]

# HuggingFace dataset IDs and configs
# ag_news: short news articles — tests domain shift (news)
# pubmed: PubMed biomedical abstracts — tests domain shift (biomedical)
# arxiv: arXiv scientific abstracts — tests domain shift (scientific)
_DATASET_CONFIGS = {
    "wikitext": ("wikitext",         "wikitext-103-raw-v1", "test"),
    "lambada":  ("lambada",          None,                  "test"),
    "ptb":      ("ptb_text_only",    None,                  "test"),
    "ag_news":  ("ag_news",          None,                  "test"),
    "pubmed":   ("scientific_papers", "pubmed",             "test"),
    "arxiv":    ("scientific_papers", "arxiv",              "test"),
}

# Text field name per dataset
_TEXT_FIELD = {
    "wikitext": "text",
    "lambada":  "text",
    "ptb":      "sentence",
    "ag_news":  "description",
    "pubmed":   "abstract",
    "arxiv":    "abstract",
}


# ------------------------------------------------------------------ #
# Checkpoint loader (loads full BertForDiffusion, not just backbone)
# ------------------------------------------------------------------ #

def load_diffusion_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[BertConfig] = None,
) -> BertForDiffusion:
    """
    Load a BertForDiffusion model from an MDLM pre-training checkpoint.

    The checkpoint has keys like "bert.*" and "mlm_head.*".
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    if config is None:
        config = BertConfig()

    bert  = BertModel(config)
    model = BertForDiffusion(config, bert)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(
            f"Missing keys ({len(missing)}): {missing[:3]}"
            f"{'…' if len(missing) > 3 else ''}"
        )
    return model


# ------------------------------------------------------------------ #
# Tokenised text dataset
# ------------------------------------------------------------------ #

class TokenisedTextDataset(Dataset):
    """
    Chunks a flat list of token ids into non-overlapping windows of
    `max_seq_length` tokens, wrapped with [CLS] and [SEP].
    """

    def __init__(
        self,
        token_ids:      List[int],
        tokenizer:      BertTokenizerFast,
        max_seq_length: int = 512,
    ):
        content_len = max_seq_length - 2   # reserve [CLS] and [SEP]
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        self.examples: List[Dict[str, torch.Tensor]] = []

        for start in range(0, len(token_ids), content_len):
            chunk = token_ids[start : start + content_len]
            if not chunk:
                continue

            ids  = [cls_id] + chunk + [sep_id]
            slen = len(ids)
            pad  = max_seq_length - slen
            ids  += [pad_id] * pad
            mask  = [1] * slen + [0] * pad
            ttype = [0] * max_seq_length

            self.examples.append({
                "input_ids":      torch.tensor(ids,   dtype=torch.long),
                "attention_mask": torch.tensor(mask,  dtype=torch.long),
                "token_type_ids": torch.tensor(ttype, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------ #
# PLL perplexity computation
# ------------------------------------------------------------------ #

@torch.no_grad()
def compute_pll_perplexity(
    model:          BertForDiffusion,
    dataset:        TokenisedTextDataset,
    tokenizer:      BertTokenizerFast,
    batch_size:     int = 16,
    device:         torch.device = torch.device("cpu"),
    max_batches:    Optional[int] = None,
) -> float:
    """
    Compute PLL perplexity over a TokenisedTextDataset.

    For each token position i (skipping [CLS], [SEP], and [PAD]):
      1. Replace token at position i with [MASK] in all batch examples.
      2. Forward pass through model.
      3. Collect log P(true_token_i | masked_context).

    Returns exp(-mean_log_prob) — the PLL perplexity.
    """
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mask_id      = tokenizer.mask_token_id
    special_ids  = set(tokenizer.all_special_ids)

    total_log_prob = 0.0
    total_tokens   = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids      = batch["input_ids"].to(device)       # (B, L)
        attention_mask = batch["attention_mask"].to(device)   # (B, L)
        token_type_ids = batch["token_type_ids"].to(device)   # (B, L)

        B, L = input_ids.shape

        # Iterate over positions — one forward pass per position across full batch
        for pos in range(L):
            # Build a boolean mask of examples where position `pos` is maskable
            is_real    = attention_mask[:, pos].bool()              # not padding
            is_special = torch.zeros(B, dtype=torch.bool, device=device)
            for sid in special_ids:
                is_special |= input_ids[:, pos].eq(sid)
            maskable = is_real & ~is_special

            if not maskable.any():
                continue

            # Create masked input
            masked_ids = input_ids.clone()
            masked_ids[:, pos] = mask_id

            outputs  = model(
                input_ids=masked_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits   = outputs["logits"]                        # (B, L, V)
            log_prob = F.log_softmax(logits[:, pos, :], dim=-1) # (B, V)

            # Gather log P of the true token at this position
            true_ids    = input_ids[:, pos].unsqueeze(1)            # (B, 1)
            true_log_p  = log_prob.gather(1, true_ids).squeeze(1)   # (B,)

            total_log_prob += (true_log_p * maskable.float()).sum().item()
            total_tokens   += maskable.sum().item()

        if batch_idx % 10 == 0:
            logger.info(
                f"  Processed {batch_idx + 1} batches | "
                f"tokens so far: {total_tokens:,}"
            )

    if total_tokens == 0:
        return float("inf")

    mean_log_prob = total_log_prob / total_tokens
    return math.exp(-mean_log_prob)


# ------------------------------------------------------------------ #
# Per-dataset loader
# ------------------------------------------------------------------ #

def _load_dataset_tokens(
    dataset_name: str,
    tokenizer:    BertTokenizerFast,
    max_chars:    Optional[int] = None,
) -> List[int]:
    """Download the dataset and tokenise all text into a flat token list."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    ds_id, config, split = _DATASET_CONFIGS[dataset_name]
    text_field           = _TEXT_FIELD[dataset_name]

    logger.info(f"Loading {dataset_name} ({ds_id}, split={split}) …")
    kwargs = {"trust_remote_code": True}
    if config:
        hf_ds = hf_load(ds_id, config, split=split, **kwargs)
    else:
        hf_ds = hf_load(ds_id, split=split, **kwargs)

    all_text = "\n\n".join(
        ex[text_field] for ex in hf_ds if ex[text_field].strip()
    )
    if max_chars is not None:
        all_text = all_text[:max_chars]

    logger.info(f"  Tokenising {len(all_text):,} characters …")
    token_ids = tokenizer.encode(all_text, add_special_tokens=False)
    logger.info(f"  {len(token_ids):,} tokens.")
    return token_ids


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def run(
    checkpoint_dir: str,
    output_dir:     str,
    datasets:       Optional[List[str]] = None,
    max_seq_length: int                 = 128,
    batch_size:     int                 = 16,
    max_batches:    Optional[int]       = None,
    seed:           int                 = 42,
) -> Dict[str, float]:
    """
    Compute PLL perplexity on one or more zero-shot datasets.

    Args:
        checkpoint_dir: path to MDLM pre-training checkpoint directory
        output_dir:     directory to write results JSON
        datasets:       list of dataset names (default: all six)
        max_seq_length: sequence length (128 or 512)
        batch_size:     examples per forward pass
        max_batches:    cap on batches per dataset (None = full evaluation)
        seed:           for reproducibility
    """
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)

    if datasets is None:
        datasets = SUPPORTED_DATASETS

    invalid = set(datasets) - set(SUPPORTED_DATASETS)
    if invalid:
        raise ValueError(f"Unknown dataset(s): {invalid}. Choose from {SUPPORTED_DATASETS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    logger.info(f"Loading diffusion model from {ckpt_file} …")
    model = load_diffusion_model_from_checkpoint(ckpt_file)

    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, float] = {}

    for ds_name in datasets:
        logger.info(f"\n{'='*50}\nEvaluating PLL perplexity on {ds_name.upper()}\n{'='*50}")
        try:
            token_ids = _load_dataset_tokens(ds_name, tokenizer)
            ds        = TokenisedTextDataset(token_ids, tokenizer, max_seq_length)
            logger.info(f"  {len(ds):,} chunks of length {max_seq_length}")

            ppl = compute_pll_perplexity(
                model=model,
                dataset=ds,
                tokenizer=tokenizer,
                batch_size=batch_size,
                device=device,
                max_batches=max_batches,
            )
            results[ds_name] = ppl
            print(f"  {ds_name:>12}  PLL-PPL = {ppl:.2f}")

        except Exception as exc:
            logger.warning(f"  {ds_name} failed: {exc}")
            results[ds_name] = float("nan")

    # Write JSON summary
    import json
    out_path = os.path.join(output_dir, "perplexity_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults written to {out_path}")

    print("\n=== PLL Perplexity Results ===")
    for name, ppl in results.items():
        print(f"  {name:<12} {ppl:.2f}")

    return results


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PLL perplexity evaluation for diffusion BERT"
    )
    p.add_argument("--checkpoint",     required=True,
                   help="Path to MDLM checkpoint.pt or its parent directory")
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--datasets",       nargs="+", default=None,
                   choices=SUPPORTED_DATASETS,
                   help=f"Datasets to evaluate. Default: all. Choices: {SUPPORTED_DATASETS}")
    p.add_argument("--max_seq_length", type=int, default=128,
                   help="Chunk length (128 or 512)")
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--max_batches",    type=int, default=None,
                   help="Cap on batches per dataset (for quick testing)")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        datasets=args.datasets,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        seed=args.seed,
    )
