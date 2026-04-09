"""
Generation quality evaluation for the MDLM diffusion BERT model.

Generates text unconditionally using the MDLM reverse diffusion process
(absorbing diffusion sampler), then evaluates the generated sequences using
three complementary metrics:

  1. Generative Perplexity (GenPPL)
       Score generated sequences with a frozen GPT-2 Large autoregressive
       model.  Lower = generated text is more fluent / natural.

  2. MAUVE Score
       Measures the distribution-level divergence between generated text and
       real reference text (Pillutla et al., 2021).  Higher = closer to the
       human text distribution (max 1.0).
       Requires: pip install mauve-text

  3. Unigram Entropy
       Token-level diversity of the generated sequences.  Guards against
       degenerate repetition (very low entropy) or incoherent randomness
       (very high entropy).  Reference range for OpenWebText: [5.37, 5.55].

MDLM Reverse Sampling (Absorbing Diffusion):
    Start from a fully masked sequence (x_T = [MASK, MASK, …]).
    At each step from t=T down to t=1:
        1. Predict clean tokens: x̂_0 ~ Categorical(softmax(model(x_t)))
        2. For each masked position i:
             - With prob α(t-1)/α(t):   stay masked  in x_{t-1}
             - With prob 1-α(t-1)/α(t): reveal x̂_0[i] in x_{t-1}
    Return the final denoised sequence.

Reference:
  Shi et al., "Simplified and Effective Masked Diffusion Language Models",
  NeurIPS 2024.  https://arxiv.org/abs/2406.07524

  Pillutla et al., "MAUVE: Measuring the Gap Between Neural Text and Human
  Text using Divergence Frontiers", NeurIPS 2021.

Usage:
    python -m evaluate.generation_quality --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.generation_quality --checkpoint ./ckpt --num_samples 512 --steps 1000
"""

import argparse
import logging
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast

from config                import BertConfig
from model.bert            import BertModel
from model.diffusion_heads import BertForDiffusion
from training.noise_schedule import get_noise_scheduler, BaseAlphaScheduler
from evaluate.metrics      import compute_generative_perplexity, compute_unigram_entropy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Checkpoint loader
# ------------------------------------------------------------------ #

def _load_diffusion_model(
    checkpoint_path: str,
    config: Optional[BertConfig] = None,
) -> BertForDiffusion:
    ckpt       = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    if config is None:
        config = BertConfig()
    bert  = BertModel(config)
    model = BertForDiffusion(config, bert)
    model.load_state_dict(state_dict, strict=False)
    return model


# ------------------------------------------------------------------ #
# MDLM reverse diffusion sampler
# ------------------------------------------------------------------ #

@torch.no_grad()
def mdlm_sample(
    model:          BertForDiffusion,
    tokenizer:      BertTokenizerFast,
    num_samples:    int                  = 64,
    seq_len:        int                  = 128,
    num_steps:      int                  = 1000,
    noise_schedule: str                  = "linear",
    time_epsilon:   float                = 1e-3,
    device:         torch.device         = torch.device("cpu"),
    temperature:    float                = 1.0,
) -> List[str]:
    """
    Generate `num_samples` sequences using MDLM absorbing diffusion sampling.

    Algorithm (Shi et al. 2024, Section 3):
      x_T = all [MASK]
      for t from T down to 1:
        α_t   = scheduler.alpha(t / T)
        α_{t-1} = scheduler.alpha((t-1) / T)
        x̂_0 ~ Categorical(softmax(model(x_t) / temperature))
        for each masked position i:
          u ~ Uniform(0, 1)
          x_{t-1}[i] = [MASK]   if u < α_{t-1}/α_t
          x_{t-1}[i] = x̂_0[i]  otherwise

    Returns a list of decoded strings (special tokens stripped).
    """
    model.eval()
    model.to(device)

    scheduler  = get_noise_scheduler(noise_schedule)
    mask_id    = tokenizer.mask_token_id
    cls_id     = tokenizer.cls_token_id
    sep_id     = tokenizer.sep_token_id
    pad_id     = tokenizer.pad_token_id

    # Initialise: fully masked content tokens, [CLS] and [SEP] fixed
    # Shape: (num_samples, seq_len)
    x = torch.full((num_samples, seq_len), mask_id, dtype=torch.long, device=device)
    x[:, 0]  = cls_id
    x[:, -1] = sep_id

    # Positions that can be unmasked (exclude [CLS], [SEP])
    maskable = torch.ones(seq_len, dtype=torch.bool, device=device)
    maskable[0]  = False
    maskable[-1] = False

    attention_mask = torch.ones(num_samples, seq_len, dtype=torch.long, device=device)
    token_type_ids = torch.zeros(num_samples, seq_len, dtype=torch.long, device=device)

    for step in range(num_steps, 0, -1):
        t_now  = max(step / num_steps, time_epsilon)
        t_prev = max((step - 1) / num_steps, 0.0)

        alpha_t    = float(scheduler.alpha(t_now))
        alpha_prev = float(scheduler.alpha(t_prev))

        # Forward pass: predict clean tokens
        outputs = model(
            input_ids=x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs["logits"]                      # (B, L, V)

        if temperature != 1.0:
            logits = logits / temperature

        # Sample x̂_0 from model predictions
        probs  = F.softmax(logits, dim=-1)              # (B, L, V)
        B, L, V = probs.shape
        x0_hat = torch.multinomial(
            probs.view(B * L, V), num_samples=1
        ).view(B, L)                                    # (B, L)

        # Reverse step: update masked positions
        currently_masked = (x == mask_id) & maskable.unsqueeze(0)  # (B, L)
        if not currently_masked.any():
            break

        # Ratio α_{t-1} / α_t  (clamp to [0, 1] for numerical safety)
        ratio = (alpha_prev / alpha_t) if alpha_t > 0 else 0.0
        ratio = max(0.0, min(1.0, ratio))

        # For each masked position: stay masked with prob ratio, else reveal
        stay_mask = torch.bernoulli(
            torch.full_like(x, ratio, dtype=torch.float)
        ).bool()

        # New x_{t-1}: start from current x, then update unmasked revelations
        x_new = x.clone()
        reveal = currently_masked & ~stay_mask
        x_new[reveal] = x0_hat[reveal]
        x = x_new

    # Decode generated sequences
    texts = tokenizer.batch_decode(x, skip_special_tokens=True)
    return texts


# ------------------------------------------------------------------ #
# Reference text loader
# ------------------------------------------------------------------ #

def _load_reference_texts(
    num_samples: int,
    seq_len:     int,
    tokenizer:   BertTokenizerFast,
) -> List[str]:
    """
    Load reference text from Wikitext-103 validation set.
    Returns `num_samples` passages chunked to roughly `seq_len` tokens.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    logger.info("Loading Wikitext-103 validation as reference …")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation",
                      trust_remote_code=True)

    texts  = []
    buffer = []

    for ex in ds:
        line = ex["text"].strip()
        if not line:
            continue
        buffer.append(line)
        ids = tokenizer.encode(" ".join(buffer), add_special_tokens=False)
        if len(ids) >= seq_len - 2:
            # Take the first seq_len - 2 tokens and decode back to text
            chunk = tokenizer.decode(ids[: seq_len - 2], skip_special_tokens=True)
            texts.append(chunk)
            buffer = []
            if len(texts) >= num_samples:
                break

    return texts


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def run(
    checkpoint_dir: str,
    output_dir:     str,
    num_samples:    int   = 512,
    seq_len:        int   = 128,
    num_steps:      int   = 1000,
    noise_schedule: str   = "linear",
    temperature:    float = 1.0,
    batch_size:     int   = 64,
    seed:           int   = 42,
    skip_mauve:     bool  = False,
) -> Dict[str, float]:
    """
    Generate text with the MDLM model and evaluate generation quality.

    Args:
        checkpoint_dir: MDLM pre-training checkpoint directory
        output_dir:     directory to write results and generated samples
        num_samples:    number of sequences to generate
        seq_len:        sequence length (tokens, including [CLS]/[SEP])
        num_steps:      diffusion denoising steps T
        noise_schedule: "linear" or "cosine"
        temperature:    sampling temperature (1.0 = no scaling)
        batch_size:     samples per generation batch
        seed:           RNG seed
        skip_mauve:     skip MAUVE computation (requires mauve-text package)
    """
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    logger.info(f"Loading diffusion model from {ckpt_file} …")
    model = _load_diffusion_model(ckpt_file)

    os.makedirs(output_dir, exist_ok=True)

    # ---- Generate text in batches ---- #
    logger.info(
        f"Generating {num_samples} samples "
        f"(seq_len={seq_len}, T={num_steps}, schedule={noise_schedule}) …"
    )
    generated_texts: List[str] = []
    remaining = num_samples

    while remaining > 0:
        n = min(batch_size, remaining)
        batch_texts = mdlm_sample(
            model=model,
            tokenizer=tokenizer,
            num_samples=n,
            seq_len=seq_len,
            num_steps=num_steps,
            noise_schedule=noise_schedule,
            time_epsilon=1e-3,
            device=device,
            temperature=temperature,
        )
        generated_texts.extend(batch_texts)
        remaining -= n
        logger.info(f"  Generated {len(generated_texts)}/{num_samples}")

    # Save generated samples
    gen_path = os.path.join(output_dir, "generated_samples.txt")
    with open(gen_path, "w", encoding="utf-8") as f:
        for text in generated_texts:
            f.write(text.replace("\n", " ") + "\n")
    logger.info(f"Generated samples saved to {gen_path}")

    # ---- Load reference texts ---- #
    reference_texts = _load_reference_texts(num_samples, seq_len, tokenizer)
    logger.info(f"  Reference texts: {len(reference_texts):,}")

    results: Dict[str, float] = {}

    # ---- Generative Perplexity (GPT-2 Large scorer) ---- #
    logger.info("Computing Generative Perplexity (GPT-2 Large) …")
    try:
        gen_ppl = compute_generative_perplexity(generated_texts, device=device)
        results["generative_perplexity"] = gen_ppl
        logger.info(f"  Generative PPL = {gen_ppl:.2f}")
    except Exception as exc:
        logger.warning(f"  Generative PPL failed: {exc}")
        results["generative_perplexity"] = float("nan")

    # ---- Unigram Entropy ---- #
    logger.info("Computing Unigram Entropy …")
    try:
        entropy = compute_unigram_entropy(generated_texts, tokenizer)
        results["unigram_entropy"] = entropy
        logger.info(f"  Unigram Entropy = {entropy:.4f}  (reference range: [5.37, 5.55])")
    except Exception as exc:
        logger.warning(f"  Entropy failed: {exc}")
        results["unigram_entropy"] = float("nan")

    # ---- MAUVE ---- #
    if not skip_mauve:
        logger.info("Computing MAUVE score …")
        try:
            import mauve as mauve_lib
            out = mauve_lib.compute_mauve(
                p_text=generated_texts,
                q_text=reference_texts,
                device_id=0 if device.type == "cuda" else -1,
                max_text_length=seq_len * 4,  # char budget
                verbose=False,
            )
            results["mauve"] = float(out.mauve)
            logger.info(f"  MAUVE = {out.mauve:.4f}")
        except ImportError:
            logger.warning(
                "  MAUVE skipped: install with 'pip install mauve-text'"
            )
            results["mauve"] = float("nan")
        except Exception as exc:
            logger.warning(f"  MAUVE failed: {exc}")
            results["mauve"] = float("nan")
    else:
        results["mauve"] = float("nan")

    # ---- Write summary ---- #
    import json
    out_path = os.path.join(output_dir, "generation_quality_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Generation Quality Results ===")
    for k, v in results.items():
        print(f"  {k:<28} {v:.4f}")
    logger.info(f"Results written to {out_path}")

    return results


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MDLM generation quality: Generative PPL + MAUVE + Entropy"
    )
    p.add_argument("--checkpoint",     required=True,
                   help="Path to MDLM checkpoint.pt or its parent directory")
    p.add_argument("--output_dir",     default="./results")
    p.add_argument("--num_samples",    type=int,   default=512,
                   help="Number of sequences to generate")
    p.add_argument("--seq_len",        type=int,   default=128,
                   help="Sequence length in tokens (including [CLS]/[SEP])")
    p.add_argument("--num_steps",      type=int,   default=1000,
                   help="Number of MDLM reverse diffusion steps T")
    p.add_argument("--noise_schedule", default="linear",
                   choices=["linear", "cosine"])
    p.add_argument("--temperature",    type=float, default=1.0,
                   help="Sampling temperature (1.0 = no scaling)")
    p.add_argument("--batch_size",     type=int,   default=64,
                   help="Samples per generation batch")
    p.add_argument("--skip_mauve",     action="store_true",
                   help="Skip MAUVE (requires pip install mauve-text)")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        noise_schedule=args.noise_schedule,
        temperature=args.temperature,
        batch_size=args.batch_size,
        seed=args.seed,
        skip_mauve=args.skip_mauve,
    )
