"""
Shared utilities for diffusion-style evaluation.

  load_diffusion_model_from_checkpoint — load BertForDiffusion from MDLM ckpt
  mdlm_sample_conditional             — conditional reverse diffusion sampler
  score_completion_pll                — PLL scoring for zero-shot MC tasks
"""

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

from config                  import BertConfig
from model.bert              import BertModel
from model.diffusion_heads   import BertForDiffusion

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Checkpoint loading
# ------------------------------------------------------------------ #

def load_diffusion_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[BertConfig] = None,
) -> BertForDiffusion:
    """
    Load a BertForDiffusion model (backbone + mlm_head) from an MDLM checkpoint.

    The checkpoint contains keys "bert.*" and "mlm_head.*".  Both are loaded.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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


def resolve_checkpoint(checkpoint_dir: str) -> str:
    """Return path to checkpoint.pt, accepting either a file or directory."""
    if os.path.isdir(checkpoint_dir):
        return os.path.join(checkpoint_dir, "checkpoint.pt")
    return checkpoint_dir


# ------------------------------------------------------------------ #
# Conditional MDLM reverse diffusion sampler
# ------------------------------------------------------------------ #

@torch.no_grad()
def mdlm_sample_conditional(
    model:           BertForDiffusion,
    tokenizer,
    input_ids:       torch.LongTensor,   # (B, L)
    condition_mask:  torch.BoolTensor,   # (B, L)  True = fixed, False = denoised
    scheduler,                           # BaseAlphaScheduler
    attention_mask:  torch.LongTensor,   # (B, L)
    num_steps:       int   = 200,
    temperature:     float = 1.0,
    device:          Optional[torch.device] = None,
) -> torch.LongTensor:
    """
    Conditional MDLM absorbing-diffusion sampler.

    Condition positions (condition_mask=True) are fixed throughout.
    Target positions (condition_mask=False) start as [MASK] and are
    iteratively revealed by the reverse diffusion process.

    Args:
        input_ids:      (B, L) — full sequence; condition tokens pre-filled,
                        target positions will be overwritten with [MASK]
        condition_mask: (B, L) bool — True = fixed condition position
        scheduler:      BaseAlphaScheduler instance (linear or cosine)
        attention_mask: (B, L) — 1 for real tokens, 0 for padding
        num_steps:      T — number of reverse diffusion steps
        temperature:    sampling temperature (1.0 = no scaling)
        device:         inference device; defaults to model device

    Returns:
        (B, L) LongTensor — full sequence with target positions filled in
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    mask_id = tokenizer.mask_token_id

    x               = input_ids.clone().to(device)
    condition_mask  = condition_mask.to(device)
    attention_mask  = attention_mask.to(device)

    # Initialise: mask every target position
    x[~condition_mask] = mask_id

    for step in range(num_steps, 0, -1):
        t_now  = step / num_steps
        t_prev = (step - 1) / num_steps

        alpha_t    = float(scheduler.alpha(t_now))
        alpha_prev = float(scheduler.alpha(t_prev))

        logits = model(x, attention_mask)["logits"]   # (B, L, V)

        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        B, L, V = probs.shape
        x0_hat = torch.multinomial(probs.view(B * L, V), num_samples=1).view(B, L)

        currently_masked = (x == mask_id) & ~condition_mask
        if not currently_masked.any():
            break

        # alpha_prev / alpha_t: probability of a masked position staying masked
        ratio = (alpha_prev / alpha_t) if alpha_t > 1e-8 else 0.0
        ratio = max(0.0, min(1.0, ratio))

        stay   = torch.bernoulli(torch.full((B, L), ratio, device=device)).bool()
        reveal = currently_masked & ~stay
        x[reveal] = x0_hat[reveal]

    return x


# ------------------------------------------------------------------ #
# PLL completion scorer
# ------------------------------------------------------------------ #

@torch.no_grad()
def score_completion_pll(
    model:            BertForDiffusion,
    full_ids:         torch.LongTensor,  # (B, L)
    completion_start: int,
    completion_end:   int,
    attention_mask:   torch.LongTensor,  # (B, L)
    tokenizer,
    device:           torch.device,
) -> torch.Tensor:
    """
    Compute sum of log P(completion_token | rest) over completion positions.

    Uses the PLL pattern: for each position in [completion_start, completion_end),
    mask that position, run the model, read log-prob of the true token.

    Args:
        full_ids:         (B, L) — full sequence including context and completion
        completion_start: first position of completion tokens (inclusive)
        completion_end:   last position + 1 of completion tokens (exclusive)
        attention_mask:   (B, L)
        tokenizer:        BertTokenizerFast
        device:           torch device

    Returns:
        (B,) float tensor — summed log-prob of completion tokens per example
    """
    mask_id     = tokenizer.mask_token_id
    B           = full_ids.size(0)
    total_score = torch.zeros(B, device=device)

    for pos in range(completion_start, completion_end):
        masked            = full_ids.clone()
        masked[:, pos]    = mask_id

        logits  = model(masked, attention_mask)["logits"]          # (B, L, V)
        log_p   = F.log_softmax(logits[:, pos, :], dim=-1)         # (B, V)
        true_id = full_ids[:, pos].unsqueeze(1)                    # (B, 1)
        total_score += log_p.gather(1, true_id).squeeze(1)         # (B,)

    return total_score
