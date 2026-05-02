"""
MDLM Pre-training Trainer.

Implements the Masked Diffusion Language Modeling (MDLM) pre-training
objective from Shi et al. (2024):
    https://arxiv.org/abs/2406.07524

Algorithm (one training step)
──────────────────────────────
1.  Sample a timestep t_i ~ Uniform[ε, 1)  for each example i in the batch.
2.  Compute masking probability p_mask(t_i) = 1 − α(t_i)  from the noise schedule.
3.  Sample a binary mask for every maskable token position independently:
        masked[i,j] = Bernoulli(p_mask(t_i))  if labels[i,j] != -100
4.  Replace masked positions with [MASK]:
        noised_ids[i,j] = [MASK]  if masked[i,j] else input_ids[i,j]
5.  Forward pass through BertForDiffusion → logits (B, S, V).
6.  Compute per-token cross-entropy only at masked positions.
7.  Weight each loss term by w(t_i) = −dα/dt(t_i) / (1 − α(t_i) + ε).
8.  Normalise by the number of maskable tokens (or by sequence/batch — configurable).
9.  Backward + optimizer step (with gradient accumulation, AMP, grad clipping).

Key differences from BertPreTrainer (standard MLM+NSP):
  - Masking is applied here (not in the collator).
  - All masked tokens → [MASK] (no 80/10/10 scheme).
  - Loss is weighted by w(t) per example.
  - No NSP objective.
  - Noise schedule (α) replaces the fixed 15 % masking rate.

Everything else is identical to BertPreTrainer:
  - Two-phase sequence-length curriculum (128 → 512).
  - Gradient accumulation + AMP + DataParallel.
  - Same checkpoint format (compatible with BertPreTrainer checkpoints).
  - Same optimizer / scheduler.
"""

import os
import time
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from training.optimizer       import get_bert_optimizer
from training.scheduler       import get_linear_schedule_with_warmup
from training.noise_schedule  import BaseAlphaScheduler, LinearAlphaScheduler

logger = logging.getLogger(__name__)


class MDLMPreTrainer:
    """
    MDLM pre-training trainer.

    Args:
        model:           BertForDiffusion instance
        config:          BertConfig — supplies default hyper-parameters
        train_dataset:   BertPretrainingDataset (or any Dataset)
        collator:        DiffusionCollator
        scheduler:       BaseAlphaScheduler (noise schedule α(t));
                         defaults to LinearAlphaScheduler if None
        output_dir:      directory for checkpoints
        use_amp:         enable automatic mixed precision (FP16 on CUDA)
    """

    def __init__(
        self,
        model,
        config,
        train_dataset,
        collator,
        scheduler:            Optional[BaseAlphaScheduler] = None,
        output_dir:           str  = "./output_mdlm",
        use_amp:              bool = True,
        post_save_callback=None,
    ):
        self.config        = config
        self.train_dataset = train_dataset
        self.collator      = collator
        self.output_dir    = output_dir
        self.post_save_callback = post_save_callback

        self.noise_scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon    = getattr(config, "time_epsilon",    1e-3)
        self.loss_norm_type  = getattr(config, "loss_norm_type",  "token")

        # Device & multi-GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        self.use_amp = use_amp and (self.device.type == "cuda")

    # ------------------------------------------------------------------ #
    # Main training entry point
    # ------------------------------------------------------------------ #

    def train(
        self,
        num_train_steps:              int   = 1_000_000,
        batch_size:                   int   = 256,
        gradient_accumulation_steps:  int   = 1,
        learning_rate:                float = 1e-4,
        warmup_steps:                 int   = 10_000,
        weight_decay:                 float = 0.01,
        max_grad_norm:                float = 1.0,
        save_steps:                   int   = 10_000,
        logging_steps:                int   = 100,
        phase1_ratio:                 float = 0.9,
        phase1_max_seq_len:           int   = 128,
        phase2_max_seq_len:           int   = 512,
        dataloader_num_workers:       int   = 4,
        resume_from_checkpoint:       Optional[str] = None,
    ) -> None:
        """
        Run MDLM pre-training.

        The two-phase curriculum (Phase 1: seq_len=128, Phase 2: seq_len=512)
        is preserved from the standard trainer for training efficiency.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        effective_batch = max(1, batch_size // max(1, torch.cuda.device_count()))
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=effective_batch,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=dataloader_num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        optimizer = get_bert_optimizer(
            self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_steps,
        )
        scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None

        global_step  = 0
        running_loss = 0.0
        phase1_steps = int(num_train_steps * phase1_ratio)

        if resume_from_checkpoint:
            global_step = self._load_checkpoint(
                resume_from_checkpoint, optimizer, lr_scheduler, scaler
            )
            logger.info(f"Resumed from step {global_step}")

        logger.info(
            f"MDLM pre-training: {num_train_steps:,} steps | "
            f"batch={batch_size} | schedule={self.noise_scheduler.__class__.__name__} | "
            f"ε={self.time_epsilon} | norm={self.loss_norm_type}"
        )

        optimizer.zero_grad()

        while global_step < num_train_steps:
            for batch in train_loader:
                if global_step >= num_train_steps:
                    break

                # Phase curriculum: trim sequences in Phase 1
                current_max_len = (
                    phase1_max_seq_len if global_step < phase1_steps
                    else phase2_max_seq_len
                )
                batch = self._to_device(batch)
                batch = _trim_to_length(batch, current_max_len)

                # ── MDLM forward + loss ──────────────────────────────
                if self.use_amp:
                    with autocast():
                        loss = self._compute_mdlm_loss(batch)
                    loss = loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    loss = self._compute_mdlm_loss(batch)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                running_loss += loss.item() * gradient_accumulation_steps

                # ── Optimizer step ───────────────────────────────────
                micro_step = global_step * gradient_accumulation_steps
                if (micro_step + 1) % gradient_accumulation_steps == 0 or \
                        gradient_accumulation_steps == 1:
                    if self.use_amp:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # ── Logging ──────────────────────────────────────────
                if global_step % logging_steps == 0 and global_step > 0:
                    avg_loss   = running_loss / logging_steps
                    current_lr = lr_scheduler.get_last_lr()[0]
                    logger.info(
                        f"[MDLM] step {global_step:>7,}/{num_train_steps:,} | "
                        f"loss={avg_loss:.4f} | lr={current_lr:.2e} | "
                        f"seq_len={current_max_len}"
                    )
                    running_loss = 0.0

                # ── Checkpoint ───────────────────────────────────────
                if global_step % save_steps == 0 and global_step > 0:
                    self._save_checkpoint(global_step, optimizer, lr_scheduler, scaler)

        # Final checkpoint
        self._save_checkpoint(global_step, optimizer, lr_scheduler, scaler, tag="final")
        logger.info("MDLM pre-training complete.")

    # ------------------------------------------------------------------ #
    # MDLM loss computation (core of the algorithm)
    # ------------------------------------------------------------------ #

    def _compute_mdlm_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute the MDLM loss for one batch.

        Steps:
          1. Sample t ~ U[ε, 1) per example
          2. p_mask(t) = 1 − α(t)
          3. Apply stochastic masking to maskable positions
          4. Forward pass → logits
          5. Weighted cross-entropy at masked positions
          6. Normalise

        Args:
            batch: dict with keys input_ids, attention_mask, token_type_ids, labels
                   (labels = original token ids, -100 for special/padding tokens)

        Returns:
            Scalar loss tensor.
        """
        input_ids      = batch["input_ids"]       # (B, S)
        attention_mask = batch["attention_mask"]   # (B, S)
        token_type_ids = batch.get("token_type_ids")
        labels         = batch["labels"]           # (B, S)  — -100 for non-maskable

        B, S = input_ids.shape
        maskable = (labels != -100)                # (B, S) bool

        # ── 1. Sample diffusion timesteps ──────────────────────────
        t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            B, device=self.device
        )                                          # (B,)

        # ── 2. Per-token masking probability ───────────────────────
        alpha_t = self.noise_scheduler.alpha(t)    # (B,)
        p_mask  = (1.0 - alpha_t).unsqueeze(1).expand(B, S)   # (B, S)

        # ── 3. Stochastic mask — only applies to maskable tokens ───
        rand    = torch.rand(B, S, device=self.device)
        masked  = (rand < p_mask) & maskable       # (B, S) bool

        # ── 4. Apply forward process (absorbing state: → [MASK]) ───
        mask_token_id = self._get_mask_token_id()
        noised_ids = torch.where(masked, torch.full_like(input_ids, mask_token_id), input_ids)

        # ── 5. Forward pass ─────────────────────────────────────────
        outputs = self.model(
            input_ids=noised_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs["logits"]                 # (B, S, V)
        # Reduce over DataParallel if needed
        if logits.dim() > 3:
            logits = logits.mean(dim=0)

        # ── 6. Weighted cross-entropy ────────────────────────────────
        # CE against the ORIGINAL (clean) token ids at every position;
        # only masked positions contribute via the masked.float() gate.
        token_nll = F.cross_entropy(
            logits.transpose(1, 2),   # (B, V, S) — required by F.cross_entropy
            input_ids,                 # (B, S)
            reduction="none",
        )                              # (B, S)

        w = self.noise_scheduler.weight(t).unsqueeze(1).expand(B, S)  # (B, S)
        token_nll = token_nll * w * masked.float()                     # (B, S)

        # ── 7. Normalise ─────────────────────────────────────────────
        num_maskable = maskable.sum().clamp_min(1)
        if self.loss_norm_type == "token":
            loss = token_nll.sum() / num_maskable
        elif self.loss_norm_type == "sequence":
            # Per-sequence token count, then average over batch
            seq_counts = maskable.sum(dim=1, keepdim=True).clamp_min(1).float()
            loss = (token_nll / seq_counts).sum() / B
        elif self.loss_norm_type == "batch":
            loss = token_nll.sum() / B
        else:
            raise ValueError(f"Invalid loss_norm_type '{self.loss_norm_type}'")

        return loss

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_mask_token_id(self) -> int:
        """Return the [MASK] token id stored in config or default (103 for bert-base-uncased)."""
        return getattr(self.config, "mask_token_id", 103)

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _save_checkpoint(
        self,
        step:       int,
        optimizer,
        scheduler,
        scaler,
        tag:        Optional[str] = None,
    ) -> None:
        import shutil
        import tempfile

        name = tag if tag else f"step_{step:07d}"

        # When a GDrive callback is set, use a temp dir — nothing persists on disk.
        # Otherwise fall back to output_dir for local storage.
        if self.post_save_callback:
            save_dir = tempfile.mkdtemp(prefix=f"bert_ckpt_{name}_")
        else:
            save_dir = os.path.join(self.output_dir, f"checkpoint_{name}")
            os.makedirs(save_dir, exist_ok=True)

        raw_model = (
            self.model.module if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        state = {
            "global_step":      step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
        }
        if scaler is not None:
            state["scaler_state"] = scaler.state_dict()

        torch.save(state, os.path.join(save_dir, "checkpoint.pt"))
        logger.info(f"Checkpoint serialised ({name})")

        if self.post_save_callback:
            try:
                self.post_save_callback(save_dir)
            except Exception as exc:
                logger.warning(f"Post-save callback failed: {exc}")
            finally:
                shutil.rmtree(save_dir, ignore_errors=True)
                logger.info("Temp checkpoint removed from disk.")
        else:
            logger.info(f"Checkpoint saved → {save_dir}")

    def _load_checkpoint(
        self,
        checkpoint_dir: str,
        optimizer,
        scheduler,
        scaler,
    ) -> int:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        state     = torch.load(ckpt_path, map_location=self.device)

        raw_model = (
            self.model.module if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        raw_model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if scaler is not None and "scaler_state" in state:
            scaler.load_state_dict(state["scaler_state"])

        return state["global_step"]


# ------------------------------------------------------------------ #
# Utility (mirrors BertPreTrainer._trim_to_length)
# ------------------------------------------------------------------ #

def _trim_to_length(batch: dict, max_len: int) -> dict:
    """Truncate all 2-D batch tensors to max_len along dim 1 (Phase 1 curriculum)."""
    return {
        k: v[:, :max_len] if isinstance(v, torch.Tensor) and v.dim() == 2 else v
        for k, v in batch.items()
    }
