"""
BERT Pre-training Trainer.

Implements the two-phase training schedule described in Appendix A of
Devlin et al. (2019):

  Phase 1 (90 % of steps): max_seq_length = 128
    Trains on shorter sequences — much faster per step and handles the
    majority of the token-level language model signal.

  Phase 2 (10 % of steps): max_seq_length = 512
    Fine-tunes the model on full-length sequences so it learns
    long-range dependencies and position embeddings up to index 511.

Both phases use the *same* dataset (pre-built with max_seq_length=512).
In Phase 1 each batch is truncated on-the-fly to 128 tokens — this avoids
building two separate datasets and saves disk / memory.

Features
--------
* Mixed-precision training via torch.amp (optional, requires CUDA)
* Multi-GPU via nn.DataParallel (optional, automatic if >1 GPU)
* Gradient accumulation for simulating large batches on limited hardware
* Periodic checkpoint saving (state dict only, portable)
* Structured logging of loss, learning rate, throughput
"""

import os
import time
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from training.optimizer import get_bert_optimizer
from training.scheduler import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class BertPreTrainer:
    """
    Manages the BERT pre-training loop.

    Args:
        model:           BertForPreTraining instance
        config:          BertConfig (used for default hyper-parameters)
        train_dataset:   BertPretrainingDataset
        collator:        DataCollatorForLanguageModeling
        output_dir:      directory for saving checkpoints
        use_amp:         enable automatic mixed precision (FP16 on CUDA)
    """

    def __init__(
        self,
        model,
        config,
        train_dataset,
        collator,
        output_dir: str = "./output",
        use_amp: bool = True,
    ):
        self.config       = config
        self.train_dataset = train_dataset
        self.collator     = collator
        self.output_dir   = output_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training device: {self.device}")

        # Wrap with DataParallel when multiple GPUs are available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
            self.model = nn.DataParallel(model)
        else:
            self.model = model

        self.model.to(self.device)

        # Mixed precision only makes sense on CUDA
        self.use_amp = use_amp and self.device.type == "cuda"
        if self.use_amp:
            logger.info("Mixed-precision training (AMP) enabled.")

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #

    def train(
        self,
        num_train_steps: int        = 1_000_000,
        batch_size: int             = 256,
        gradient_accumulation_steps: int = 1,
        learning_rate: float        = 1e-4,
        warmup_steps: int           = 10_000,
        weight_decay: float         = 0.01,
        max_grad_norm: float        = 1.0,
        save_steps: int             = 10_000,
        logging_steps: int          = 100,
        phase1_ratio: float         = 0.9,
        phase1_max_seq_len: int     = 128,
        phase2_max_seq_len: int     = 512,
        dataloader_num_workers: int = 4,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """
        Run the BERT pre-training loop.

        Args:
            num_train_steps:             total optimiser update steps (1 M per paper)
            batch_size:                  sequences per step (256 per paper)
            gradient_accumulation_steps: accumulate gradients before each update;
                                         use this to simulate larger batches when
                                         GPU memory is limited (e.g. set to 4 to
                                         simulate batch_size × 4 with a smaller GPU)
            learning_rate:               peak learning rate (1e-4 per paper)
            warmup_steps:                linear warmup steps (10,000 per paper)
            weight_decay:                L2 coefficient (0.01 per paper)
            max_grad_norm:               gradient clipping threshold (1.0)
            save_steps:                  save checkpoint every N *update* steps
            logging_steps:               log metrics every N *update* steps
            phase1_ratio:                fraction of steps using seq_len=128 (0.9)
            phase1_max_seq_len:          max sequence length for Phase 1
            phase2_max_seq_len:          max sequence length for Phase 2
            dataloader_num_workers:      DataLoader worker processes
            resume_from_checkpoint:      path to a checkpoint directory to resume from
        """
        # ---- DataLoader ------------------------------------------------ #
        # `batch_size` here is per-step; if using DataParallel divide by GPU count
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

        # ---- Optimiser & scheduler ------------------------------------- #
        optimizer = get_bert_optimizer(
            self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_steps,
        )
        scaler = GradScaler() if self.use_amp else None

        # ---- Optionally resume ----------------------------------------- #
        global_step = 0
        if resume_from_checkpoint:
            global_step = self._load_checkpoint(
                resume_from_checkpoint, optimizer, scheduler, scaler
            )
            logger.info(f"Resumed from step {global_step}.")

        phase1_steps = int(num_train_steps * phase1_ratio)

        logger.info("=" * 60)
        logger.info("BERT Pre-Training")
        logger.info(f"  Total steps          : {num_train_steps:,}")
        logger.info(f"  Batch size           : {batch_size}")
        logger.info(f"  Grad accumulation    : {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size : {batch_size * gradient_accumulation_steps}")
        logger.info(f"  Learning rate        : {learning_rate}")
        logger.info(f"  Warmup steps         : {warmup_steps}")
        logger.info(f"  Phase 1 steps        : {phase1_steps} (seq_len={phase1_max_seq_len})")
        logger.info(f"  Phase 2 steps        : {num_train_steps - phase1_steps} (seq_len={phase2_max_seq_len})")
        logger.info("=" * 60)

        # ---- Training loop --------------------------------------------- #
        self.model.train()
        optimizer.zero_grad()

        running_loss = 0.0
        t0 = time.time()

        while global_step < num_train_steps:
            for batch in train_loader:
                if global_step >= num_train_steps:
                    break

                # Determine current sequence length based on training phase
                current_max_len = (
                    phase1_max_seq_len if global_step < phase1_steps
                    else phase2_max_seq_len
                )

                batch = self._to_device(batch)
                batch = self._trim_to_length(batch, current_max_len)

                # ---- Forward + backward -------------------------------- #
                if self.use_amp:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs["loss"]
                        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                            loss = loss.mean()           # DataParallel reduces
                        loss = loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                running_loss += loss.item() * gradient_accumulation_steps

                # ---- Gradient update every `gradient_accumulation_steps` #
                micro_step = global_step * gradient_accumulation_steps
                if (micro_step + 1) % gradient_accumulation_steps == 0:
                    if self.use_amp:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # ---- Logging --------------------------------------- #
                    if global_step % logging_steps == 0:
                        avg_loss = running_loss / logging_steps
                        elapsed  = time.time() - t0
                        steps_s  = logging_steps / elapsed
                        current_lr = scheduler.get_last_lr()[0]

                        logger.info(
                            f"Step {global_step:>7,} / {num_train_steps:,} | "
                            f"loss={avg_loss:.4f} | "
                            f"lr={current_lr:.2e} | "
                            f"seq_len={current_max_len} | "
                            f"{steps_s:.2f} steps/s"
                        )
                        running_loss = 0.0
                        t0 = time.time()

                    # ---- Checkpoint ------------------------------------ #
                    if global_step % save_steps == 0:
                        self._save_checkpoint(global_step, optimizer, scheduler, scaler)

        # ---- Final save ------------------------------------------------ #
        self._save_checkpoint(global_step, optimizer, scheduler, scaler, tag="final")
        logger.info("Pre-training complete.")

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @staticmethod
    def _trim_to_length(batch: dict, max_len: int) -> dict:
        """
        Trim all 2-D tensors in the batch to `max_len` along dim=1.
        This implements the phase-based sequence-length curriculum cheaply.
        """
        return {
            k: v[:, :max_len] if isinstance(v, torch.Tensor) and v.dim() == 2 else v
            for k, v in batch.items()
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(
        self,
        step: int,
        optimizer,
        scheduler,
        scaler,
        tag: Optional[str] = None,
    ) -> None:
        name = tag if tag else f"step_{step:07d}"
        save_dir = os.path.join(self.output_dir, f"checkpoint_{name}")
        os.makedirs(save_dir, exist_ok=True)

        # Unwrap DataParallel
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
        logger.info(f"Checkpoint saved → {save_dir}")

    def _load_checkpoint(
        self,
        checkpoint_dir: str,
        optimizer,
        scheduler,
        scaler,
    ) -> int:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        state = torch.load(ckpt_path, map_location=self.device)

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
