"""
Generic fine-tuning trainer for BERT downstream tasks.

FineTuningTrainer wraps a model, optimizer, and data loaders into a
standard training loop that supports:
  - linear warmup + linear decay LR schedule
  - gradient accumulation
  - mixed-precision (AMP) training
  - per-epoch evaluation with best-model checkpointing
  - structured logging

TrainingArgs is a simple dataclass that mirrors the most common
fine-tuning hyperparameters from the BERT paper (Section 4):
  - learning rate: 2e-5 to 5e-5
  - epochs: 2–4
  - batch size: 16–32
"""

import os
import math
import time
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from training.optimizer  import get_bert_optimizer
from training.scheduler  import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    """Hyperparameters for downstream fine-tuning."""
    output_dir:                   str
    num_epochs:                   int   = 3
    batch_size:                   int   = 32
    eval_batch_size:              int   = 64
    learning_rate:                float = 2e-5
    warmup_ratio:                 float = 0.1          # fraction of steps used for warmup
    weight_decay:                 float = 0.01
    max_grad_norm:                float = 1.0
    gradient_accumulation_steps:  int   = 1
    seed:                         int   = 42
    use_amp:                      bool  = True
    logging_steps:                int   = 50
    save_best:                    bool  = True          # save checkpoint with best eval metric
    metric_for_best:              str   = "loss"        # key in eval metrics dict; "loss" for lowest
    greater_is_better:            bool  = False         # True for accuracy/F1, False for loss


class FineTuningTrainer:
    """
    Task-agnostic fine-tuning trainer.

    Args:
        model:           any nn.Module whose forward() returns a dict
                         with at least the key "loss"
        train_dataset:   PyTorch Dataset
        eval_dataset:    PyTorch Dataset (may be None to skip evaluation)
        collate_fn:      optional DataLoader collate function
        compute_metrics: callable(eval_output_dict) → metrics_dict
                         called after each evaluation epoch; receives the
                         accumulated {"preds": ..., "labels": ...} dict
        args:            TrainingArgs
    """

    def __init__(
        self,
        model:           nn.Module,
        train_dataset:   Dataset,
        eval_dataset:    Optional[Dataset],
        args:            TrainingArgs,
        collate_fn:      Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        torch.manual_seed(args.seed)
        self.args            = args
        self.compute_metrics = compute_metrics

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
            self.model = nn.DataParallel(self.model)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        self.eval_loader = (
            DataLoader(
                eval_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            if eval_dataset is not None else None
        )

        # Optimizer and scheduler
        num_training_steps = (
            math.ceil(len(self.train_loader) / args.gradient_accumulation_steps)
            * args.num_epochs
        )
        num_warmup_steps = int(num_training_steps * args.warmup_ratio)

        self.optimizer = get_bert_optimizer(
            self.model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )

        # AMP scaler (CUDA only)
        self.scaler: Optional[GradScaler] = (
            GradScaler() if args.use_amp and torch.cuda.is_available() else None
        )

        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(self) -> Dict[str, List[float]]:
        """
        Run fine-tuning for num_epochs.

        Returns history dict {"train_loss": [...], "eval_<metric>": [...]}.
        """
        args    = self.args
        history = {"train_loss": []}

        best_score      = float("inf") if not args.greater_is_better else float("-inf")
        best_model_path = os.path.join(args.output_dir, "best_model.pt")

        global_step = 0
        for epoch in range(1, args.num_epochs + 1):
            self.model.train()
            epoch_loss     = 0.0
            optimizer_steps = 0
            t0 = time.time()

            self.optimizer.zero_grad()
            for step, batch in enumerate(self.train_loader, 1):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(**batch)
                    loss = outputs["loss"]
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                    loss = loss / args.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()

                epoch_loss += loss.item() * args.gradient_accumulation_steps

                if step % args.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    optimizer_steps += 1

                    if global_step % args.logging_steps == 0:
                        avg_loss = epoch_loss / step
                        lr       = self.scheduler.get_last_lr()[0]
                        elapsed  = time.time() - t0
                        logger.info(
                            f"Epoch {epoch} | step {global_step} | "
                            f"loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s"
                        )

            avg_train_loss = epoch_loss / len(self.train_loader)
            history["train_loss"].append(avg_train_loss)
            logger.info(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}")

            # Evaluate
            if self.eval_loader is not None:
                eval_metrics = self.evaluate()
                for k, v in eval_metrics.items():
                    history.setdefault(f"eval_{k}", []).append(v)
                    logger.info(f"[Epoch {epoch}] eval_{k}={v:.4f}")

                # Save best model
                if args.save_best:
                    score = eval_metrics.get(args.metric_for_best, avg_train_loss)
                    improved = (
                        score < best_score if not args.greater_is_better else score > best_score
                    )
                    if improved:
                        best_score = score
                        raw = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                        torch.save(raw.state_dict(), best_model_path)
                        logger.info(f"  → Best model saved (score={best_score:.4f})")

        logger.info("Training complete.")
        return history

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #

    def evaluate(self) -> Dict[str, float]:
        """
        Run one evaluation pass; return computed metrics.

        Calls self.compute_metrics({"preds": ..., "labels": ..., "logits": ...})
        if provided, otherwise returns {"loss": avg_eval_loss}.
        """
        assert self.eval_loader is not None, "No eval dataset provided"
        self.model.eval()

        total_loss = 0.0
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.eval_loader:
                batch   = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                if outputs.get("loss") is not None:
                    loss = outputs["loss"]
                    if loss.dim() > 0:
                        loss = loss.mean()
                    total_loss += loss.item()

                if "logits" in outputs:
                    all_logits.append(outputs["logits"].detach().cpu())
                if "labels" in batch:
                    all_labels.append(batch["labels"].detach().cpu())

        metrics: Dict[str, float] = {}
        if self.eval_loader and total_loss > 0:
            metrics["loss"] = total_loss / len(self.eval_loader)

        if self.compute_metrics is not None and all_logits:
            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0) if all_labels else None
            extra  = self.compute_metrics({"logits": logits, "labels": labels})
            metrics.update(extra)

        return metrics

    # ------------------------------------------------------------------ #
    # Prediction (inference)
    # ------------------------------------------------------------------ #

    def predict(self, dataset: Dataset, collate_fn: Optional[Callable] = None) -> Dict:
        """Run inference on a dataset; return raw logits and any additional outputs."""
        loader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        self.model.eval()
        all_outputs: Dict[str, List[torch.Tensor]] = {}

        with torch.no_grad():
            for batch in loader:
                batch   = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        all_outputs.setdefault(k, []).append(v.detach().cpu())

        return {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}
