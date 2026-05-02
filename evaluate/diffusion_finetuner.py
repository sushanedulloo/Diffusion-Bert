"""
Instruction fine-tuning loop for BertForDiffusion (MDLM objective).

DiffusionFineTuner — trains a BertForDiffusion on instruction-formatted tasks
    using the MDLM loss restricted to answer (target) positions only.

The key idea: condition positions have labels=-100, so the MDLM masking
logic naturally skips them.  Only the answer [MASK] positions are noised
and trained, matching the reverse diffusion inference procedure exactly.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.diffusion_heads    import BertForDiffusion
from evaluate.diffusion_utils import mdlm_sample_conditional
from evaluate.instruction_templates import decode_answer, match_label

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Training arguments
# ------------------------------------------------------------------ #

@dataclass
class FineTuneArgs:
    batch_size:    int   = 32
    num_epochs:    int   = 3
    learning_rate: float = 2e-5
    use_amp:       bool  = True
    time_epsilon:  float = 1e-3
    num_workers:   int   = 0
    weight_decay:  float = 0.01


# ------------------------------------------------------------------ #
# Fine-tuner
# ------------------------------------------------------------------ #

class DiffusionFineTuner:
    """
    Instruction fine-tuning loop for BertForDiffusion.

    Training dataset must return dicts with keys:
        input_ids      (L,) LongTensor
        condition_mask (L,) BoolTensor  — True = condition, False = target
        attention_mask (L,) LongTensor
        labels         (L,) LongTensor  — -100 at condition/pad, true token at target
        raw_label      scalar float/int — original task label (for metric)
    """

    def __init__(
        self,
        model:         BertForDiffusion,
        train_dataset: Dataset,
        eval_dataset:  Dataset,
        scheduler,                      # BaseAlphaScheduler
        args:          FineTuneArgs,
        tokenizer,
        device:        torch.device,
    ):
        self.model          = model
        self.train_dataset  = train_dataset
        self.eval_dataset   = eval_dataset
        self.scheduler      = scheduler
        self.args           = args
        self.tokenizer      = tokenizer
        self.device         = device

    # -------------------------------------------------------------- #
    # MDLM loss
    # -------------------------------------------------------------- #

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        MDLM loss restricted to target (answer) positions.

        Steps:
          1. Sample timestep t ~ Uniform[time_epsilon, 1) per example
          2. Compute masking probability p_mask = 1 - alpha(t)
          3. Randomly mask targetable positions with probability p_mask
          4. Forward pass on noised sequence
          5. Weighted cross-entropy at masked positions only
        """
        B, L = batch["input_ids"].shape

        # Sample continuous timestep per example
        eps = self.args.time_epsilon
        t   = torch.rand(B, device=self.device) * (1.0 - eps) + eps  # (B,)

        # Compute alpha and masking probability per example (tensor path)
        alpha_t = self.scheduler.alpha(t)   # (B,) tensor
        p_mask  = 1.0 - alpha_t             # (B,)

        # Maskable = label positions (condition positions already have label=-100)
        maskable = (batch["labels"] != -100)   # (B, L)

        rand    = torch.rand(B, L, device=self.device)
        do_mask = (rand < p_mask.unsqueeze(1)) & maskable   # (B, L)

        noised_ids          = batch["input_ids"].clone()
        noised_ids[do_mask] = self.tokenizer.mask_token_id

        logits = self.model(noised_ids, batch["attention_mask"])["logits"]  # (B, L, V)
        V      = logits.size(-1)

        token_nll = F.cross_entropy(
            logits.view(-1, V),
            batch["labels"].view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, L)                                         # (B, L)

        # MDLM loss weight w(t) = -dalpha/dt / (1 - alpha(t) + eps)
        w = self.scheduler.weight(t)   # (B,) tensor

        n_masked = do_mask.sum().clamp(min=1)
        loss     = (token_nll * w.unsqueeze(1) * do_mask.float()).sum() / n_masked
        return loss

    # -------------------------------------------------------------- #
    # Training loop
    # -------------------------------------------------------------- #

    def train(self) -> None:
        """Run MDLM instruction fine-tuning."""
        self.model.to(self.device)

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

        total_steps  = len(loader) * self.args.num_epochs
        warmup_steps = max(1, int(0.1 * total_steps))

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)

        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        use_amp = self.args.use_amp and self.device.type == "cuda"
        scaler  = torch.cuda.amp.GradScaler() if use_amp else None

        global_step = 0
        self.model.train()

        for epoch in range(self.args.num_epochs):
            epoch_loss = 0.0

            for batch in loader:
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = self._compute_loss(batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = self._compute_loss(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                lr_sched.step()
                epoch_loss  += loss.item()
                global_step += 1

            avg = epoch_loss / max(1, len(loader))
            logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs} — avg loss: {avg:.4f}")

        self.model.eval()

    # -------------------------------------------------------------- #
    # Inference
    # -------------------------------------------------------------- #

    def generate_predictions(
        self,
        dataset:     Dataset,
        batch_size:  int   = 32,
        num_steps:   int   = 200,
        temperature: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """
        Run conditional MDLM sampling on `dataset`.

        Returns:
            List of (decoded_answer_string, raw_label) per example.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        results: List[Tuple[str, float]] = []

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)       # (B, L)
            condition_mask = batch["condition_mask"].to(self.device)   # (B, L)
            attention_mask = batch["attention_mask"].to(self.device)   # (B, L)
            raw_labels     = batch["raw_label"]                        # (B,)

            generated = mdlm_sample_conditional(
                model=self.model,
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                condition_mask=condition_mask,
                scheduler=self.scheduler,
                attention_mask=attention_mask,
                num_steps=num_steps,
                temperature=temperature,
                device=self.device,
            )

            for i in range(generated.size(0)):
                decoded = decode_answer(generated[i], condition_mask[i], self.tokenizer)
                results.append((decoded, float(raw_labels[i].item())))

        return results

    # -------------------------------------------------------------- #
    # Evaluation
    # -------------------------------------------------------------- #

    def evaluate(
        self,
        label_words:    Dict[int, str],
        metric_fn:      Callable,
        batch_size:     int   = 32,
        num_steps:      int   = 200,
        temperature:    float = 1.0,
        is_regression:  bool  = False,
    ) -> Dict[str, float]:
        """
        Generate predictions on eval_dataset, decode labels, compute metric.

        Args:
            label_words:   {label_id: word} for classification or digit words for regression
            metric_fn:     callable(preds, labels) -> Dict[str, float]
            is_regression: True for STS-B (returns float predictions)

        Returns:
            metric dict from metric_fn
        """
        predictions = self.generate_predictions(
            self.eval_dataset,
            batch_size=batch_size,
            num_steps=num_steps,
            temperature=temperature,
        )

        pred_list  = []
        label_list = []

        for decoded, raw_label in predictions:
            label_int = match_label(decoded, label_words)
            if is_regression:
                pred_list.append(float(label_int))
            else:
                pred_list.append(label_int)
            label_list.append(raw_label if is_regression else int(raw_label))

        return metric_fn(pred_list, label_list)
