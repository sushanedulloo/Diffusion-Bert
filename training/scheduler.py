"""
Learning-rate scheduler for BERT pre-training.

From Appendix A of Devlin et al. (2019):
  "learning rate warmup over the first 10,000 steps,
   and linear decay of the learning rate."

Schedule (per step t):
  t < warmup  →  lr = base_lr × (t / warmup)
  t ≥ warmup  →  lr = base_lr × (total - t) / (total - warmup)

This decays linearly to 0 at t = total_training_steps.
"""

from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Linear warmup + linear decay scheduler.

    Args:
        optimizer:           the Adam optimiser returned by get_bert_optimizer
        num_warmup_steps:    number of warmup steps  (10,000 per paper)
        num_training_steps:  total training steps    (1,000,000 per paper)

    Returns:
        LambdaLR scheduler — call scheduler.step() after each optimiser step
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear ramp-up
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay to 0
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)
