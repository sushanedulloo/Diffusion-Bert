"""
Adam optimiser configuration for BERT pre-training.

From Appendix A of Devlin et al. (2019):
  "We use Adam with learning rate 1e-4, β₁ = 0.9, β₂ = 0.999,
   L2 weight decay of 0.01, learning rate warmup over the first
   10,000 steps, and linear decay of the learning rate."

Weight decay is applied to all parameters *except*:
  - bias terms
  - LayerNorm.weight  (scale parameter)
  - LayerNorm.bias    (shift parameter)

This matches the practice in the original implementation and prevents
LayerNorm from being under-regularised.
"""

from torch.optim import Adam
from typing import List


_NO_DECAY_PARAMS = ["bias", "LayerNorm.weight", "LayerNorm.bias"]


def get_bert_optimizer(
    model,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-6,
) -> Adam:
    """
    Build an Adam optimiser with layer-specific weight decay.

    Args:
        model:         the BertForPreTraining model
        learning_rate: initial learning rate (1e-4 per paper)
        weight_decay:  L2 regularisation coefficient (0.01 per paper)
        beta1:         Adam β₁ (0.9 per paper)
        beta2:         Adam β₂ (0.999 per paper)
        epsilon:       Adam ε  (1e-6 per paper; some implementations use 1e-8)

    Returns:
        Adam optimiser with two parameter groups
    """
    decay_params:    List = []
    no_decay_params: List = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in _NO_DECAY_PARAMS):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return Adam(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
    )
