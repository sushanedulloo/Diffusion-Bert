"""
Position-wise Feed-Forward Network (FFN) used inside each Transformer layer.

From Section 3.1 / Appendix A of Devlin et al. (2019):
    FFN(x) = GELU(x W_1 + b_1) W_2 + b_2

Dimensions:
    input/output : hidden_size       (768 for BERT-Base)
    inner        : intermediate_size (3072 = 4 × 768 for BERT-Base)

The module layout mirrors the Google BERT implementation:
  BertIntermediate  — first linear + GELU
  BertOutput        — second linear + residual + LayerNorm
"""

import math
import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
# Activation function
# ------------------------------------------------------------------ #

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Error Linear Unit as used in BERT.
    Approximation: x * 0.5 * (1 + erf(x / sqrt(2)))
    Reference: Hendrycks & Gimpel (2016), https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


_ACTIVATIONS = {
    "gelu": gelu,
    "relu": torch.relu,
}


# ------------------------------------------------------------------ #
# FFN sub-layers
# ------------------------------------------------------------------ #

class BertIntermediate(nn.Module):
    """
    First half of the FFN:
        hidden_size → intermediate_size, with GELU activation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act   = _ACTIVATIONS[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(hidden_states))


class BertOutput(nn.Module):
    """
    Second half of the FFN:
        intermediate_size → hidden_size, then residual + LayerNorm.
    """

    def __init__(self, config):
        super().__init__()
        self.dense     = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: output of BertIntermediate  (batch, seq, intermediate)
            input_tensor:  input to FFN (= attention output)  (batch, seq, hidden)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # residual
        return hidden_states
