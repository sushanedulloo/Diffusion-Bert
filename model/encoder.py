"""
BERT Transformer Encoder.

Stacks N identical BertLayer modules (N = num_hidden_layers).
Each layer applies:
    1. Multi-head self-attention  (BertAttention)
    2. Position-wise FFN          (BertIntermediate + BertOutput)

Both sub-layers use a residual connection followed by LayerNorm (post-norm).

Reference: Section 3.1, Devlin et al. (2019).
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from model.attention   import BertAttention
from model.feed_forward import BertIntermediate, BertOutput


class BertLayer(nn.Module):
    """
    A single Transformer encoder layer:
        x → Attention(x) + x → LN → FFN(·) + · → LN → output
    """

    def __init__(self, config):
        super().__init__()
        self.attention    = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output       = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:  (batch, seq, hidden)
            attention_mask: (batch, 1, 1, seq) additive mask

        Returns:
            layer_output:    (batch, seq, hidden)
            attention_probs: (batch, heads, seq, seq)
        """
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertEncoder(nn.Module):
    """
    Stack of BertLayer modules.
    Optionally returns all intermediate attention weight matrices.
    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            hidden_states:     (batch, seq, hidden)
            attention_mask:    (batch, 1, 1, seq) additive mask
            output_attentions: if True, collect per-layer attention weights

        Returns:
            hidden_states:  final hidden states (batch, seq, hidden)
            all_attentions: list of (batch, heads, seq, seq) tensors per layer,
                            or None when output_attentions=False
        """
        all_attentions: Optional[List[torch.Tensor]] = [] if output_attentions else None

        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            if output_attentions:
                all_attentions.append(attention_probs)

        return hidden_states, all_attentions
