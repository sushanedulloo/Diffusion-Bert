"""
Multi-Head Self-Attention as described in Section 3.1 of Devlin et al. (2019)
and "Attention Is All You Need" (Vaswani et al., 2017).

Each attention head computes:
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

All heads are concatenated and projected back to hidden_size:
    MultiHead(Q, K, V) = Concat(head_1, …, head_h) W^O

Module layout follows the original Google BERT implementation:
  BertAttention
    └── MultiHeadSelfAttention  (compute attention context)
    └── BertSelfOutput          (projection + residual + LayerNorm)
"""

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, config):
        super().__init__()
        self.num_heads  = config.num_attention_heads
        self.head_dim   = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        # Projection matrices for Q, K, V
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key   = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, seq, hidden) → (batch, heads, seq, head_dim).
        """
        B, S, _ = x.size()
        x = x.view(B, S, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)           # (B, H, S, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, heads, seq, head_dim) → (batch, seq, hidden).
        """
        B, _, S, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, S, H, d_k)
        return x.view(B, S, self.hidden_size)    # (B, S, hidden)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Args:
            hidden_states:  (batch, seq, hidden)
            attention_mask: (batch, 1, 1, seq) additive mask
                            (0 for valid positions, -10000 for padding/masked)

        Returns:
            context_layer:  (batch, seq, hidden)
            attention_probs: (batch, heads, seq, seq)
        """
        Q = self._split_heads(self.query(hidden_states))   # (B, H, S, d_k)
        K = self._split_heads(self.key(hidden_states))
        V = self._split_heads(self.value(hidden_states))

        # Scaled dot-product attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, S, S)

        if attention_mask is not None:
            scores = scores + attention_mask

        attention_probs = torch.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Context vector
        context = torch.matmul(attention_probs, V)  # (B, H, S, d_k)
        context = self._merge_heads(context)         # (B, S, hidden)

        return context, attention_probs


class BertSelfOutput(nn.Module):
    """
    Linear projection applied to the attention context, followed by
    a residual connection and LayerNorm (post-norm, as in the paper).
    """

    def __init__(self, config):
        super().__init__()
        self.dense     = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # residual
        return hidden_states


class BertAttention(nn.Module):
    """
    Full attention sub-layer:
        x → MultiHeadSelfAttention → BertSelfOutput(+residual+LN) → y
    """

    def __init__(self, config):
        super().__init__()
        self.self   = MultiHeadSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Returns:
            attention_output: (batch, seq, hidden)
            attention_probs:  (batch, heads, seq, seq)
        """
        self_output, attention_probs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output, attention_probs
