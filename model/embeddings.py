"""
BERT Embedding layer.

Combines three learned embedding tables:
  1. Token embeddings   — one vector per WordPiece token
  2. Position embeddings — one vector per position index (0 … max_pos-1)
  3. Segment embeddings  — one vector for segment A (0) and one for B (1)

The three embeddings are summed, then passed through LayerNorm + Dropout.

Reference: Section 3.1 of Devlin et al. (2019).
"""

import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Token (word-piece) embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # Absolute position embeddings (learned, not sinusoidal)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Segment (token-type) embeddings: 0 = sentence A, 1 = sentence B
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Pre-built position id buffer — avoids re-allocating every forward pass
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),  # (1, max_pos)
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:       (batch, seq_len)  — WordPiece token indices
            token_type_ids:  (batch, seq_len)  — 0 for sentence A, 1 for B
            position_ids:    (batch, seq_len)  — defaults to [0, 1, …, seq_len-1]

        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        seq_len = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]  # (1, seq_len)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeds     = self.word_embeddings(input_ids)          # (B, S, H)
        position_embeds = self.position_embeddings(position_ids)   # (1, S, H)
        segment_embeds  = self.token_type_embeddings(token_type_ids)  # (B, S, H)

        embeddings = word_embeds + position_embeds + segment_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
