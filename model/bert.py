"""
Core BertModel:  Embeddings → Encoder → Pooler.

BertPooler extracts the [CLS] token representation and passes it through a
dense + Tanh layer to produce a fixed-size sentence embedding used for NSP.

Weight initialisation follows the paper (§A): truncated normal with σ = 0.02,
bias = 0, LayerNorm weight = 1, LayerNorm bias = 0.

Reference: Section 3, Devlin et al. (2019).
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from model.embeddings import BertEmbeddings
from model.encoder    import BertEncoder


class BertPooler(nn.Module):
    """
    Produces a sentence-level representation from the [CLS] token.

    The paper uses:  pooled = Tanh(W · h_[CLS] + b)
    This representation is used for the NSP classification head.
    """

    def __init__(self, config):
        super().__init__()
        self.dense      = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        cls_token_state = sequence_output[:, 0]          # first token = [CLS]
        return self.activation(self.dense(cls_token_state))


class BertModel(nn.Module):
    """
    Bare BERT model (no pre-training heads).

    Input:   token ids, optional attention mask, optional segment ids
    Output:  sequence_output (all token hidden states),
             pooled_output   ([CLS] representation),
             all_attentions  (optional per-layer weights)
    """

    def __init__(self, config):
        super().__init__()
        self.config      = config
        self.embeddings  = BertEmbeddings(config)
        self.encoder     = BertEncoder(config)
        self.pooler      = BertPooler(config)

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        """Initialise all sub-modules with the scheme from Appendix A."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(
                    mean=0.0, std=self.config.initializer_range
                )
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(
                    mean=0.0, std=self.config.initializer_range
                )
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    # ------------------------------------------------------------------ #
    # Attention mask conversion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_extended_attention_mask(
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a binary mask (1 = keep, 0 = ignore) to an additive mask
        compatible with the attention scores:
            0       → kept position   (add 0 to logits)
            -10000  → ignored position (softmax drives weight to ~0)

        Shape:  (batch, seq) → (batch, 1, 1, seq)
        The singleton dimensions broadcast over (batch, heads, query_seq, key_seq).
        """
        extended = (1.0 - attention_mask[:, None, None, :].float()) * -10_000.0
        return extended

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            input_ids:       (batch, seq)
            attention_mask:  (batch, seq) — 1 for real tokens, 0 for padding
            token_type_ids:  (batch, seq) — 0 for sentence A, 1 for sentence B
            position_ids:    (batch, seq) — defaults to 0 … seq-1
            output_attentions: return per-layer attention weights

        Returns:
            sequence_output:  (batch, seq, hidden)
            pooled_output:    (batch, hidden)
            all_attentions:   list[Tensor] or None
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = self._make_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        sequence_output, all_attentions = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
        )

        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output, all_attentions
