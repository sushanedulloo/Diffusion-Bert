"""
BertForDiffusion — BERT encoder + MLM head for MDLM pre-training.

Differences from BertForPreTraining:
  - No NSP head (MDLM has a single denoising objective)
  - No loss computation inside the model; the trainer (MDLMPreTrainer)
    computes the timestep-weighted cross-entropy loss externally
  - forward() returns {"logits": (B, S, V)}

The MLM head (BertLMPredictionHead) and weight-tying logic are reused
directly from model/pretraining_heads.py.

Reference: Section 3 of Shi et al. (2024) — https://arxiv.org/abs/2406.07524
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from model.bert import BertModel
from model.pretraining_heads import BertLMPredictionHead
from config import BertConfig


class BertForDiffusion(nn.Module):
    """
    BERT encoder with an MLM prediction head, configured for MDLM pre-training.

    The model takes a *noised* (partially masked) token sequence as input and
    predicts the original (clean) token at every position.  The MDLMPreTrainer
    applies the stochastic masking and loss weighting externally.

    Args:
        config: BertConfig instance
        bert:   pre-constructed BertModel (allows weight sharing with other
                models or loading a pre-trained backbone)

    Forward inputs:
        input_ids:      (B, S) — noised token ids ([MASK] inserted by trainer)
        attention_mask: (B, S) — 1 for real tokens, 0 for padding
        token_type_ids: (B, S) — optional segment ids (0/1)

    Forward output:
        {"logits": (B, S, vocab_size)}
    """

    def __init__(self, config: BertConfig, bert: BertModel):
        super().__init__()
        self.config   = config
        self.bert     = bert
        self.mlm_head = BertLMPredictionHead(config)

        # Tie MLM decoder weights to the input word embedding table.
        # Matches the weight-tying in BertForPreTraining.
        self.mlm_head.decoder.weight = self.bert.embeddings.word_embeddings.weight

        self._init_mlm_head(config)

    def _init_mlm_head(self, config: BertConfig) -> None:
        """Initialise the non-tied MLM head parameters."""
        for m in self.mlm_head.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids:      torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,                                   # absorbs unexpected keys
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (B, S) — noised sequence (some tokens replaced with [MASK])
            attention_mask: (B, S) — 1 = real token, 0 = padding
            token_type_ids: (B, S) — segment ids (optional; defaults to all-zeros)

        Returns:
            {"logits": (B, S, vocab_size)}
        """
        sequence_output, _, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.mlm_head(sequence_output)   # (B, S, V)
        return {"logits": logits}
