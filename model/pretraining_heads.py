"""
Pre-training head for BERT.

BERT is pre-trained on Masked Language Modeling (MLM) (Section 3.3):
  15 % of WordPiece tokens are selected; of those:
    - 80 % are replaced with [MASK]
    - 10 % are replaced with a random token
    - 10 % are kept unchanged
  The head predicts the original token at each masked position.

NSP is not used — this model is pre-trained with MDLM (diffusion).

Weight tying: the MLM decoder weight matrix is shared with the input
token embedding table (reduces parameter count and speeds convergence).

Reference: Sections 3.3 and 4, Devlin et al. (2019).
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional

from model.bert import BertModel


# ------------------------------------------------------------------ #
# Helper: GELU (duplicated here to keep this file self-contained)
# ------------------------------------------------------------------ #

def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ------------------------------------------------------------------ #
# MLM head
# ------------------------------------------------------------------ #

class BertPredictionHeadTransform(nn.Module):
    """
    Transform applied before the final vocab projection in the MLM head:
        hidden → Linear → GELU → LayerNorm
    """

    def __init__(self, config):
        super().__init__()
        self.dense     = nn.Linear(config.hidden_size, config.hidden_size)
        self.act       = gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.act(self.dense(hidden_states)))


class BertLMPredictionHead(nn.Module):
    """
    MLM output head:
        token hidden states → transform → linear(vocab_size) + bias → logits
    """

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The weight matrix is tied to the input word embedding (set externally).
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Separate trainable bias (the embedding weight has no bias column)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.transform(hidden_states))  # (batch, seq, vocab)


# ------------------------------------------------------------------ #
# Combined pre-training model
# ------------------------------------------------------------------ #

class BertForPreTraining(nn.Module):
    """
    BertModel + MLM head.

    The forward signature accepts the exact keys produced by the data collator:
        input_ids, attention_mask, token_type_ids,
        labels (MLM targets, -100 = not masked)

    Returns a dict with keys:
        loss       — scalar MLM loss, or None if labels absent
        mlm_logits — (batch, seq, vocab)
    """

    def __init__(self, config, bert: BertModel):
        super().__init__()
        self.bert     = bert
        self.mlm_head = BertLMPredictionHead(config)

        # Tie MLM decoder weights to input word embeddings
        self.mlm_head.decoder.weight = self.bert.embeddings.word_embeddings.weight

        self._init_heads(config)

    def _init_heads(self, config):
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
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,        # MLM targets
        **kwargs,                               # absorbs extra collator keys
    ) -> Dict[str, Optional[torch.Tensor]]:

        sequence_output, _, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        mlm_logits = self.mlm_head(sequence_output)   # (B, S, V)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                mlm_logits.reshape(-1, mlm_logits.size(-1)),
                labels.reshape(-1),
            )

        return {
            "loss":       loss,
            "mlm_logits": mlm_logits,
        }
