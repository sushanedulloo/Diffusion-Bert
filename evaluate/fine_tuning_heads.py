"""
Task-specific fine-tuning heads for BERT downstream evaluation.

Wraps BertModel with the four head types needed across all benchmarks:

  BertForSequenceClassification  — GLUE, XNLI, MLDoc
  BertForTokenClassification     — NER, cross-lingual NER, POS tagging
  BertForQuestionAnswering       — SQuAD v1.1 and v2.0
  BertForMultipleChoice          — SWAG
  BertForDependencyParsing       — Universal Dependencies (biaffine parser)

All models follow the fine-tuning recipe from Section 4 of Devlin et al.
(2019): replace the pre-training head with a task head, fine-tune all
parameters end-to-end.

Checkpoint loading
------------------
load_bert_from_checkpoint() extracts the BertModel backbone from a
BertForPreTraining checkpoint saved by training/trainer.py.
"""

import torch
import torch.nn as nn
from dataclasses import fields as dc_fields
from typing import Dict, Optional, Tuple

from model.bert import BertModel
from config import BertConfig


# ------------------------------------------------------------------ #
# Checkpoint loading
# ------------------------------------------------------------------ #

def load_bert_from_checkpoint(
    checkpoint_path: str,
    config: Optional[BertConfig] = None,
) -> Tuple[BertModel, BertConfig]:
    """
    Load BertModel weights from a BertForPreTraining checkpoint.

    The pre-trainer saves the full BertForPreTraining state dict, which
    has keys like "bert.embeddings.*", "mlm_head.*", "nsp_head.*".
    This function strips the pre-training heads and loads only the
    encoder backbone.

    Args:
        checkpoint_path: path to checkpoint.pt produced by BertPreTrainer
        config:          BertConfig; if None, uses default BertConfig

    Returns:
        (BertModel, BertConfig)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    if config is None:
        config = BertConfig()

    # Keep only the encoder backbone keys; strip the "bert." prefix
    _skip = ("mlm_head.", "nsp_head.")
    bert_state: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("bert."):
            bert_state[k[5:]] = v          # "bert.embeddings.X" → "embeddings.X"
        elif not any(k.startswith(p) for p in _skip):
            bert_state[k] = v

    bert = BertModel(config)
    missing, unexpected = bert.load_state_dict(bert_state, strict=False)
    if missing:
        print(f"[load_bert] Missing keys ({len(missing)}): {missing[:3]}{'…' if len(missing)>3 else ''}")
    return bert, config


# ------------------------------------------------------------------ #
# Sequence Classification (GLUE, XNLI, MLDoc)
# ------------------------------------------------------------------ #

class BertForSequenceClassification(nn.Module):
    """
    BERT + linear classifier on the pooled [CLS] output.

    For regression tasks (STS-B): set num_labels=1; MSE loss is used.
    For classification tasks:     num_labels ≥ 2; cross-entropy is used.

    Used for: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI,
              XNLI, MLDoc.
    """

    def __init__(self, config: BertConfig, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.bert       = BertModel(config)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self._init(config)

    def _init(self, config: BertConfig) -> None:
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids:      torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels:         Optional[torch.Tensor]     = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        labels: (batch,) long for classification or float for regression
        """
        _, pooled, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            if self.num_labels == 1:          # regression
                loss = nn.MSELoss()(logits.squeeze(-1), labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits, labels.long())

        return {"loss": loss, "logits": logits}


# ------------------------------------------------------------------ #
# Token Classification (NER, POS tagging)
# ------------------------------------------------------------------ #

class BertForTokenClassification(nn.Module):
    """
    BERT + per-token linear classifier.

    Positions labeled -100 are excluded from the loss (used for
    padding tokens and non-first wordpieces in NER/POS alignment).

    Used for: CoNLL-2003 NER, cross-lingual NER (WikiANN), POS tagging.
    """

    def __init__(self, config: BertConfig, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.bert       = BertModel(config)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self._init(config)

    def _init(self, config: BertConfig) -> None:
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids:      torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels:         Optional[torch.LongTensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        seq_out, _, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.classifier(self.dropout(seq_out))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return {"loss": loss, "logits": logits}


# ------------------------------------------------------------------ #
# Question Answering (SQuAD v1.1 and v2.0)
# ------------------------------------------------------------------ #

class BertForQuestionAnswering(nn.Module):
    """
    BERT + linear layer predicting start and end token positions.

    For SQuAD v2.0 (has_answer_head=True), an additional binary
    classifier on the [CLS] pooled output predicts whether the passage
    contains an answer at all.

    Used for: SQuAD v1.1, SQuAD v2.0.
    """

    def __init__(self, config: BertConfig, has_answer_head: bool = False):
        super().__init__()
        self.has_answer_head = has_answer_head
        self.bert            = BertModel(config)
        self.qa_outputs      = nn.Linear(config.hidden_size, 2)
        if has_answer_head:
            self.has_answer = nn.Linear(config.hidden_size, 2)
        self._init(config)

    def _init(self, config: BertConfig) -> None:
        for layer in [self.qa_outputs] + ([self.has_answer] if self.has_answer_head else []):
            layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
            layer.bias.data.zero_()

    def forward(
        self,
        input_ids:       torch.LongTensor,
        attention_mask:  Optional[torch.LongTensor] = None,
        token_type_ids:  Optional[torch.LongTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions:   Optional[torch.LongTensor] = None,
        is_impossible:   Optional[torch.LongTensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        seq_out, pooled, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        logits        = self.qa_outputs(seq_out)          # (B, S, 2)
        start_logits  = logits[:, :, 0]                   # (B, S)
        end_logits    = logits[:, :, 1]                   # (B, S)

        loss = None
        if start_positions is not None and end_positions is not None:
            S = input_ids.size(1)
            start_positions = start_positions.clamp(0, S - 1)
            end_positions   = end_positions.clamp(0, S - 1)
            loss = (
                nn.CrossEntropyLoss()(start_logits, start_positions) +
                nn.CrossEntropyLoss()(end_logits,   end_positions)
            ) / 2.0
            if self.has_answer_head and is_impossible is not None:
                loss += nn.CrossEntropyLoss()(self.has_answer(pooled), is_impossible)

        out = {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}
        if self.has_answer_head:
            out["has_answer_logits"] = self.has_answer(pooled)
        return out


# ------------------------------------------------------------------ #
# Multiple Choice (SWAG)
# ------------------------------------------------------------------ #

class BertForMultipleChoice(nn.Module):
    """
    BERT + linear scorer applied to the [CLS] representation of each
    candidate, then softmax over candidates.

    Input shape: (batch, num_choices, seq_len) — each candidate is a
    separate sequence "[CLS] question [SEP] choice [SEP]".

    Used for: SWAG (4 choices).
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert       = BertModel(config)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self._init(config)

    def _init(self, config: BertConfig) -> None:
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids:      torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels:         Optional[torch.LongTensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_choices = input_ids.size(1)

        # Flatten choice dimension into batch
        def _flat(t):
            return t.view(-1, t.size(-1)) if t is not None else None

        _, pooled, _ = self.bert(
            input_ids=_flat(input_ids),
            attention_mask=_flat(attention_mask),
            token_type_ids=_flat(token_type_ids),
        )

        logits = self.classifier(self.dropout(pooled))     # (B * C, 1)
        logits = logits.view(-1, num_choices)               # (B, C)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


# ------------------------------------------------------------------ #
# Dependency Parsing (Universal Dependencies)
# ------------------------------------------------------------------ #

class _MLP(nn.Module):
    """Two-layer MLP with dropout — used to project BERT states to arc/label spaces."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.33):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,     hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BiaffineScorer(nn.Module):
    """
    Biaffine scorer (Dozat & Manning 2017):
        score(i, j) = dep_i^T U head_j  +  W [dep_i ; head_j]  +  b

    For arc prediction:  out_features = 1   → (B, S, S) arc scores
    For label prediction: out_features = num_labels → (B, S, S, L) scores
    """

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.out_features = out_features
        # Bilinear weight: (out_features, in_features, in_features)
        self.U = nn.Parameter(torch.zeros(out_features, in_features, in_features))
        # Linear on concatenation
        self.W = nn.Linear(2 * in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.U)

    def forward(self, dep: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        """
        dep:  (B, S, d)
        head: (B, S, d)
        Returns: (B, S, S[, out_features]) — score[b, i, j] = dep_i → head_j
        """
        B, S, d = dep.size()

        # Pairwise expansion: (B, S, S, d)
        dep_e  = dep.unsqueeze(2).expand(-1, -1, S, -1)
        head_e = head.unsqueeze(1).expand(-1, S, -1, -1)

        dep_f  = dep_e.reshape(B * S * S, d)
        head_f = head_e.reshape(B * S * S, d)

        # Bilinear: dep^T U head — computed via einsum over feature dims
        # (B*S*S, d) x (out, d, d) x (B*S*S, d) → (B*S*S, out)
        bilinear = torch.einsum("bi,oij,bj->bo", dep_f, self.U, head_f)
        bilinear = bilinear.view(B, S, S, self.out_features)

        # Linear part
        linear = self.W(torch.cat([dep_e, head_e], dim=-1))  # (B, S, S, out)

        scores = bilinear + linear
        return scores.squeeze(-1) if self.out_features == 1 else scores


class BertForDependencyParsing(nn.Module):
    """
    BERT + biaffine dependency parser (Dozat & Manning 2017).

    Two separate biaffine scorers:
      - arc scorer:   (B, S, S) — which token is the head of each dependent
      - label scorer: (B, S, S, num_labels) — which label for each arc

    Training: minimise cross-entropy over gold arcs and gold labels (using
    gold arcs for label scoring, as is standard).

    Evaluation: greedy decoding (argmax per dependent for arc + label).

    Used for: Universal Dependencies cross-lingual dependency parsing.
    """

    _ARC_DIM   = 512
    _LABEL_DIM = 128

    def __init__(self, config: BertConfig, num_labels: int):
        super().__init__()
        self.bert = BertModel(config)
        H = config.hidden_size

        # MLPs projecting BERT output to arc and label spaces
        self.arc_head_mlp  = _MLP(H, H, self._ARC_DIM)
        self.arc_dep_mlp   = _MLP(H, H, self._ARC_DIM)
        self.lbl_head_mlp  = _MLP(H, H, self._LABEL_DIM)
        self.lbl_dep_mlp   = _MLP(H, H, self._LABEL_DIM)

        # Biaffine scorers
        self.arc_scorer    = _BiaffineScorer(self._ARC_DIM,   out_features=1)
        self.label_scorer  = _BiaffineScorer(self._LABEL_DIM, out_features=num_labels)

    def forward(
        self,
        input_ids:      torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_ids:       Optional[torch.LongTensor] = None,   # (B, S) gold head indices
        label_ids:      Optional[torch.LongTensor] = None,   # (B, S) gold label indices
    ) -> Dict[str, Optional[torch.Tensor]]:
        seq_out, _, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        arc_head  = self.arc_head_mlp(seq_out)   # (B, S, arc_dim)
        arc_dep   = self.arc_dep_mlp(seq_out)
        arc_scores = self.arc_scorer(arc_dep, arc_head)     # (B, S, S)

        lbl_head  = self.lbl_head_mlp(seq_out)
        lbl_dep   = self.lbl_dep_mlp(seq_out)
        lbl_scores = self.label_scorer(lbl_dep, lbl_head)   # (B, S, S, num_labels)

        loss = None
        if head_ids is not None and label_ids is not None:
            B, S = head_ids.size()
            mask = (head_ids != -100)                        # valid positions

            # Arc loss: cross-entropy over head positions
            arc_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                arc_scores.view(B * S, S),
                head_ids.clamp(min=0).view(B * S),
            )
            # Set ignored positions to 0 before masking
            _hi = head_ids.clone()
            _hi[~mask] = 0
            arc_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                arc_scores.reshape(B * S, S),
                head_ids.reshape(B * S).clamp(min=0),
            )

            # Label loss: score only at gold head positions
            # Gather (B, S, num_labels) from (B, S, S, num_labels)
            head_idx = _hi.unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, lbl_scores.size(-1))
            gold_lbl_scores = lbl_scores.gather(2, head_idx).squeeze(2)  # (B, S, num_labels)
            lbl_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                gold_lbl_scores.reshape(B * S, -1),
                label_ids.reshape(B * S),
            )

            loss = arc_loss + lbl_loss

        # Greedy decode
        pred_heads  = arc_scores.argmax(dim=-1)              # (B, S)
        # For labels: gather at predicted heads
        ph_idx = pred_heads.unsqueeze(-1).unsqueeze(-1).expand(
            pred_heads.size(0), pred_heads.size(1), 1, lbl_scores.size(-1)
        )
        pred_labels = lbl_scores.gather(2, ph_idx).squeeze(2).argmax(dim=-1)

        return {
            "loss":        loss,
            "arc_scores":  arc_scores,
            "lbl_scores":  lbl_scores,
            "pred_heads":  pred_heads,
            "pred_labels": pred_labels,
        }
