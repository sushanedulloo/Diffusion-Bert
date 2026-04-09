"""
Evaluation metrics for all BERT downstream benchmarks.

  Task                   Metric(s)
  ─────────────────────  ─────────────────────────────────────
  CoLA                   Matthews correlation coefficient
  SST-2                  Accuracy
  MRPC, QQP              Accuracy + binary F1
  STS-B                  Pearson + Spearman correlation
  MNLI                   Accuracy (matched + mismatched)
  QNLI, RTE, WNLI        Accuracy
  SQuAD v1.1/v2.0        Exact Match (EM) + F1
  NER, XL-NER            Span-level F1  (seqeval)
  POS tagging            UPOS accuracy
  SWAG, RACE, Story Cloze Accuracy
  SciTail                Accuracy
  XNLI                   Per-language accuracy
  MLDoc                  Per-language accuracy
  Dep. parsing           LAS + UAS
  Generation quality     Generative PPL + MAUVE + Unigram Entropy
"""

import re
import string
import collections
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score


# ------------------------------------------------------------------ #
# GLUE metrics
# ------------------------------------------------------------------ #

def compute_matthews(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """Matthews correlation coefficient — CoLA."""
    return {"matthews_corrcoef": float(matthews_corrcoef(labels, preds))}


def compute_accuracy(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """Accuracy — SST-2, MNLI, QNLI, RTE, WNLI, SWAG, XNLI, MLDoc."""
    return {"accuracy": float(accuracy_score(labels, preds))}


def compute_accuracy_and_f1(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """Accuracy + binary F1 — MRPC, QQP."""
    acc = float(accuracy_score(labels, preds))
    f1  = float(f1_score(labels, preds, average="binary", zero_division=0))
    return {"accuracy": acc, "f1": f1, "acc_and_f1": (acc + f1) / 2}


def compute_pearson_spearman(preds: List[float], labels: List[float]) -> Dict[str, float]:
    """Pearson + Spearman correlation — STS-B."""
    p = float(pearsonr(preds, labels)[0])
    s = float(spearmanr(preds, labels)[0])
    return {"pearson": p, "spearman": s, "corr": (p + s) / 2}


# ------------------------------------------------------------------ #
# SQuAD metrics
# ------------------------------------------------------------------ #

def _normalize_answer(s: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _get_tokens(s: str) -> List[str]:
    return _normalize_answer(s).split()


def _compute_exact(pred: str, gold: str) -> int:
    return int(_normalize_answer(pred) == _normalize_answer(gold))


def _compute_f1(pred: str, gold: str) -> float:
    pred_toks  = _get_tokens(pred)
    gold_toks  = _get_tokens(gold)
    if not pred_toks or not gold_toks:
        return int(pred_toks == gold_toks)
    common     = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num_same   = sum(common.values())
    if num_same == 0:
        return 0.0
    precision  = num_same / len(pred_toks)
    recall     = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def compute_squad_metrics(
    predictions: Dict[str, str],
    references:  Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Compute SQuAD EM and F1.

    Args:
        predictions: {question_id: predicted_text}
        references:  {question_id: [list_of_gold_texts]}

    Returns:
        {"exact_match": float, "f1": float}
    """
    exact_scores: Dict[str, int]   = {}
    f1_scores:    Dict[str, float] = {}

    for qid, pred in predictions.items():
        golds = references.get(qid, [])
        if not golds:
            exact_scores[qid] = 0
            f1_scores[qid]    = 0.0
        else:
            exact_scores[qid] = max(_compute_exact(pred, g) for g in golds)
            f1_scores[qid]    = max(_compute_f1(pred, g)    for g in golds)

    total = len(exact_scores)
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0}
    return {
        "exact_match": 100.0 * sum(exact_scores.values()) / total,
        "f1":          100.0 * sum(f1_scores.values())    / total,
    }


# ------------------------------------------------------------------ #
# NER / POS metrics
# ------------------------------------------------------------------ #

def compute_ner_f1(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, float]:
    """
    Span-level F1 for NER using seqeval (IOB2 scheme).

    Falls back to micro token-level F1 on non-O tags if seqeval is
    not installed.
    """
    try:
        from seqeval.metrics import f1_score as seq_f1
        return {"f1": float(seq_f1(true_labels, pred_labels))}
    except ImportError:
        flat_t = [t for seq in true_labels for t in seq]
        flat_p = [p for seq in pred_labels for p in seq]
        non_o  = [i for i, t in enumerate(flat_t) if t != "O"]
        if not non_o:
            return {"f1": 0.0}
        f1 = float(f1_score(
            [flat_t[i] for i in non_o],
            [flat_p[i] for i in non_o],
            average="micro",
            zero_division=0,
        ))
        return {"f1": f1}


def compute_pos_accuracy(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    UPOS accuracy, ignoring positions labeled -100 (padding / special tokens).
    """
    valid = [(p, l) for p, l in zip(preds, labels) if l != -100]
    if not valid:
        return {"accuracy": 0.0}
    preds_v, labels_v = zip(*valid)
    return {"accuracy": float(accuracy_score(labels_v, preds_v))}


# ------------------------------------------------------------------ #
# Dependency parsing metrics
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# Generation quality metrics
# ------------------------------------------------------------------ #

def compute_generative_perplexity(
    texts:     List[str],
    model_name: str = "gpt2-large",
    device:    "torch.device" = None,
    batch_size: int = 8,
    max_length: int = 1024,
) -> float:
    """
    Score generated texts with a frozen GPT-2 Large autoregressive model.

    Generative perplexity = exp(-mean log P_GPT2(text)).
    Lower values indicate more fluent, natural-looking text.

    Args:
        texts:      list of generated text strings
        model_name: HuggingFace model id for the scorer (default: gpt2-large)
        device:     torch device; auto-detects CUDA if None
        batch_size: examples per forward pass
        max_length: truncation length in GPT-2 tokens

    Returns:
        Scalar generative perplexity.
    """
    import math
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scorer_tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    scorer_model.eval()

    total_log_prob = 0.0
    total_tokens   = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = scorer_tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = scorer_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            # outputs.loss is average NLL over non-padding tokens in the batch
            # We recompute token-level to be precise
            logits = outputs.logits[:, :-1, :]         # (B, L-1, V)
            targets = input_ids[:, 1:]                  # (B, L-1)
            mask    = attention_mask[:, 1:].bool()

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_prob = log_probs.gather(
                2, targets.unsqueeze(2)
            ).squeeze(2)                               # (B, L-1)

            total_log_prob += (token_log_prob * mask).sum().item()
            total_tokens   += mask.sum().item()

    if total_tokens == 0:
        return float("inf")
    return math.exp(-total_log_prob / total_tokens)


def compute_unigram_entropy(
    texts:     List[str],
    tokenizer,
) -> float:
    """
    Compute the unigram token entropy of generated texts.

        H = -Σ_v p(v) log p(v)

    where p(v) is the empirical frequency of token v across all texts.
    Higher entropy → more diverse; lower → more repetitive.
    Reference range for OpenWebText: [5.37, 5.55] nats.

    Args:
        texts:     list of generated strings
        tokenizer: BertTokenizerFast (used to tokenise generated texts)

    Returns:
        Scalar entropy in nats.
    """
    import math
    import collections

    counter: collections.Counter = collections.Counter()
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(ids)

    total = sum(counter.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def compute_dep_parsing_metrics(
    pred_heads:  List[List[int]],
    pred_labels: List[List[int]],
    gold_heads:  List[List[int]],
    gold_labels: List[List[int]],
    masks:       List[List[bool]],
) -> Dict[str, float]:
    """
    Labeled Attachment Score (LAS) and Unlabeled Attachment Score (UAS).

    A token is counted only when mask[i] is True.
    Typically mask excludes the [CLS]/[SEP] tokens and padding.

    Returns:
        {"uas": float, "las": float}  — values in range [0, 100]
    """
    total = uas_ok = las_ok = 0
    for ph, pl, gh, gl, m in zip(pred_heads, pred_labels, gold_heads, gold_labels, masks):
        for i, valid in enumerate(m):
            if not valid:
                continue
            total += 1
            if ph[i] == gh[i]:
                uas_ok += 1
                if pl[i] == gl[i]:
                    las_ok += 1

    if total == 0:
        return {"uas": 0.0, "las": 0.0}
    return {
        "uas": 100.0 * uas_ok / total,
        "las": 100.0 * las_ok / total,
    }
