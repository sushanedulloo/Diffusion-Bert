"""
Instruction templates and sequence-building utilities for diffusion-style evaluation.

GLUE_TEMPLATES — per-task prompt config (instruction, label words, answer length)

build_instruction_sequence — tokenises an instruction + [MASK] answer slots
build_instruction_labels   — builds labels tensor for MDLM fine-tuning
decode_answer              — extracts generated tokens from target positions
match_label                — maps a decoded string to an integer label
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch


# ------------------------------------------------------------------ #
# Per-task GLUE template configs
# ------------------------------------------------------------------ #

# Each entry contains:
#   instruction_fn : (example_dict) -> str  — builds the instruction string
#   answer_prefix  : str — appended at the end of the instruction (e.g. "Answer:")
#   label_words    : Dict[int, str] — label_id -> label word
#   answer_len     : int — number of [MASK] tokens to allocate

GLUE_TEMPLATES: Dict[str, Dict] = {
    "cola": {
        "instruction_fn": lambda ex: (
            f"Is this sentence grammatically acceptable? "
            f"Sentence: {ex['sentence']}"
        ),
        "answer_prefix": "Answer:",
        "label_words":   {0: "no", 1: "yes"},
        "answer_len":    1,
    },
    "sst2": {
        "instruction_fn": lambda ex: (
            f"What is the sentiment of this movie review? "
            f"Review: {ex['sentence']}"
        ),
        "answer_prefix": "Sentiment:",
        "label_words":   {0: "negative", 1: "positive"},
        "answer_len":    1,
    },
    "mrpc": {
        "instruction_fn": lambda ex: (
            f"Are these two sentences paraphrases of each other? "
            f"Sentence 1: {ex['sentence1']} "
            f"Sentence 2: {ex['sentence2']}"
        ),
        "answer_prefix": "Answer:",
        "label_words":   {0: "no", 1: "yes"},
        "answer_len":    1,
    },
    "stsb": {
        "instruction_fn": lambda ex: (
            f"Rate the semantic similarity of these sentences on a scale from 0 to 5. "
            f"Sentence 1: {ex['sentence1']} "
            f"Sentence 2: {ex['sentence2']}"
        ),
        "answer_prefix": "Score:",
        "label_words":   {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
        "answer_len":    1,
    },
    "qqp": {
        "instruction_fn": lambda ex: (
            f"Are these two questions asking the same thing? "
            f"Question 1: {ex['question1']} "
            f"Question 2: {ex['question2']}"
        ),
        "answer_prefix": "Answer:",
        "label_words":   {0: "no", 1: "yes"},
        "answer_len":    1,
    },
    "mnli": {
        "instruction_fn": lambda ex: (
            f"Premise: {ex['premise']} "
            f"Hypothesis: {ex['hypothesis']} "
            f"Does the premise entail, contradict, or neither support the hypothesis?"
        ),
        "answer_prefix": "Answer:",
        # Single-token proxies: entailment=yes, neutral=maybe, contradiction=no
        "label_words":   {0: "yes", 1: "maybe", 2: "no"},
        "answer_len":    1,
    },
    "qnli": {
        "instruction_fn": lambda ex: (
            f"Question: {ex['question']} "
            f"Sentence: {ex['sentence']} "
            f"Does this sentence answer the question?"
        ),
        "answer_prefix": "Answer:",
        "label_words":   {0: "yes", 1: "no"},
        "answer_len":    1,
    },
    "rte": {
        "instruction_fn": lambda ex: (
            f"Premise: {ex['sentence1']} "
            f"Hypothesis: {ex['sentence2']} "
            f"Does the premise entail the hypothesis?"
        ),
        "answer_prefix": "Answer:",
        "label_words":   {0: "yes", 1: "no"},
        "answer_len":    1,
    },
    "wnli": {
        "instruction_fn": lambda ex: (
            f"Sentence: {ex['sentence1']} "
            f"Does \"{ex['sentence2']}\" logically follow?"
        ),
        "answer_prefix": "Answer:",
        "label_words":   {0: "no", 1: "yes"},
        "answer_len":    1,
    },
}


# ------------------------------------------------------------------ #
# Urdu (Indic) task templates
# ------------------------------------------------------------------ #
#
# Same schema as GLUE_TEMPLATES, but prompts, answer prefixes and label
# words are in Urdu so they tokenize correctly under an Urdu / multilingual
# tokenizer (default: jhu-clsp/mmBERT-base).
#
# IMPORTANT: pick label words that tokenize to exactly `answer_len` tokens
# under the tokenizer you're using.  Verify with:
#     tok.tokenize(label_word)
# If a word splits into more pieces, either pick a shorter synonym or
# bump answer_len.  The defaults below were chosen to be short and common.

URDU_TEMPLATES: Dict[str, Dict] = {
    # Binary sentiment — works with ai4bharat/IndicSentiment (ur split)
    # or any 0=negative / 1=positive Urdu dataset.
    "urdu_sentiment": {
        "instruction_fn": lambda ex: (
            f"اس جائزے کا جذبہ کیا ہے؟ جائزہ: "
            f"{ex.get('INDIC REVIEW', ex.get('text', ex.get('sentence', '')))}"
        ),
        "answer_prefix": "جذبہ:",
        # 0 = negative (منفی), 1 = positive (مثبت)
        "label_words":   {0: "منفی", 1: "مثبت"},
        "answer_len":    2,   # both words usually split into 2 subwords under mmBERT
    },

    # 3-way NLI — works with facebook/xnli (ur) and Divyanshu/indicxnli (ur).
    # XNLI label convention: 0=entailment, 1=neutral, 2=contradiction
    "urdu_nli": {
        "instruction_fn": lambda ex: (
            f"مقدمہ: {ex['premise']} "
            f"مفروضہ: {ex['hypothesis']} "
            f"کیا مقدمہ مفروضے کی تائید کرتا ہے؟"
        ),
        "answer_prefix": "جواب:",
        # entailment=yes (ہاں), neutral=maybe (شاید), contradiction=no (نہیں)
        "label_words":   {0: "ہاں", 1: "شاید", 2: "نہیں"},
        "answer_len":    2,
    },
}


# ------------------------------------------------------------------ #
# Sequence builder
# ------------------------------------------------------------------ #

def build_instruction_sequence(
    instruction_fn: Callable,
    answer_prefix:  str,
    answer_len:     int,
    example:        Dict,
    tokenizer,
    max_seq_length: int = 128,
) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor]:
    """
    Build a tokenised instruction sequence with [MASK] answer slots.

    Layout:
        [CLS] <instruction + answer_prefix> [MASK]×answer_len [SEP] [PAD]×...

    Returns:
        input_ids:      (max_seq_length,) LongTensor
        condition_mask: (max_seq_length,) BoolTensor
                         True  = condition (fixed during sampling)
                         False = target ([MASK] positions denoised by sampler)
        attention_mask: (max_seq_length,) LongTensor  (1 = real, 0 = pad)
    """
    cls_id  = tokenizer.cls_token_id
    sep_id  = tokenizer.sep_token_id
    pad_id  = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    instruction_text = instruction_fn(example)
    prefix_text      = instruction_text + " " + answer_prefix
    prefix_tokens    = tokenizer.encode(prefix_text, add_special_tokens=False)

    # Reserve slots: [CLS] + prefix + [MASK]×answer_len + [SEP]
    max_prefix_len = max_seq_length - answer_len - 2
    prefix_tokens  = prefix_tokens[:max_prefix_len]

    ids      = [cls_id] + prefix_tokens + [mask_id] * answer_len + [sep_id]
    real_len = len(ids)
    pad_len  = max_seq_length - real_len
    ids     += [pad_id] * pad_len

    attn = [1] * real_len + [0] * pad_len

    # Target positions: [1 + len(prefix_tokens) … 1 + len(prefix_tokens) + answer_len)
    ans_start = 1 + len(prefix_tokens)
    ans_end   = ans_start + answer_len

    cond = [True] * max_seq_length
    for i in range(ans_start, min(ans_end, max_seq_length)):
        cond[i] = False

    return (
        torch.tensor(ids,  dtype=torch.long),
        torch.tensor(cond, dtype=torch.bool),
        torch.tensor(attn, dtype=torch.long),
    )


# ------------------------------------------------------------------ #
# Labels builder (for MDLM fine-tuning)
# ------------------------------------------------------------------ #

def build_instruction_labels(
    input_ids:       torch.LongTensor,
    condition_mask:  torch.BoolTensor,
    true_answer_ids: List[int],
    tokenizer,
) -> torch.LongTensor:
    """
    Build a labels tensor for instruction fine-tuning.

    Condition positions → -100  (no loss)
    Padding positions   → -100  (no loss)
    Target positions    → true_answer_ids (MDLM loss computed here)

    If true_answer_ids is shorter than answer_len, remaining target
    positions are set to -100.
    """
    pad_id  = tokenizer.pad_token_id
    labels  = input_ids.clone()

    # Mask out condition and padding
    labels[condition_mask]        = -100
    labels[input_ids == pad_id]   = -100

    # Fill in true answer tokens at target positions
    target_positions = (~condition_mask).nonzero(as_tuple=True)[0]
    for i, pos in enumerate(target_positions):
        if i < len(true_answer_ids):
            labels[pos] = true_answer_ids[i]
        else:
            labels[pos] = -100  # answer shorter than answer_len

    return labels


# ------------------------------------------------------------------ #
# Answer decoding helpers
# ------------------------------------------------------------------ #

def decode_answer(
    generated_ids:  torch.LongTensor,   # (L,) single example
    condition_mask: torch.BoolTensor,   # (L,)
    tokenizer,
) -> str:
    """
    Decode the generated tokens at target positions (condition_mask=False).
    """
    pad_id     = tokenizer.pad_token_id
    target_ids = generated_ids[~condition_mask]
    target_ids = target_ids[target_ids != pad_id]
    return tokenizer.decode(target_ids.tolist(), skip_special_tokens=True).strip()


def match_label(
    decoded_answer: str,
    label_words:    Dict[int, str],
) -> int:
    """
    Map a decoded answer string to an integer label.

    Priority:
      1. Exact match (case-insensitive)
      2. Prefix match (decoded starts with label word, or label word starts with decoded)
      3. Fallback: label 0
    """
    da = decoded_answer.lower().strip()

    for label_id, word in label_words.items():
        if da == word.lower():
            return label_id

    for label_id, word in label_words.items():
        w = word.lower()
        if da.startswith(w) or w.startswith(da):
            return label_id

    return 0
