"""
Diffusion-style evaluation for CoNLL-2003 Named Entity Recognition.

Replaces per-token classification with instruction-prompted generation:
  - The model generates a structured entity list: "entity [TYPE] , ..."
  - Generated strings are parsed back to IOB2 sequences for evaluation
  - Metric: seqeval span-level F1

Template:
    [CLS] Extract named entities from the following sentence.
    Sentence: {sentence} Entities: [MASK]×50 [SEP]

Fine-tuning builds the target string from gold IOB2 tags:
    "London [LOC] , Barack Obama [PER]"

Entity types: PER, ORG, LOC, MISC

Usage:
    python evaluate.py ner --checkpoint ./ckpt --eval_mode generative
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from training.noise_schedule  import LinearAlphaScheduler
from evaluate.diffusion_utils import (
    load_diffusion_model_from_checkpoint,
    resolve_checkpoint,
    mdlm_sample_conditional,
)
from evaluate.instruction_templates import decode_answer
from evaluate.diffusion_finetuner   import DiffusionFineTuner, FineTuneArgs
from evaluate.metrics               import compute_ner_f1

logger = logging.getLogger(__name__)

# CoNLL-2003 label vocabulary
CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
LABEL2ID     = {l: i for i, l in enumerate(CONLL_LABELS)}
ID2LABEL     = {i: l for l, i in LABEL2ID.items()}

_ANSWER_LEN  = 50   # [MASK] tokens for the entity list


# ------------------------------------------------------------------ #
# Entity string helpers
# ------------------------------------------------------------------ #

def _iob2_to_entity_string(words: List[str], tags: List[str]) -> str:
    """
    Convert a word list + IOB2 tag list to a structured entity string.

    Example output: "London [LOC] , Barack Obama [PER]"
    """
    entities: List[Tuple[str, str]] = []
    current_words: List[str] = []
    current_type: Optional[str] = None

    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current_words and current_type:
                entities.append((" ".join(current_words), current_type))
            current_words = [word]
            current_type  = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_words.append(word)
        else:
            if current_words and current_type:
                entities.append((" ".join(current_words), current_type))
            current_words = []
            current_type  = None

    if current_words and current_type:
        entities.append((" ".join(current_words), current_type))

    if not entities:
        return "none"
    return " , ".join(f"{e} [{t}]" for e, t in entities)


def parse_entity_string(text: str) -> List[Tuple[str, str]]:
    """
    Parse a generated entity string back to (entity_text, entity_type) pairs.

    Handles formats like:
        "London [LOC] , Barack Obama [PER]"
        "none"  (no entities)
    """
    text = text.strip()
    if not text or text.lower() == "none":
        return []

    valid_types = {"PER", "ORG", "LOC", "MISC"}
    pattern     = re.compile(r"(.+?)\s*\[(\w+)\]")
    results: List[Tuple[str, str]] = []

    for part in text.split(","):
        part = part.strip()
        m    = pattern.match(part)
        if m:
            entity_text = m.group(1).strip()
            entity_type = m.group(2).strip().upper()
            if entity_type in valid_types and entity_text:
                results.append((entity_text, entity_type))

    return results


def _entities_to_iob2(words: List[str], entities: List[Tuple[str, str]]) -> List[str]:
    """
    Reconstruct IOB2 tags for `words` given a list of (entity_text, type) pairs.

    Simple greedy string-matching: for each entity, find the first occurrence
    of its words in the sequence and tag them.
    """
    tags = ["O"] * len(words)

    for entity_text, entity_type in entities:
        entity_words = entity_text.lower().split()
        n            = len(entity_words)

        for start in range(len(words) - n + 1):
            if [w.lower() for w in words[start : start + n]] == entity_words:
                # Check not already tagged
                if all(t == "O" for t in tags[start : start + n]):
                    tags[start] = f"B-{entity_type}"
                    for j in range(1, n):
                        tags[start + j] = f"I-{entity_type}"
                    break

    return tags


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class NERInstructionDataset(Dataset):
    """
    CoNLL-2003 examples formatted as instruction-prompted diffusion sequences.

    Each item:
        input_ids:      (max_seq_length,) LongTensor
        condition_mask: (max_seq_length,) BoolTensor
        attention_mask: (max_seq_length,) LongTensor
        labels:         (max_seq_length,) LongTensor  (training)
        raw_label:      0.0 (placeholder)

    Gold data stored separately (not batched):
        self.word_lists: List[List[str]]   — original words per example
        self.tag_lists:  List[List[str]]   — gold IOB2 tags per example
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:      BertTokenizerFast,
        max_seq_length: int  = 128,
        is_train:       bool = True,
    ):
        self.examples:  List[Dict]        = []
        self.word_lists: List[List[str]]  = []
        self.tag_lists:  List[List[str]]  = []

        cls_id  = tokenizer.cls_token_id
        sep_id  = tokenizer.sep_token_id
        pad_id  = tokenizer.pad_token_id
        mask_id = tokenizer.mask_token_id

        for ex in hf_dataset:
            words = ex["tokens"]
            tags  = [ID2LABEL[t] for t in ex["ner_tags"]]

            sentence = " ".join(words)

            # Instruction prefix
            prefix_text  = (
                f"Extract named entities from the following sentence. "
                f"Sentence: {sentence} Entities:"
            )
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

            max_prefix_len = max_seq_length - _ANSWER_LEN - 2
            prefix_tokens  = prefix_tokens[:max_prefix_len]

            ids      = [cls_id] + prefix_tokens + [mask_id] * _ANSWER_LEN + [sep_id]
            real_len = len(ids)
            pad_len  = max_seq_length - real_len
            ids     += [pad_id] * pad_len
            attn     = [1] * real_len + [0] * pad_len

            ans_start = 1 + len(prefix_tokens)
            ans_end   = ans_start + _ANSWER_LEN

            cond = [True] * max_seq_length
            for i in range(ans_start, min(ans_end, max_seq_length)):
                cond[i] = False

            input_ids      = torch.tensor(ids,  dtype=torch.long)
            condition_mask = torch.tensor(cond, dtype=torch.bool)
            attention_mask = torch.tensor(attn, dtype=torch.long)

            if is_train:
                entity_str = _iob2_to_entity_string(words, tags)
                true_ids   = tokenizer.encode(entity_str, add_special_tokens=False)[:_ANSWER_LEN]

                labels = input_ids.clone()
                labels[condition_mask]      = -100
                labels[input_ids == pad_id] = -100

                target_positions = (~condition_mask).nonzero(as_tuple=True)[0]
                for i, pos in enumerate(target_positions):
                    if i < len(true_ids):
                        labels[pos] = true_ids[i]
                    else:
                        labels[pos] = -100
            else:
                labels = torch.full((max_seq_length,), -100, dtype=torch.long)

            self.examples.append({
                "input_ids":      input_ids,
                "condition_mask": condition_mask,
                "attention_mask": attention_mask,
                "labels":         labels,
                "raw_label":      torch.tensor(0.0, dtype=torch.float),
            })
            self.word_lists.append(words)
            self.tag_lists.append(tags)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def run(
    checkpoint_dir: str,
    output_dir:     str,
    max_seq_length: int   = 128,
    batch_size:     int   = 32,
    num_epochs:     int   = 3,
    learning_rate:  float = 2e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
    num_steps:      int   = 200,
    temperature:    float = 1.0,
) -> Dict[str, float]:
    """
    Instruction-prompted diffusion evaluation for CoNLL-2003 NER.

    Returns:
        {"f1": float}  — seqeval span-level F1
    """
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = resolve_checkpoint(checkpoint_dir)

    logger.info(f"[NER-Diffusion] device={device}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    logger.info(f"Loading diffusion model from {ckpt_file} …")
    model     = load_diffusion_model_from_checkpoint(ckpt_file)
    scheduler = LinearAlphaScheduler()

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    hf_ds = hf_load("conll2003", trust_remote_code=True)

    logger.info("Building instruction datasets …")
    train_ds = NERInstructionDataset(
        hf_ds["train"], tokenizer, max_seq_length, is_train=True
    )
    eval_ds  = NERInstructionDataset(
        hf_ds["validation"], tokenizer, max_seq_length, is_train=False
    )

    args = FineTuneArgs(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_amp=use_amp,
    )

    finetuner = DiffusionFineTuner(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        scheduler=scheduler,
        args=args,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info(f"Fine-tuning on {len(train_ds)} examples for {num_epochs} epoch(s) …")
    finetuner.train()

    # ── Inference ────────────────────────────────────────────────────
    logger.info("Running conditional sampling on validation set …")
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    model.eval()

    all_true_tags: List[List[str]] = []
    all_pred_tags: List[List[str]] = []
    example_idx   = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        condition_mask = batch["condition_mask"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bsz            = input_ids.size(0)

        generated = mdlm_sample_conditional(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            condition_mask=condition_mask,
            scheduler=scheduler,
            attention_mask=attention_mask,
            num_steps=num_steps,
            temperature=temperature,
            device=device,
        )

        for i in range(bsz):
            decoded  = decode_answer(generated[i], condition_mask[i], tokenizer)
            words    = eval_ds.word_lists[example_idx]
            gold_tags = eval_ds.tag_lists[example_idx]

            entities  = parse_entity_string(decoded)
            pred_tags = _entities_to_iob2(words, entities)

            all_true_tags.append(gold_tags)
            all_pred_tags.append(pred_tags)
            example_idx += 1

    metrics = compute_ner_f1(all_true_tags, all_pred_tags)
    logger.info(f"  NER F1: {metrics['f1']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ner_diffusion_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results written to {out_path}")

    return metrics
