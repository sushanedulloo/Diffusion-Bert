"""
Data collator for Masked Language Modeling (MLM).

Applies *dynamic* masking at batch construction time — each epoch the same
example receives a different mask, which provides implicit data augmentation
(this is the approach used in RoBERTa and is equivalent to what the original
BERT paper describes as "training with independent masks").

MLM masking scheme (Section 3.3.1, Devlin et al., 2019):
  Of the 15 % selected tokens:
    - 80 %: replaced with [MASK]
    - 10 %: replaced with a uniformly random vocabulary token
    - 10 %: kept unchanged (but still predicted at that position)

Labels:
  -100 at all non-selected positions (ignored by CrossEntropyLoss)
  original token id at all selected positions

Special tokens ([CLS], [SEP], [PAD]) are never selected for masking.
"""

import torch
from typing import Dict, List


class DataCollatorForLanguageModeling:
    """
    Collates a list of dataset examples into a padded batch and applies
    MLM masking in-place on `input_ids`.

    Args:
        tokenizer:       HuggingFace BertTokenizer (provides special token ids)
        mlm_probability: fraction of tokens to mask (default 0.15)
    """

    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self.tokenizer       = tokenizer
        self.mlm_probability = mlm_probability

        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size    = tokenizer.vocab_size
        self.pad_token_id  = tokenizer.pad_token_id

        # All special token ids that must never be masked
        self._special_ids = set(tokenizer.all_special_ids)

    # ------------------------------------------------------------------ #
    # Collation entry point
    # ------------------------------------------------------------------ #

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            examples: list of dicts from BertPretrainingDataset.__getitem__

        Returns:
            batch dict with keys:
                input_ids      — (B, S)  masked token ids
                attention_mask — (B, S)
                token_type_ids — (B, S)
                labels         — (B, S)  original ids at masked pos, -100 elsewhere
        """
        batch = {
            "input_ids":      torch.stack([e["input_ids"]      for e in examples]),
            "attention_mask": torch.stack([e["attention_mask"] for e in examples]),
            "token_type_ids": torch.stack([e["token_type_ids"] for e in examples]),
        }

        masked_input_ids, labels = self._mask_tokens(batch["input_ids"])
        batch["input_ids"] = masked_input_ids
        batch["labels"]    = labels

        return batch

    # ------------------------------------------------------------------ #
    # MLM masking
    # ------------------------------------------------------------------ #

    def _mask_tokens(
        self, input_ids: torch.Tensor
    ):
        """
        Randomly mask tokens according to the BERT MLM recipe.

        Returns:
            masked_input_ids: input_ids with masked positions modified
            labels:           original token ids at masked positions, -100 elsewhere
        """
        labels = input_ids.clone()

        # ------ Step 1: decide which positions to mask ------ #
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)

        # Build a boolean mask of positions that must NOT be masked
        # (special tokens + padding)
        protected = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in self._special_ids:
            protected |= input_ids.eq(special_id)
        protected |= input_ids.eq(self.pad_token_id)

        probability_matrix.masked_fill_(protected, value=0.0)
        is_masked = torch.bernoulli(probability_matrix).bool()  # (B, S)

        # Non-masked positions: label = -100 (ignored by loss)
        labels[~is_masked] = -100

        # ------ Step 2: 80 % → replace with [MASK] ------ #
        replace_with_mask = torch.bernoulli(
            torch.full(input_ids.shape, 0.8)
        ).bool() & is_masked

        masked_input_ids = input_ids.clone()
        masked_input_ids[replace_with_mask] = self.mask_token_id

        # ------ Step 3: 10 % → replace with random token ------ #
        # Of the remaining 20 %, half get random replacement, half stay
        replace_with_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & is_masked
            & ~replace_with_mask
        )
        random_tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=input_ids.shape,
            dtype=torch.long,
            device=input_ids.device,
        )
        masked_input_ids[replace_with_random] = random_tokens[replace_with_random]

        # ------ Step 4: remaining 10 % stay unchanged (no action needed) ------ #

        return masked_input_ids, labels
