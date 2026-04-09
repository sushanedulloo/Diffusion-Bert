"""
Data collator for MDLM (Masked Diffusion Language Modeling) pre-training.

Key difference from DataCollatorForLanguageModeling:
  - Does NOT apply any masking.  The masking (forward diffusion process) is
    done dynamically inside MDLMPreTrainer at each training step, because the
    masking probability depends on a timestep t that is sampled per batch.
  - Sets labels = input_ids for all maskable positions.
  - Marks special tokens (CLS, SEP, PAD) as non-maskable with label = -100
    so the trainer never masks or penalises them.

The `maskable_mask = (labels != -100)` convention is the same one used by
dllm's MDLMTrainer.compute_loss to identify which positions are eligible
for stochastic masking.
"""

from typing import Dict, List, Set

import torch


class DiffusionCollator:
    """
    Collate a batch of BertPretrainingDataset examples for MDLM pre-training.

    Args:
        special_token_ids: set of token ids that must never be masked
                           (CLS, SEP, MASK, PAD, UNK …).  Passed as a set of
                           ints so this class has no dependency on the tokenizer
                           object itself.
        pad_token_id:      id used for padding; also protected.

    Usage::

        special_ids = set(tokenizer.all_special_ids)
        collator    = DiffusionCollator(special_token_ids=special_ids,
                                        pad_token_id=tokenizer.pad_token_id)
        loader = DataLoader(dataset, collate_fn=collator, batch_size=32)
    """

    def __init__(
        self,
        special_token_ids: Set[int],
        pad_token_id:      int = 0,
    ):
        self.special_token_ids = special_token_ids
        self.pad_token_id      = pad_token_id

    def __call__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Stack examples into a batch and build the labels tensor.

        Returns:
            input_ids:      (B, S) — original (unmasked) token ids
            attention_mask: (B, S) — 1 for real tokens, 0 for padding
            token_type_ids: (B, S) — segment ids
            labels:         (B, S) — original ids for maskable positions,
                                     -100 for special / padding tokens
        """
        input_ids      = torch.stack([e["input_ids"]      for e in examples])
        attention_mask = torch.stack([e["attention_mask"] for e in examples])
        token_type_ids = torch.stack([e["token_type_ids"] for e in examples])

        labels = self._build_labels(input_ids, attention_mask)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels":         labels,
        }

    def _build_labels(
        self,
        input_ids:      torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Build labels tensor:
          - Padding positions (attention_mask == 0) → -100
          - Special token positions → -100
          - All other positions → original token id

        Shape: (B, S)
        """
        labels = input_ids.clone()

        # Mask out padding
        labels[attention_mask == 0] = -100

        # Mask out every special token id
        for sid in self.special_token_ids:
            labels[input_ids == sid] = -100

        return labels
