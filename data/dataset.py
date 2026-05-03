"""
BERT Pre-training Dataset — single-sequence packing.

Produces examples of the form:
    [CLS] token_1 token_2 … token_n [SEP]

Consecutive segments from the same document are packed greedily up to
`max_seq_length - 2` content tokens, maximising sequence utilisation.
When a document is exhausted the packer moves on to the next document.

token_type_ids are all zeros (single segment — no NSP).
MLM masking is applied dynamically in the DataCollator, not here.
"""

import hashlib
import logging
import os
import pickle
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BertPretrainingDataset(Dataset):
    """
    Constructs (input_ids, attention_mask, token_type_ids) examples
    from a TextCorpus for BERT diffusion pre-training.

    Args:
        corpus:          TextCorpus object (list of documents)
        tokenizer:       HuggingFace BertTokenizer
        max_seq_length:  total sequence length including [CLS] and [SEP]
        max_examples:    cap on total examples (None = no cap)
        seed:            kept for API compatibility (packing is deterministic)
    """

    def __init__(
        self,
        corpus,
        tokenizer,
        max_seq_length: int = 512,
        max_examples: Optional[int] = None,
        seed: int = 42,
        cache_dir: str = "./dataset_cache",
    ):
        self.tokenizer          = tokenizer
        self.max_seq_length     = max_seq_length
        self.max_content_tokens = max_seq_length - 2

        cache_path = self._cache_path(corpus, max_seq_length, max_examples, cache_dir)

        if os.path.exists(cache_path):
            logger.info(f"Loading dataset from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                self.examples = pickle.load(f)
            logger.info(f"  {len(self.examples):,} examples loaded from cache.")
        else:
            logger.info("Pre-tokenising corpus documents …")
            self.tokenized_docs: List[List[List[int]]] = self._tokenize_corpus(corpus)
            logger.info(f"  {len(self.tokenized_docs):,} documents tokenised.")

            logger.info("Building pre-training examples …")
            self.examples = self._build_examples(max_examples)
            logger.info(f"  {len(self.examples):,} examples created.")

            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.examples, f)
            logger.info(f"Dataset cached to: {cache_path}")

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_path(corpus, max_seq_length, max_examples, cache_dir) -> str:
        key = f"{len(corpus.documents)}_{max_seq_length}_{max_examples}_{corpus.use_bookcorpus}"
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        return os.path.join(cache_dir, f"dataset_{h}.pkl")

    # ------------------------------------------------------------------ #
    # Tokenisation
    # ------------------------------------------------------------------ #

    def _tokenize_corpus(self, corpus) -> List[List[List[int]]]:
        """
        Convert each document (list of text segments) into a list of
        token-id sequences.  Empty segments are dropped.
        """
        tokenized_docs = []
        for doc in tqdm(corpus.documents, desc="Tokenising", unit="doc", dynamic_ncols=True):
            tokenized_doc = []
            for segment in doc:
                ids = self.tokenizer.encode(segment, add_special_tokens=False)
                if ids:
                    tokenized_doc.append(ids)
            if tokenized_doc:
                tokenized_docs.append(tokenized_doc)
        return tokenized_docs

    # ------------------------------------------------------------------ #
    # Example construction — greedy single-sequence packing
    # ------------------------------------------------------------------ #

    def _build_examples(self, max_examples: Optional[int]) -> List[Dict]:
        """
        Pack consecutive segments from each document into full-length
        sequences.  Each example is:
            [CLS] <content tokens> [SEP] <padding>
        token_type_ids are all zeros.
        """
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        examples = []
        pbar = tqdm(
            self.tokenized_docs,
            desc="Building examples",
            unit="doc",
            dynamic_ncols=True,
        )

        for document in pbar:
            token_stream: List[int] = []
            for segment in document:
                token_stream.extend(segment)

            for start in range(0, len(token_stream), self.max_content_tokens):
                chunk = token_stream[start : start + self.max_content_tokens]
                if not chunk:
                    continue

                input_ids      = [cls_id] + chunk + [sep_id]
                seq_len        = len(input_ids)
                pad_len        = self.max_seq_length - seq_len

                input_ids      += [pad_id] * pad_len
                attention_mask  = [1] * seq_len + [0] * pad_len
                token_type_ids  = [0] * self.max_seq_length

                examples.append({
                    "input_ids":      input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                })

                if max_examples and len(examples) >= max_examples:
                    pbar.close()
                    return examples

            pbar.set_postfix(examples=len(examples))

        return examples

    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "input_ids":      torch.tensor(ex["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(ex["token_type_ids"], dtype=torch.long),
        }
