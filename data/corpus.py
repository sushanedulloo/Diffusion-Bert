"""
Corpus loader for BERT pre-training.

Datasets used in the original BERT paper (Section 3.3 / Appendix A):
  - BooksCorpus    — 800 M words of unpublished English books
                     (Zhu et al., 2015; accessed via HuggingFace `bookcorpus`)
  - English Wikipedia  — 2,500 M words, paragraphs only (no lists/tables)

Design notes
------------
* Each Wikipedia article and each BooksCorpus "book chunk" is treated as a
  separate document so that sentence-pair boundaries are respected.
* Very short paragraphs / sentences (< min_chars) are dropped to avoid
  degenerate pairs.
* `max_docs` can be set during development to limit memory usage.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Minimum character length for a paragraph to be kept as a sentence segment
_MIN_CHARS = 40

# Number of BooksCorpus lines to group into one pseudo-document
_BOOK_CHUNK_SIZE = 25


class TextCorpus:
    """
    Loads and stores the English BERT pre-training corpus as a list of *documents*.
    Each document is a list of text segments (paragraphs or sentences).
    NSP pair creation in BertPretrainingDataset relies on document boundaries.

    Args:
        use_bookcorpus:  include BooksCorpus (English, ~800 M words)
        max_docs:        cap on Wikipedia articles loaded (None = all)
        wikipedia_date:  Wikipedia dump date string (HuggingFace format)
    """

    def __init__(
        self,
        use_bookcorpus: bool = True,
        max_docs: Optional[int] = None,
        wikipedia_date: str = "20220301",
    ):
        self.use_bookcorpus = use_bookcorpus
        self.max_docs       = max_docs
        self.wikipedia_date = wikipedia_date

        # List[List[str]]  —  outer = documents, inner = segments
        self.documents: List[List[str]] = []

        self._load_wikipedia()
        if use_bookcorpus:
            self._load_bookcorpus()

        logger.info(f"Corpus ready: {len(self.documents):,} documents total.")

    # ------------------------------------------------------------------ #
    # Data loaders
    # ------------------------------------------------------------------ #

    def _load_wikipedia(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install `datasets`: pip install datasets")

        subset = f"{self.wikipedia_date}.en"
        try:
            logger.info(f"Loading English Wikipedia ({subset}) …")
            ds = load_dataset(
                "wikipedia",
                subset,
                split="train",
                trust_remote_code=True,
            )
            if self.max_docs is not None:
                ds = ds.select(range(min(self.max_docs, len(ds))))

            added = 0
            for article in ds:
                doc = self._split_article(article["text"])
                if len(doc) >= 2:
                    self.documents.append(doc)
                    added += 1

            logger.info(f"  English Wikipedia: added {added:,} documents.")

        except Exception as exc:
            logger.warning(f"  English Wikipedia could not be loaded: {exc}. Skipping.")

    def _load_bookcorpus(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install `datasets`: pip install datasets")

        try:
            logger.info("Loading BooksCorpus …")
            ds = load_dataset("bookcorpus", split="train", trust_remote_code=True)

            # Respect max_docs: each pseudo-doc = _BOOK_CHUNK_SIZE sentences
            if self.max_docs is not None:
                max_sentences = self.max_docs * _BOOK_CHUNK_SIZE
                ds = ds.select(range(min(max_sentences, len(ds))))
                logger.info(f"  BooksCorpus capped at {len(ds):,} sentences (max_docs={self.max_docs}).")

            sentences = [
                item["text"].strip()
                for item in ds
                if len(item["text"].strip()) >= _MIN_CHARS
            ]

            # Group consecutive lines into pseudo-documents
            added = 0
            for start in range(0, len(sentences), _BOOK_CHUNK_SIZE):
                chunk = sentences[start : start + _BOOK_CHUNK_SIZE]
                if len(chunk) >= 2:
                    self.documents.append(chunk)
                    added += 1

            logger.info(f"  BooksCorpus: {len(sentences):,} sentences → {added:,} documents.")

        except Exception as exc:
            logger.warning(f"  BooksCorpus could not be loaded: {exc}. Skipping.")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_article(text: str) -> List[str]:
        """
        Split a Wikipedia article into paragraph-level segments.
        Empty lines and short paragraphs are dropped.
        """
        return [
            para.strip()
            for para in text.split("\n")
            if len(para.strip()) >= _MIN_CHARS
        ]

    # ------------------------------------------------------------------ #
    # Sequence interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> List[str]:
        return self.documents[idx]
