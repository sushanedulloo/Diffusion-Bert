"""
Corpus loader for BERT pre-training.

Supports two language modes:

  English (original BERT recipe)
    - BooksCorpus + English Wikipedia
    - Datasets used in the original BERT paper
      (Devlin et al., 2019, Section 3.3 / Appendix A).

  Urdu (IndicBERT-style)
    - AI4Bharat IndicCorp v2 (Urdu) — primary source
      Doddapaneni et al., "Towards Leaving No Indic Language Behind:
      Building Monolingual Corpora, Benchmark and Models for Indic Languages",
      ACL 2023.  https://arxiv.org/abs/2212.05409
    - Urdu Wikipedia (fallback / supplementary)

Design notes
------------
* Each Wikipedia article / IndicCorp document is a separate "document"
  so that sentence-pair / packing boundaries are respected.
* Very short paragraphs / sentences (< min_chars) are dropped.
* `max_docs` caps how many documents are loaded — required when running
  on small machines (Kaggle free tier, etc.).
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
    Loads a pre-training corpus as a list of *documents*, each document being
    a list of text segments (paragraphs / sentences).  NSP-pair / sequence
    packing in BertPretrainingDataset relies on document boundaries.

    Args:
        language:        "en" for the original English recipe (Wiki + BooksCorpus),
                         "ur" for Urdu (IndicCorp v2 + Urdu Wikipedia), or any
                         other language code supported by Wikipedia.
        use_bookcorpus:  include BooksCorpus (English only — ignored for non-en).
        use_indiccorp:   include AI4Bharat IndicCorp (non-en only).
        max_docs:        cap on documents loaded per source (None = all).
        wikipedia_date:  Wikipedia dump date string.
    """

    def __init__(
        self,
        language: str = "en",
        use_bookcorpus: bool = True,
        use_indiccorp: bool = True,
        max_docs: Optional[int] = None,
        wikipedia_date: str = "20231101",
    ):
        self.language       = language.lower()
        self.use_bookcorpus = use_bookcorpus and self.language == "en"
        self.use_indiccorp  = use_indiccorp and self.language != "en"
        self.max_docs       = max_docs
        self.wikipedia_date = wikipedia_date

        # List[List[str]]  —  outer = documents, inner = segments
        self.documents: List[List[str]] = []

        if self.language == "en":
            self._load_wikipedia(lang="en")
            if self.use_bookcorpus:
                self._load_bookcorpus()
        else:
            # Indic / non-English: try IndicCorp first, then Wikipedia.
            if self.use_indiccorp:
                self._load_indiccorp(lang=self.language)
            self._load_wikipedia(lang=self.language)

        logger.info(
            f"Corpus ready ({self.language}): "
            f"{len(self.documents):,} documents total."
        )

    # ------------------------------------------------------------------ #
    # Wikipedia (any language)
    # ------------------------------------------------------------------ #

    def _load_wikipedia(self, lang: str) -> None:
        """
        Load Wikipedia for ``lang`` via the modern ``wikimedia/wikipedia``
        Parquet dataset.  The legacy ``wikipedia`` repo is a loading script
        and is rejected by ``datasets`` >= 3.0.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install `datasets`: pip install datasets")

        subset = f"{self.wikipedia_date}.{lang}"
        try:
            logger.info(f"Loading Wikipedia (wikimedia/wikipedia · {subset}) …")
            streaming = self.max_docs is not None
            ds = load_dataset(
                "wikimedia/wikipedia",
                subset,
                split="train",
                streaming=streaming,
            )
            if self.max_docs is not None:
                ds = ds.take(self.max_docs)

            added = 0
            for article in ds:
                doc = self._split_article(article["text"])
                if len(doc) >= 2:
                    self.documents.append(doc)
                    added += 1

            logger.info(f"  Wikipedia ({lang}): added {added:,} documents.")

        except Exception as exc:
            logger.warning(
                f"  Wikipedia ({lang}, {subset}) could not be loaded: {exc}. "
                f"Available dump dates: see https://huggingface.co/datasets/wikimedia/wikipedia "
                f"(common: 20231101). Skipping."
            )

    # ------------------------------------------------------------------ #
    # BooksCorpus (English only)
    # ------------------------------------------------------------------ #

    def _load_bookcorpus(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install `datasets`: pip install datasets")

        try:
            logger.info("Loading BooksCorpus …")
            ds = load_dataset("bookcorpus", split="train", trust_remote_code=True)

            if self.max_docs is not None:
                max_sentences = self.max_docs * _BOOK_CHUNK_SIZE
                ds = ds.select(range(min(max_sentences, len(ds))))
                logger.info(
                    f"  BooksCorpus capped at {len(ds):,} sentences (max_docs={self.max_docs})."
                )

            sentences = [
                item["text"].strip()
                for item in ds
                if len(item["text"].strip()) >= _MIN_CHARS
            ]

            added = 0
            for start in range(0, len(sentences), _BOOK_CHUNK_SIZE):
                chunk = sentences[start : start + _BOOK_CHUNK_SIZE]
                if len(chunk) >= 2:
                    self.documents.append(chunk)
                    added += 1

            logger.info(
                f"  BooksCorpus: {len(sentences):,} sentences → {added:,} documents."
            )

        except Exception as exc:
            logger.warning(f"  BooksCorpus could not be loaded: {exc}. Skipping.")

    # ------------------------------------------------------------------ #
    # AI4Bharat IndicCorp (Indic languages, including Urdu)
    # ------------------------------------------------------------------ #

    # Mapping from short language code → IndicCorp v2 config name.
    # IndicCorp v2 uses ISO 639-3 + script tags.
    _INDICCORP_V2_CONFIGS = {
        "ur":  "urd_Arab",   # Urdu (Arabic script)
        "hi":  "hin_Deva",
        "bn":  "ben_Beng",
        "ta":  "tam_Taml",
        "te":  "tel_Telu",
        "ml":  "mal_Mlym",
        "kn":  "kan_Knda",
        "gu":  "guj_Gujr",
        "mr":  "mar_Deva",
        "pa":  "pan_Guru",
        "or":  "ori_Orya",
        "as":  "asm_Beng",
    }

    def _load_indiccorp(self, lang: str) -> None:
        """
        Load AI4Bharat IndicCorp v2 for the requested language.

        IndicCorp v2 is gated on HuggingFace (you may need to accept the
        dataset terms once via `huggingface-cli login`).  If it cannot be
        loaded we silently fall back to Wikipedia.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install `datasets`: pip install datasets")

        config = self._INDICCORP_V2_CONFIGS.get(lang)
        if config is None:
            logger.info(
                f"  No IndicCorp v2 config known for language '{lang}'. Skipping IndicCorp."
            )
            return

        try:
            logger.info(f"Loading AI4Bharat IndicCorp v2 ({config}) …")
            streaming = self.max_docs is not None
            # IndicCorp v2 ships ONE HF config ('indiccorp_v2') with
            # per-language files under data/<lang_script>/.  To download
            # only Urdu (or whichever lang) we pick the glob directly.
            ds = load_dataset(
                "ai4bharat/IndicCorpV2",
                data_files=f"data/{config}/*",
                split="train",
                streaming=streaming,
            )
            if self.max_docs is not None:
                ds = ds.take(self.max_docs)

            # IndicCorp v2 stores plain text under the "text" column,
            # one document per row.  We split on newlines to recover
            # paragraph-level segments.
            added = 0
            for row in ds:
                text = row.get("text", "")
                if not text:
                    continue
                doc = self._split_article(text)
                if len(doc) >= 2:
                    self.documents.append(doc)
                    added += 1

            logger.info(f"  IndicCorp v2 ({config}): added {added:,} documents.")

        except Exception as exc:
            logger.warning(
                f"  IndicCorp v2 ({config}) could not be loaded: {exc}.\n"
                f"  → Falling back to Wikipedia only. "
                f"To enable IndicCorp, run `huggingface-cli login` and "
                f"accept the dataset terms at "
                f"https://huggingface.co/datasets/ai4bharat/IndicCorpV2"
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_article(text: str) -> List[str]:
        """Split a document into paragraph-level segments; drop short ones."""
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
