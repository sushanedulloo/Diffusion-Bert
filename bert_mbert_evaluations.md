# BERT & mBERT Evaluations

## BERT — GLUE Benchmark (9 Tasks)

| Evaluation | Description | Task Type | Dataset Name | Dataset Description | Metric(s) |
|---|---|---|---|---|---|
| CoLA | Determines if an English sentence is grammatically acceptable | Linguistic acceptability | Corpus of Linguistic Acceptability | 8.5k sentences drawn from linguistics books and journal articles, labeled acceptable or unacceptable | Matthews correlation |
| SST-2 | Classify the sentiment of a movie review sentence as positive or negative | Sentiment classification | Stanford Sentiment Treebank v2 | 67k single sentences from movie reviews with binary positive/negative labels | Accuracy |
| MRPC | Determine whether two sentences are semantically equivalent paraphrases | Paraphrase detection | Microsoft Research Paraphrase Corpus | 3.7k sentence pairs extracted from online news sources, annotated for semantic equivalence | Accuracy, F1 |
| STS-B | Predict how similar two sentences are on a 1–5 scale | Semantic textual similarity | Semantic Textual Similarity Benchmark | 7k sentence pairs from news headlines, image captions, and Q&A forums with human similarity scores | Pearson correlation, Spearman correlation |
| QQP | Determine whether two Quora questions ask the same thing | Paraphrase detection | Quora Question Pairs | 364k question pairs from Quora, labeled as duplicate or not duplicate | Accuracy, F1 |
| MNLI | Predict whether a hypothesis is entailed, contradicted, or neutral given a premise | Natural language inference | Multi-Genre NLI Corpus | 393k sentence pairs spanning 10 genres (transcribed speech, fiction, government reports, etc.), 3-class labels | Accuracy (matched), Accuracy (mismatched) |
| QNLI | Given a question and a sentence, determine if the sentence contains the answer | QA / NLI | Stanford QA Dataset (converted) | 108k question–sentence pairs converted from SQuAD; binary: sentence contains the answer vs. not | Accuracy |
| RTE | Determine whether one sentence textually entails another | Textual entailment | RTE 1/2/3/5 combined | 2.5k sentence pairs aggregated from four annual RTE challenges; sourced from news and Wikipedia; binary entailment | Accuracy |
| WNLI | Resolve pronoun coreference and predict entailment (Winograd schema challenge converted to NLI) | Coreference / NLI | Winograd NLI (converted) | 634 training / 146 test examples derived from fiction books; adversarial dev set; BERT excluded this task from its final GLUE score | Accuracy |

---

## BERT — Question Answering

| Evaluation | Description | Task Type | Dataset Name | Dataset Description | Metric(s) |
|---|---|---|---|---|---|
| SQuAD v1.1 | Extract the exact answer span from a Wikipedia passage given a question | Extractive QA (span selection) | Stanford Question Answering Dataset v1.1 | 100k+ question–answer pairs on Wikipedia articles; every question has an answer span present in the passage | Exact Match (EM), F1 |
| SQuAD v2.0 | Extract answer span or predict "no answer" when the passage does not contain one | Extractive QA (with unanswerable questions) | Stanford Question Answering Dataset v2.0 | SQuAD v1.1 plus ~50k adversarially added unanswerable questions; model must abstain when no answer is present | Exact Match (EM), F1 |

---

## BERT — Named Entity Recognition

| Evaluation | Description | Task Type | Dataset Name | Dataset Description | Metric(s) |
|---|---|---|---|---|---|
| NER | Tag each token in a sentence with its named entity class (PER, LOC, ORG, MISC, or O) | Token classification (sequence labeling) | CoNLL-2003 (English) | News wire articles from the Reuters corpus annotated with 4 entity types: person, location, organization, miscellaneous; IOB2 tagging scheme | F1 (span-level) |

---

## BERT — Commonsense Inference

| Evaluation | Description | Task Type | Dataset Name | Dataset Description | Metric(s) |
|---|---|---|---|---|---|
| SWAG | Select the most plausible continuation of a given sentence from 4 choices; grounded in video captioning scenarios | Multiple-choice commonsense inference | Situations With Adversarial Generations | 113k multiple-choice questions from video-caption sentence pairs; adversarially filtered to remove superficial cues; 4 candidate sentence completions per question | Accuracy |

---

## mBERT — Cross-Lingual Transfer Evaluations

| Evaluation | Description | Task Type | Dataset Name | Dataset Description | Metric(s) |
|---|---|---|---|---|---|
| XNLI | Zero-shot cross-lingual NLI: fine-tune on English MultiNLI, evaluate on 15 other languages | Cross-lingual natural language inference | Cross-lingual NLI (XNLI) | Multilingual extension of MultiNLI covering 15 languages; 5k dev + 5k test premise–hypothesis pairs per language; human translated and annotated | Accuracy |
| Cross-lingual NER | Fine-tune on English CoNLL NER, evaluate zero-shot on other languages to measure transfer | Cross-lingual token classification | CoNLL-2003 + multilingual NER corpora | English Reuters NER as source; target languages evaluated on their respective news NER datasets; measures zero-shot entity transfer across scripts and languages | F1 (span-level) |
| Cross-lingual POS tagging | Fine-tune on English UD treebank, evaluate on other language treebanks without target-language training data | Cross-lingual sequence labeling | Universal Dependencies (UD) treebanks | Parallel and non-parallel UD treebanks across many languages with standardized UPOS tags; used to probe whether mBERT shares syntactic representations across languages | UPOS accuracy |
| Document classification (MLDoc) | Classify news documents into 4 topic categories; trained on English, evaluated on 7 other languages | Cross-lingual text classification | MLDoc (Reuters multilingual corpus) | A balanced, multilingual subset of the Reuters corpus covering 8 languages (English, German, French, Spanish, Italian, Russian, Chinese, Japanese); 4-class topic labels | Accuracy |
| Dependency parsing | Fine-tune on English UD treebank, evaluate cross-lingual dependency parsing on other language treebanks | Cross-lingual syntactic parsing | Universal Dependencies (UD) treebanks | Same UD treebank collection as POS tagging; evaluates labeled and unlabeled head-attachment accuracy of parsed dependency trees across languages | LAS (labeled attachment score), UAS (unlabeled attachment score) |

---

*Sources: Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (NAACL 2019); Wu & Dredze (2019) — Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT.*
