"""
evaluate/ — Downstream evaluation suite for BERT (diffusion-pretrained).

Each sub-module corresponds to one (or more) benchmark(s):

    BERT paper tasks (Devlin et al., 2019):
        glue.py              — 9 GLUE tasks (CoLA, SST-2, MRPC, STS-B, QQP,
                               MNLI, QNLI, RTE, WNLI)
        squad.py             — SQuAD v1.1 and v2.0 extractive QA
        ner.py               — CoNLL-2003 Named Entity Recognition
        swag.py              — SWAG 4-way commonsense multiple choice

    GPT paper tasks (Radford et al., 2018):
        snli.py              — SNLI 3-way NLI (570K image-caption pairs)
        race.py              — RACE 4-way MC reading comprehension
        story_cloze.py       — Story Cloze 2-way MC commonsense completion
        scitail.py           — SciTail binary NLI (science domain)

    MDLM diffusion-specific evaluations (Shi et al., NeurIPS 2024):
        perplexity.py        — Pseudo-log-likelihood perplexity (Wikitext-103,
                               Lambada, PTB, AG News, PubMed, ArXiv)
                               — no fine-tuning required
        generation_quality.py— MDLM reverse-diffusion text generation scored by
                               Generative PPL (GPT-2 Large), MAUVE, Unigram Entropy

    Shared infrastructure:
        fine_tuning_heads.py  — Task-specific model heads + checkpoint loader
        fine_tuning_trainer.py— Generic fine-tuning training loop
        metrics.py            — All evaluation metrics
"""
