# BERT Pre-training from Scratch (MDLM)

A clean, modular PyTorch implementation of **BERT pre-training**, built entirely from scratch (no HuggingFace model classes). Pre-training uses the **MDLM (Masked Diffusion Language Modeling)** objective — a continuous-time absorbing-state diffusion process that replaces the original MLM + NSP objectives.

> Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).
> **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.**
> *NAACL 2019.* https://arxiv.org/abs/1810.04805

> Shi, J., Han, K., Wang, Z., Doucet, A., & Titsias, M. K. (2024).
> **Simplified and Effective Masked Diffusion Language Models.**
> *NeurIPS 2024.* https://arxiv.org/abs/2406.07524

---

## What is built from scratch

| Component | File | Description |
|-----------|------|-------------|
| Token + Position + Segment Embeddings | `model/embeddings.py` | Learned absolute position embeddings, summed with token and segment embeddings |
| Multi-Head Self-Attention | `model/attention.py` | Scaled dot-product attention across all heads in parallel |
| Position-wise FFN | `model/feed_forward.py` | Two-layer MLP with GELU activation |
| Transformer Encoder Layer | `model/encoder.py` | Attention + FFN with residual connections and post-LayerNorm |
| BertModel | `model/bert.py` | Full encoder stack + pooler |
| MLM Pre-training Head | `model/pretraining_heads.py` | MLM with weight tying (no NSP) |
| MDLM model head | `model/diffusion_heads.py` | BertForDiffusion — MLM head only, no loss computed inside model |
| Dataset | `data/dataset.py` | Greedy single-sequence packing `[CLS] tokens [SEP]` (no NSP) |
| MLM Collator | `data/collator.py` | Dynamic masking: 80% → `[MASK]`, 10% → random token, 10% → unchanged |
| MDLM Collator | `data/diffusion_collator.py` | Sets labels = input_ids; protects special tokens |
| Optimizer | `training/optimizer.py` | Adam with layer-specific weight decay |
| LR Scheduler | `training/scheduler.py` | Linear warmup + linear decay to 0 |
| Standard Trainer | `training/trainer.py` | MLM-only training loop, gradient accumulation, mixed precision, checkpointing |
| MDLM noise schedules | `training/noise_schedule.py` | Linear and cosine α(t) schedulers |
| MDLM Trainer | `training/diffusion_trainer.py` | Per-step timestep sampling, stochastic masking, weighted cross-entropy |

**Only the tokenizer is reused** (`bert-base-uncased` from HuggingFace Transformers). No model weights are loaded from HuggingFace.

---

## Training datasets (original BERT paper)

| Dataset | Size | Source |
|---------|------|--------|
| BooksCorpus | ~800 M words | Zhu et al., 2015 — `bookcorpus` on HuggingFace |
| English Wikipedia | ~2.5 B words | `wikipedia` (20220301.en) on HuggingFace |

These are the exact two datasets used in Section 3.1 of the original BERT paper.

---

## Model architecture (BERT-Base, English)

| Hyperparameter | Value |
|---|---|
| Hidden size (H) | 768 |
| Transformer layers (L) | 12 |
| Attention heads (A) | 12 |
| FFN intermediate size | 3,072 |
| Max sequence length | 512 |
| Vocabulary size | 30,522 (`bert-base-uncased`) |
| Type vocab size | 1 (single segment, no NSP) |
| Parameters | ~110 M |

---

## Training hyperparameters (original BERT paper values)

| Hyperparameter | Value | Reference |
|---|---|---|
| Optimizer | Adam | Appendix A |
| Learning rate | 1e-4 | Table 1 |
| Adam β₁ | 0.9 | Appendix A |
| Adam β₂ | 0.999 | Appendix A |
| Adam ε | 1e-6 | Appendix A |
| L2 weight decay | 0.01 | Appendix A |
| LR warmup steps | 10,000 | Appendix A |
| LR schedule | Linear decay | Appendix A |
| Dropout | 0.1 | Appendix A |
| Batch size | 256 | Appendix A |
| Total steps | 1,000,000 | Appendix A |
| MLM masking probability | 15% | Section 3.3.1 |
| Phase 1 (seq_len=128) | 900,000 steps | Appendix A |
| Phase 2 (seq_len=512) | 100,000 steps | Appendix A |
| Gradient clipping | 1.0 | Appendix A |
| Activation | GELU | Section 3.1 |

---

## Repository layout

```
BERT from scratch/
├── config.py                    # BertConfig dataclass — all hyperparameters
├── train.py                     # Pre-training entry point (CLI)
├── evaluate.py                  # Evaluation entry point (CLI, all benchmarks)
│
├── model/
│   ├── embeddings.py            # Token + position + segment embeddings
│   ├── attention.py             # Multi-head self-attention
│   ├── feed_forward.py          # GELU FFN (Intermediate + Output)
│   ├── encoder.py               # BertLayer + BertEncoder
│   ├── bert.py                  # BertModel + BertPooler
│   ├── pretraining_heads.py     # MLM head + BertForPreTraining (no NSP)
│   └── diffusion_heads.py       # BertForDiffusion — MLM head only (MDLM)
│
├── data/
│   ├── corpus.py                # English Wikipedia + BooksCorpus loader
│   ├── dataset.py               # Single-sequence packing (no NSP)
│   ├── collator.py              # Dynamic MLM masking
│   └── diffusion_collator.py    # MDLM collator — labels=input_ids
│
├── training/
│   ├── optimizer.py             # Adam with weight decay groups
│   ├── scheduler.py             # Linear warmup + linear decay
│   ├── trainer.py               # BertPreTrainer — standard MLM training loop
│   ├── noise_schedule.py        # MDLM: LinearAlphaScheduler, CosineAlphaScheduler
│   └── diffusion_trainer.py     # MDLMPreTrainer — timestep-sampled diffusion loop
│
├── evaluate/
│   ├── fine_tuning_heads.py     # Task-specific heads + checkpoint loader
│   ├── fine_tuning_trainer.py   # Generic fine-tuning training loop
│   ├── metrics.py               # All evaluation metrics
│   ├── glue.py                  # GLUE (9 tasks)
│   ├── squad.py                 # SQuAD v1.1 and v2.0
│   ├── ner.py                   # CoNLL-2003 NER
│   ├── swag.py                  # SWAG commonsense MC
│   ├── snli.py                  # SNLI 3-way NLI (GPT-1 benchmark)
│   ├── race.py                  # RACE reading comprehension (GPT-1 benchmark)
│   ├── story_cloze.py           # Story Cloze commonsense MC (GPT-1 benchmark)
│   ├── scitail.py               # SciTail binary NLI (GPT-1 benchmark)
│   ├── perplexity.py            # PLL perplexity — zero-shot, no fine-tuning
│   └── generation_quality.py    # MDLM generation: GenPPL + MAUVE + Entropy
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Recommended) Install PyTorch with CUDA support
#    Visit https://pytorch.org/get-started/locally/ for the right command, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. (Optional) MAUVE score support
pip install mauve-text

# 5. (Optional) seqeval for NER span-level F1
pip install seqeval
```

---

## Pre-training

### Argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--training_objective` | `mdlm` | `standard` (MLM only) or `mdlm` (diffusion) |
| `--noise_schedule` | `linear` | `linear` or `cosine` α(t) schedule |
| `--time_epsilon` | `1e-3` | Minimum timestep ε |
| `--loss_norm_type` | `token` | Loss normalisation: `token`, `sequence`, or `batch` |
| `--use_bookcorpus` | `True` | Include BooksCorpus in training data |
| `--max_docs` | `None` | Cap on Wikipedia documents to load (None = full corpus) |
| `--max_examples` | `None` | Cap on total training examples |
| `--batch_size` | `256` | Per-step batch size |
| `--gradient_accumulation_steps` | `1` | Steps to accumulate before optimizer update |
| `--num_train_steps` | `1,000,000` | Total training steps |
| `--warmup_steps` | `10,000` | Linear LR warmup steps |
| `--learning_rate` | `1e-4` | Peak learning rate |
| `--max_seq_length` | `512` | Maximum sequence length |
| `--output_dir` | `./output` | Directory for checkpoints and logs |
| `--logging_steps` | `100` | Log loss every N steps |
| `--save_steps` | `10,000` | Save checkpoint every N steps |
| `--no_amp` | off | Disable automatic mixed precision (FP16) |
| `--resume_from_checkpoint` | `None` | Path to checkpoint directory to resume from |

---

### Quick smoke-test (CPU, MDLM)

Verifies the full pipeline runs end-to-end without a GPU.

```bash
python train.py \
    --training_objective mdlm \
    --max_docs 20 \
    --max_examples 500 \
    --num_train_steps 100 \
    --batch_size 4 \
    --no_amp \
    --logging_steps 10 \
    --output_dir ./smoke_mdlm
```

### Full MDLM pre-training (single GPU)

```bash
python train.py \
    --training_objective mdlm \
    --noise_schedule cosine \
    --loss_norm_type token \
    --output_dir ./output_mdlm \
    --logging_steps 100 \
    --save_steps 10000
```

### Multi-GPU pre-training (DataParallel)

DataParallel is enabled automatically when `torch.cuda.device_count() > 1`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --training_objective mdlm \
    --noise_schedule cosine \
    --batch_size 64 \
    --gradient_accumulation_steps 4 \
    --output_dir ./output_mdlm
```

### Resume from a checkpoint

```bash
python train.py \
    --training_objective mdlm \
    --resume_from_checkpoint ./output_mdlm/checkpoint_step_0500000 \
    --output_dir ./output_mdlm
```

---

## MDLM Diffusion Pre-training — How It Works

Instead of a fixed 15% MLM masking rate, MDLM treats pre-training as a continuous-time absorbing-state diffusion process:

1. For each training example, sample timestep `t ~ U[ε, 1)`
2. Masking probability at time `t` is `p_mask(t) = 1 − α(t)`
3. All selected tokens are replaced with `[MASK]` (no random-token or keep strategy)
4. Loss is weighted by `w(t) = −dα/dt / (1 − α(t))`, upweighting harder examples

### NSP is not used

This implementation removes Next Sentence Prediction entirely. The dataset packs a single sequence greedily into each `[CLS] ... [SEP]` window (following RoBERTa), which maximises token utilisation per training step.

### Noise schedules

| Schedule | α(t) formula | Behaviour |
|----------|------|-----------|
| `linear` (default) | `1 − t` | Uniform masking increase |
| `cosine` | `1 − cos(π/2 · (1−t))` | Slower masking at extremes, faster in the middle |

At `t → 0`: almost no masking (near-clean input). At `t → 1`: almost all tokens masked.

### Key differences from standard BERT

| Aspect | Standard BERT | This implementation (MDLM) |
|--------|--------------|------|
| Masking rate | Fixed 15% | Variable: `1 − α(t)`, `t ~ U[ε, 1)` |
| Token replacement | 80% [MASK] / 10% random / 10% unchanged | 100% [MASK] at selected positions |
| Loss weighting | Uniform | `w(t) = −dα/dt / (1 − α(t))` per example |
| Second objective | NSP | None |
| Where masking happens | Collator | Trainer (per-step, per-example) |

---

## Downstream Evaluation

All benchmarks from both the **BERT paper** and **GPT-1 paper** are implemented, plus MDLM-specific diffusion evaluations. Use `evaluate.py` as the unified entry point.

### Common fine-tuning arguments

These arguments apply to all fine-tuning sub-commands:

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Path to `checkpoint.pt` or its parent directory |
| `--output_dir` | `./results` | Directory to write results and fine-tuned checkpoints |
| `--batch_size` | `32` | Training batch size |
| `--epochs` | `3` | Fine-tuning epochs |
| `--lr` | `2e-5` | Learning rate |
| `--max_seq_length` | `128` | Maximum input length (tokens) |
| `--seed` | `42` | RNG seed |
| `--no_amp` | off | Disable automatic mixed precision |

---

### BERT paper tasks

#### GLUE (9 tasks)

```bash
# Single task
python evaluate.py glue --task cola       --checkpoint ./output_mdlm --output_dir ./results
python evaluate.py glue --task sst2       --checkpoint ./output_mdlm --output_dir ./results
python evaluate.py glue --task mrpc       --checkpoint ./output_mdlm --output_dir ./results
python evaluate.py glue --task stsb       --checkpoint ./output_mdlm --output_dir ./results
python evaluate.py glue --task qqp        --checkpoint ./output_mdlm --output_dir ./results
python evaluate.py glue --task mnli       --checkpoint ./output_mdlm --output_dir ./results
python evaluate.py glue --task qnli       --checkpoint ./output_mdlm --output_dir ./results
python evaluate.py glue --task rte        --checkpoint ./output_mdlm --output_dir ./results
python evaluate.py glue --task wnli       --checkpoint ./output_mdlm --output_dir ./results
```

| Task | Type | Metric |
|------|------|--------|
| CoLA | Linguistic acceptability | Matthews correlation coefficient |
| SST-2 | Sentiment classification | Accuracy |
| MRPC | Paraphrase detection | Accuracy + F1 |
| STS-B | Semantic textual similarity | Pearson + Spearman correlation |
| QQP | Question-pair paraphrase | Accuracy + F1 |
| MNLI | 3-way NLI | Accuracy (matched + mismatched) |
| QNLI | Question NLI | Accuracy |
| RTE | Textual entailment | Accuracy |
| WNLI | Winograd NLI | Accuracy |

#### SQuAD (extractive QA)

```bash
# SQuAD v1.1
python evaluate.py squad --version 1 --checkpoint ./output_mdlm --output_dir ./results

# SQuAD v2.0 (includes unanswerable questions)
python evaluate.py squad --version 2 --checkpoint ./output_mdlm --output_dir ./results
```

Metrics: **Exact Match (EM)**, **F1**

#### Named Entity Recognition — CoNLL-2003

```bash
python evaluate.py ner --checkpoint ./output_mdlm --output_dir ./results
```

Metric: **Span-level F1** (seqeval, IOB2 scheme)

#### SWAG (commonsense multiple choice)

```bash
python evaluate.py swag --checkpoint ./output_mdlm --output_dir ./results
```

Metric: **Accuracy** (4-way multiple choice, grounded commonsense inference)

---

### GPT-1 paper tasks

#### SNLI (Stanford Natural Language Inference)

570K premise-hypothesis pairs drawn from image captions. The largest NLI benchmark evaluated in the GPT-1 paper.

```bash
python evaluate.py snli --checkpoint ./output_mdlm --output_dir ./results
```

Metric: **Accuracy** (3-way: entailment / neutral / contradiction)

#### RACE (reading comprehension)

Multi-paragraph reading comprehension with 4-way multiple choice, sourced from Chinese English exams.

```bash
# Full dataset (middle + high school)
python evaluate.py race --checkpoint ./output_mdlm --output_dir ./results

# Sub-splits
python evaluate.py race --split middle --checkpoint ./output_mdlm --output_dir ./results
python evaluate.py race --split high   --checkpoint ./output_mdlm --output_dir ./results
```

> Note: RACE uses long passages. `--max_seq_length 512` is set automatically.

Metric: **Accuracy** (4-way multiple choice)

#### Story Cloze (commonsense story completion)

2-way multiple choice: given 4 story sentences, pick the correct 5th sentence.

```bash
# Automatic download via HuggingFace (requires accepting terms on the dataset page)
python evaluate.py story_cloze --checkpoint ./output_mdlm --output_dir ./results

# Local TSV files (if HuggingFace download is unavailable)
python evaluate.py story_cloze \
    --checkpoint ./output_mdlm \
    --output_dir ./results \
    --data_dir ./story_cloze_data
```

> If HuggingFace access fails, download the ROCStories dataset from
> https://cs.rochester.edu/nlp/rocstories/ and pass the directory with
> `--data_dir`. Expected files: `cloze_test_val__spring2016*.csv`

Metric: **Accuracy** (2-way)

#### SciTail (science-domain NLI)

Binary NLI from science exam questions: entails vs neutral.

```bash
python evaluate.py scitail --checkpoint ./output_mdlm --output_dir ./results
```

Metric: **Accuracy** (binary: entails / neutral)

---

### MDLM diffusion-specific evaluations (no fine-tuning)

These evaluations use the MDLM checkpoint directly — no task-specific fine-tuning.

#### PLL Perplexity (Pseudo-Log-Likelihood)

Standard autoregressive perplexity is undefined for masked/diffusion LMs. PLL is the correct proxy: mask each token position one at a time, collect `log P(token | context)`, then exponentiate the negative mean.

Evaluated on 6 datasets spanning different domains:

| Dataset | Domain |
|---------|--------|
| Wikitext-103 | General (encyclopedic) |
| Lambada | Long-range dependency |
| PTB (Penn Treebank) | Classic LM benchmark |
| AG News | News articles |
| PubMed | Biomedical abstracts |
| ArXiv | Scientific abstracts |

```bash
# All 6 datasets (default)
python evaluate.py perplexity --checkpoint ./output_mdlm --output_dir ./results

# Specific datasets
python evaluate.py perplexity \
    --checkpoint ./output_mdlm \
    --output_dir ./results \
    --datasets wikitext lambada ptb

# Domain-shift datasets only
python evaluate.py perplexity \
    --checkpoint ./output_mdlm \
    --output_dir ./results \
    --datasets ag_news pubmed arxiv

# Faster evaluation (cap batches per dataset)
python evaluate.py perplexity \
    --checkpoint ./output_mdlm \
    --output_dir ./results \
    --max_batches 50
```

Additional perplexity arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | all six | Datasets to evaluate |
| `--max_seq_length` | `128` | Chunk length in tokens |
| `--batch_size` | `16` | Batch size for forward passes |
| `--max_batches` | None | Cap batches per dataset (for quick testing) |

#### Generation Quality

Generate text unconditionally using the MDLM reverse diffusion process (absorbing diffusion sampler), then score with three metrics:

| Metric | Measures | Reference |
|--------|----------|-----------|
| **Generative PPL** | Fluency — scored by frozen GPT-2 Large | Lower = better |
| **MAUVE** | Distribution gap vs real text (Wikitext-103 val) | Higher = better (max 1.0) |
| **Unigram Entropy** | Token diversity — guards against repetition | Range [5.37, 5.55] for OpenWebText |

```bash
# Default: 512 samples, 1000 diffusion steps, linear schedule
python evaluate.py generation --checkpoint ./output_mdlm --output_dir ./results

# Custom settings
python evaluate.py generation \
    --checkpoint ./output_mdlm \
    --output_dir ./results \
    --num_samples 1024 \
    --num_steps 500 \
    --noise_schedule cosine \
    --temperature 0.9

# Skip MAUVE (if mauve-text is not installed)
python evaluate.py generation \
    --checkpoint ./output_mdlm \
    --output_dir ./results \
    --skip_mauve
```

Generation arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_samples` | `512` | Number of sequences to generate |
| `--seq_len` | `128` | Sequence length (tokens, including `[CLS]`/`[SEP]`) |
| `--num_steps` | `1000` | Diffusion denoising steps T |
| `--noise_schedule` | `linear` | `linear` or `cosine` |
| `--temperature` | `1.0` | Sampling temperature (lower = less random) |
| `--batch_size` | `64` | Sequences per generation batch |
| `--skip_mauve` | off | Skip MAUVE (requires `pip install mauve-text`) |

---

### Run all evaluations at once

Runs all fine-tuning benchmarks (GLUE, SQuAD, NER, SWAG, SNLI, RACE, Story Cloze, SciTail) followed by PLL perplexity and generation quality.

```bash
python evaluate.py all \
    --checkpoint ./output_mdlm \
    --output_dir ./results \
    --epochs 3 \
    --batch_size 32
```

Additional `all` arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--glue_tasks` | all 8 (excl. WNLI) | GLUE tasks to run |
| `--ppl_datasets` | all six | Perplexity datasets to evaluate |
| `--num_gen_samples` | `512` | Samples for generation quality eval |
| `--skip_mauve` | off | Skip MAUVE in generation eval |

Results are saved to `results/results_summary.json`.

---

### Complete evaluation command reference

```bash
# --- BERT paper tasks ---
python evaluate.py glue        --task <TASK>  --checkpoint ./ckpt --output_dir ./results
python evaluate.py squad       --version <1|2> --checkpoint ./ckpt --output_dir ./results
python evaluate.py ner                         --checkpoint ./ckpt --output_dir ./results
python evaluate.py swag                        --checkpoint ./ckpt --output_dir ./results

# --- GPT-1 paper tasks ---
python evaluate.py snli                        --checkpoint ./ckpt --output_dir ./results
python evaluate.py race        --split <all|middle|high> --checkpoint ./ckpt --output_dir ./results
python evaluate.py story_cloze                 --checkpoint ./ckpt --output_dir ./results
python evaluate.py scitail                     --checkpoint ./ckpt --output_dir ./results

# --- MDLM diffusion-specific (no fine-tuning) ---
python evaluate.py perplexity                  --checkpoint ./ckpt --output_dir ./results
python evaluate.py generation                  --checkpoint ./ckpt --output_dir ./results

# --- All at once ---
python evaluate.py all                         --checkpoint ./ckpt --output_dir ./results
```

---

### Fine-tuning hyperparameters (Section 4 of Devlin et al.)

| Hyperparameter | Recommended range |
|---|---|
| Learning rate | 2e-5, 3e-5, 5e-5 |
| Batch size | 16, 32 |
| Epochs | 2–4 |
| Max sequence length | 128 (most tasks), 512 (SQuAD, RACE) |

---

## MLM Masking Details (Section 3.3.1)

Of the 15% selected token positions (used in standard MLM training):
- **80%** → replaced with `[MASK]`
- **10%** → replaced with a uniformly random vocabulary token
- **10%** → kept unchanged (model still predicts the original token)

`[CLS]`, `[SEP]`, and `[PAD]` tokens are never selected for masking.

In MDLM pre-training, the 80/10/10 rule is dropped — all selected positions become `[MASK]`, and the masking rate varies continuously with the timestep.

---

## Two-Phase Training Schedule (Appendix A)

| Phase | Steps | Seq Length | Rationale |
|-------|-------|-----------|-----------|
| 1 | 900,000 (90%) | 128 | Much faster per step; captures most linguistic signal |
| 2 | 100,000 (10%) | 512 | Trains long-range attention and full position embeddings |

The dataset is built at `max_seq_length=512`. During Phase 1 the trainer trims each batch to 128 tokens on-the-fly — no separate dataset rebuild needed.

---

## Hardware requirements

| Setup | Min VRAM | Estimated time (1M steps) |
|-------|---------|--------------------------|
| Single A100 80 GB (batch 256) | 80 GB | ~3–4 days |
| 4 × A100 40 GB (batch 256) | 4 × 40 GB | ~1 day |
| Single consumer GPU 8 GB (batch 8, accum 32) | 8 GB | weeks (experimentation only) |

---

## Citation

```bibtex
@inproceedings{devlin-etal-2019-bert,
    title     = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author    = "Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina",
    booktitle = "Proceedings of NAACL-HLT 2019",
    year      = "2019",
    pages     = "4171--4186",
    url       = "https://arxiv.org/abs/1810.04805"
}

@inproceedings{shi-etal-2024-mdlm,
    title     = "Simplified and Effective Masked Diffusion Language Models",
    author    = "Shi, Jiaxin and Han, Kehang and Wang, Zhe and Doucet, Arnaud and Titsias, Michalis K.",
    booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
    year      = "2024",
    url       = "https://arxiv.org/abs/2406.07524"
}

@inproceedings{radford-2018-gpt,
    title     = "Improving Language Understanding by Generative Pre-Training",
    author    = "Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya",
    year      = "2018",
    url       = "https://openai.com/research/language-unsupervised"
}
```
