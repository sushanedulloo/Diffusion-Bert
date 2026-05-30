# Urdu Diffusion-BERT Pretraining — Changes Report

This report documents the modifications made to the existing
"BERT from scratch" codebase so that it can pretrain a **text-diffusion
BERT (MDLM)** on **Urdu** text, with a configurable model size that
fits on the **Kaggle free-tier GPU** (T4 / P100, ~16 GB).

The original codebase already supported two pre-training objectives:

| Objective | Reference |
|-----------|-----------|
| `standard` — classic MLM + NSP | Devlin et al., 2019 (https://arxiv.org/abs/1810.04805) |
| `mdlm` — Masked Diffusion Language Modeling | Shi et al., 2024 (https://arxiv.org/abs/2406.07524) |

Both objectives still work; only the dataset/tokenizer pipeline and a
new model-size knob were added.

---

## 1. Codebase exploration (what was already there)

```
config.py                — BertConfig dataclass (model + training hyper-params)
train.py                 — CLI entry point, branches into standard vs mdlm
data/corpus.py           — Loads English Wikipedia + BooksCorpus
data/dataset.py          — Greedy single-sequence packing into 512-token examples
data/collator.py         — MLM masking collator (standard BERT)
data/diffusion_collator.py — No-masking collator (MDLM; masking is done in trainer)
model/bert.py            — BERT encoder
model/pretraining_heads.py — MLM + NSP heads
model/diffusion_heads.py   — MDLM head (token logits only)
training/trainer.py        — Standard BERT pre-trainer
training/diffusion_trainer.py — MDLM trainer (samples t, masks, weights loss by w(t))
training/noise_schedule.py    — Linear / cosine α(t) schedules
```

The diffusion trainer samples a timestep `t ~ U(ε, 1)` per batch,
masks each maskable token independently with probability `1 − α(t)`,
and weights the per-token cross-entropy by `w(t) = −dα/dt / (1 − α(t))`
(MDLM ELBO).

---

## 2. Changes made

### 2.1 Urdu dataset support — `data/corpus.py`

`TextCorpus` was rewritten to be language-aware:

* New `language` parameter (`"en"` keeps the original English recipe;
  `"ur"` switches to Urdu).
* For `"ur"` (and any non-English code), it loads:
  1. **AI4Bharat IndicCorp v2** for that language (Doddapaneni et al.,
     2023 — https://arxiv.org/abs/2212.05409). Urdu uses the config
     `urd_Arab`. The full mapping of supported Indic configs is in
     `_INDICCORP_V2_CONFIGS`.
  2. **Wikipedia** in that language (`20220301.ur` by default), as a
     supplement / fallback in case IndicCorp access is gated.
* If IndicCorp cannot be loaded (the dataset is gated and requires
  `huggingface-cli login` + accepting terms once), the loader emits a
  warning with the exact URL and falls back to Wikipedia only.
* `use_bookcorpus` is now silently ignored for non-English languages,
  and a new `use_indiccorp` flag lets you disable IndicCorp explicitly.
* Streaming mode is used automatically when `max_docs` is set, so we
  do not download the full multi-GB Urdu corpus on a small machine.

### 2.2 Tokenizer swap — `train.py`

`bert-base-uncased` does not cover the Urdu (Arabic) script. The
tokenizer is now resolved as:

* `--language en`  →  `bert-base-uncased` (default, unchanged)
* `--language ur` (or any other non-English) → **`jhu-clsp/mmBERT-base`**
  (Modern Multilingual BERT — JHU/CLSP, 2025; the modern mBERT replacement,
  trained on 1800+ languages including Urdu).
* Override with `--tokenizer_name <hf_id>` (e.g. `ai4bharat/indic-bert`,
  `xlm-roberta-base`, or the older `bert-base-multilingual-cased`).

`BertTokenizer` was replaced with `AutoTokenizer` so the loader works
for SentencePiece-based Indic tokenizers as well. The collators only
use standard `PreTrainedTokenizer` attributes (`mask_token_id`,
`pad_token_id`, `vocab_size`, `all_special_ids`) so they work for any
HF tokenizer.

### 2.3 Model-size presets — `train.py`

A new `--model_size` flag selects one of four presets and overrides
the matching `--hidden_size / --num_hidden_layers / …` args:

| Preset | H   | L  | A  | FFN  | Params (approx, vocab=120k) | Fits on Kaggle? |
|--------|-----|----|----|------|------|------|
| `tiny` | 128 | 2  | 2  | 512  | ~17 M  | yes, easily |
| `mini` | 256 | 4  | 4  | 1024 | ~42 M  | yes |
| `small`| 512 | 4  | 8  | 2048 | ~135 M | yes (small batch) |
| `base` | 768 | 12 | 12 | 3072 | ~178 M | only with grad-accum |

(Sizes follow Turc et al., 2019 — *Well-Read Students Learn Better*,
https://arxiv.org/abs/1908.08962.)

---

## 3. How to run

### 3.1 Smoke test — Urdu MDLM diffusion, tiny model, CPU/laptop

```bash
python train.py \
    --language ur \
    --training_objective mdlm --noise_schedule linear \
    --model_size tiny \
    --no_indiccorp \                  # skip gated dataset for a quick test
    --max_docs 50 --max_examples 500 \
    --num_train_steps 100 --batch_size 4 --no_amp \
    --output_dir ./smoke_ur_mdlm
```

### 3.2 Kaggle free-tier — Urdu MDLM diffusion, tiny/mini model

```bash
# (1) one-time, in a Kaggle cell, to enable IndicCorp:
# !huggingface-cli login
# then visit https://huggingface.co/datasets/ai4bharat/IndicCorpV2 and accept terms

python train.py \
    --language ur \
    --training_objective mdlm --noise_schedule cosine \
    --model_size mini \
    --max_docs 200000 \
    --num_epochs 1 \
    --batch_size 32 --gradient_accumulation_steps 4 \
    --phase1_max_seq_len 128 --phase2_max_seq_len 128 \
    --learning_rate 2e-4 \
    --dataloader_num_workers 2 \
    --output_dir /kaggle/working/ur_mdlm_mini
```

Notes for Kaggle:
* `--no_amp` is **not** passed — AMP/FP16 roughly halves memory on T4/P100.
* Phase 2 sequence length is held at 128 to fit in memory; bump to 256/512
  later on real compute.
* `--gradient_accumulation_steps` lets you keep an effective batch of 128
  while the actual per-step batch is small enough for the GPU.
* Use `--gdrive_folder_id <id>` (after running `setup_gdrive.py`) to push
  checkpoints to Drive — Kaggle sessions are ephemeral.

### 3.3 Full compute — Urdu MDLM, BERT-Base

```bash
python train.py \
    --language ur \
    --training_objective mdlm --noise_schedule cosine \
    --model_size base \
    --num_train_steps 1000000 \
    --batch_size 256 \
    --learning_rate 1e-4 --warmup_steps 10000 \
    --output_dir ./output/ur_mdlm_base
```

---

## 4. Diffusion-style evaluation for Urdu

Pattern (already in the codebase, just reused with Urdu prompts):

```
[CLS] <Urdu instruction>  <answer_prefix>  [MASK] [MASK]  [SEP] [PAD] …
       └────────── condition (fixed) ──────────┘  └ target ┘
```

The MDLM fine-tuner trains only the answer slots; the conditional sampler
fills only the answer slots at inference. The decoded tokens are matched
to a small Urdu `label_words` dict to recover the integer label. The
model never grows a classifier head — classification = generation.

### 4.1 What was added

| File | Change |
|------|--------|
| `evaluate/instruction_templates.py` | New `URDU_TEMPLATES` dict with `urdu_sentiment` (مثبت/منفی) and `urdu_nli` (ہاں/شاید/نہیں). |
| `evaluate/diffusion_urdu.py` (new) | Loads `ai4bharat/IndicSentiment` (Urdu) and `facebook/xnli` (`ur`), wraps them in `UrduInstructionDataset`, runs `DiffusionFineTuner` + conditional sampler, verifies that each label word tokenises within `answer_len` under the active tokenizer. |
| `evaluate.py` | Two new sub-commands: `urdu_sentiment` and `urdu_nli`, both accepting `--tokenizer_name`, `--num_steps`, `--temperature`, `--max_train`, `--max_eval`. |

### 4.2 Usage

```bash
# Urdu sentiment (IndicSentiment, ~1k examples — easy to run on Kaggle):
python evaluate.py urdu_sentiment \
    --checkpoint ./output/ur_mdlm_mini \
    --output_dir ./results \
    --tokenizer_name jhu-clsp/mmBERT-base \
    --epochs 5 --batch_size 16 --num_steps 50

# Urdu XNLI (the full train split is large — cap it for Kaggle):
python evaluate.py urdu_nli \
    --checkpoint ./output/ur_mdlm_mini \
    --output_dir ./results \
    --max_train 20000 --max_eval 2000 \
    --epochs 2 --batch_size 16 --num_steps 50
```

Results are written to `./results/<task>_diffusion_results.json` as
`{"accuracy": 0.xxx}`.

### 4.3 Notes / gotchas

* **Label-word token count.** Each Urdu label word must tokenise into at
  most `answer_len` pieces under your tokenizer. The runner logs the
  exact token IDs at start-up; if you see a "will be truncated" warning,
  either pick a shorter Urdu synonym or bump `answer_len` in
  `URDU_TEMPLATES`.
* **Pure zero-shot variant.** Set `--epochs 0` (or comment out
  `finetuner.train()`) to evaluate the pretrained checkpoint directly —
  useful as a baseline.
* **Other tasks.** Adding e.g. WikiANN-ur (NER) or IndicNLP topic
  classification is just one more entry in `URDU_TEMPLATES` plus one
  loader in `_TASK_LOADERS`. Everything else is reused.

## 5. Files touched

| File | Change |
|------|--------|
| `data/corpus.py` | Added `language` / `use_indiccorp` args; new `_load_indiccorp()` for AI4Bharat IndicCorp v2 (Urdu config `urd_Arab`); generalised `_load_wikipedia()` to any language. |
| `train.py` | Added `--language`, `--tokenizer_name`, `--no_indiccorp`, `--model_size`; switched `BertTokenizer` → `AutoTokenizer`; added `MODEL_SIZE_PRESETS` (tiny / mini / small / base). |

Nothing in `config.py`, the model, the collators, or the trainers
needed to change — they were already tokenizer-agnostic and objective-agnostic.

---

## 6. Caveats / follow-ups

* **IndicCorp v2 is gated.** First-time use requires
  `huggingface-cli login` plus accepting the dataset terms once at
  https://huggingface.co/datasets/ai4bharat/IndicCorpV2 . If access is
  blocked, the loader transparently falls back to Urdu Wikipedia.
* **Tokenizer choice matters.** The default for non-English is now
  `jhu-clsp/mmBERT-base` (Modern Multilingual BERT, 2025), which has
  much better Urdu coverage than the old `bert-base-multilingual-cased`.
  For a Urdu-only vocab (best compression per token) use
  `--tokenizer_name ai4bharat/indic-bert` (SentencePiece, Indic-only).
* **NSP loss** in the `standard` path uses `token_type_ids` all-zero
  packing today, so even on English the NSP signal is degenerate
  (this is a pre-existing property of the codebase, not changed here).
  For diffusion (`mdlm`) this doesn't matter.
* **Sequence-length curriculum.** On Kaggle, keeping
  `phase1_max_seq_len = phase2_max_seq_len = 128` is the safest setting;
  larger values blow up activation memory on a T4.
