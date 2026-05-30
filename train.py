"""
train.py — Entry point for BERT pre-training from scratch.

Supports two pre-training objectives selected via --training_objective:

  standard  (default)
      Classic BERT MLM + NSP.
      Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers
      for Language Understanding", NAACL 2019.
      https://arxiv.org/abs/1810.04805

  mdlm
      Masked Diffusion Language Modeling (MDLM).
      Masking rate varies continuously via a noise schedule α(t);
      the loss is weighted by w(t) = −dα/dt / (1 − α(t)).
      Shi et al., "Simplified and Effective Masked Diffusion Language Models",
      NeurIPS 2024.  https://arxiv.org/abs/2406.07524

Usage examples
--------------
# Standard BERT MLM+NSP smoke-test (CPU):
python train.py \\
    --max_docs 100 --max_examples 5000 \\
    --num_train_steps 1000 --batch_size 8 --no_amp \\
    --output_dir ./smoke_standard

# MDLM diffusion smoke-test (CPU):
python train.py \\
    --training_objective mdlm --noise_schedule linear \\
    --max_docs 20 --max_examples 500 \\
    --num_train_steps 100 --batch_size 4 --no_amp \\
    --output_dir ./smoke_mdlm

# Resume from a checkpoint:
python train.py --resume_from_checkpoint ./output/checkpoint_step_0010000
"""

import argparse
import logging
import math
import sys

import torch
from transformers import AutoTokenizer

from config import BertConfig

# ------------------------------------------------------------------ #
# Model-size presets
# ------------------------------------------------------------------ #
# Quick presets to let pre-training fit on small GPUs (e.g. Kaggle free
# tier P100 / T4 with ~16 GB).  The "base" preset reproduces BERT-Base.
# "small" matches BERT-Small from Turc et al. 2019 (Well-Read Students
# Learn Better, https://arxiv.org/abs/1908.08962). "tiny" matches BERT-Tiny.
MODEL_SIZE_PRESETS = {
    "tiny":  dict(hidden_size=128, num_hidden_layers=2,  num_attention_heads=2,
                  intermediate_size=512),
    "mini":  dict(hidden_size=256, num_hidden_layers=4,  num_attention_heads=4,
                  intermediate_size=1024),
    "small": dict(hidden_size=512, num_hidden_layers=4,  num_attention_heads=8,
                  intermediate_size=2048),
    "base":  dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                  intermediate_size=3072),
}
from model.bert              import BertModel
from model.pretraining_heads import BertForPreTraining
from data.corpus             import TextCorpus
from data.dataset            import BertPretrainingDataset
from data.collator           import DataCollatorForLanguageModeling
from training.trainer        import BertPreTrainer

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Argument parser
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain BERT from scratch (standard MLM+NSP or MDLM diffusion)."
    )

    # ----- Data -------------------------------------------------------- #
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--language", type=str, default="en",
        help=(
            "Pre-training language code. "
            "'en' (default) → English Wikipedia + BooksCorpus (original BERT). "
            "'ur' → Urdu IndicCorp v2 + Urdu Wikipedia (IndicBERT-style). "
            "Any other code → Wikipedia in that language."
        ),
    )
    data.add_argument(
        "--tokenizer_name", type=str, default=None,
        help=(
            "HuggingFace tokenizer name. "
            "Defaults: 'bert-base-uncased' for --language en, "
            "'jhu-clsp/mmBERT-base' (Modern Multilingual BERT) for any "
            "other language."
        ),
    )
    data.add_argument(
        "--no_bookcorpus", action="store_true",
        help="Skip the BooksCorpus dataset (English only).",
    )
    data.add_argument(
        "--no_indiccorp", action="store_true",
        help="Skip AI4Bharat IndicCorp v2 (non-English only). "
             "Useful when IndicCorp access is gated and you only want Wikipedia.",
    )
    data.add_argument(
        "--max_docs", type=int, default=None,
        help="Maximum articles to load per source (None = all).",
    )
    data.add_argument(
        "--max_examples", type=int, default=None,
        help="Maximum total packed examples (None = all). Useful for quick tests.",
    )
    data.add_argument(
        "--wikipedia_date", type=str, default="20220301",
        help="Wikipedia dump date string used in the HuggingFace dataset id.",
    )

    # ----- Model ------------------------------------------------------- #
    model_g = parser.add_argument_group("Model (BERT-Base defaults)")
    model_g.add_argument(
        "--model_size", type=str, default=None,
        choices=list(MODEL_SIZE_PRESETS.keys()),
        help=(
            "Model-size preset. Overrides --hidden_size / "
            "--num_hidden_layers / --num_attention_heads / --intermediate_size. "
            "Use 'tiny' or 'mini' for Kaggle free-tier, 'base' for full BERT-Base."
        ),
    )
    model_g.add_argument("--vocab_size",             type=int, default=30_522)
    model_g.add_argument("--hidden_size",             type=int, default=768)
    model_g.add_argument("--num_hidden_layers",       type=int, default=12)
    model_g.add_argument("--num_attention_heads",     type=int, default=12)
    model_g.add_argument("--intermediate_size",       type=int, default=3072)
    model_g.add_argument("--max_position_embeddings", type=int, default=512)

    # ----- Training ---------------------------------------------------- #
    train_g = parser.add_argument_group("Training (paper exact values)")
    train_g.add_argument("--num_train_steps",   type=int,   default=1_000_000)
    train_g.add_argument("--batch_size",         type=int,   default=256)
    train_g.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Accumulate gradients over N mini-batches before each update.",
    )
    train_g.add_argument("--learning_rate",      type=float, default=1e-4)
    train_g.add_argument("--warmup_steps",        type=int,   default=10_000)
    train_g.add_argument("--weight_decay",        type=float, default=0.01)
    train_g.add_argument("--max_grad_norm",       type=float, default=1.0)
    train_g.add_argument("--mlm_probability",     type=float, default=0.15,
                         help="Fixed masking rate for standard MLM (ignored for mdlm).")
    train_g.add_argument("--phase1_ratio",        type=float, default=0.9)
    train_g.add_argument("--phase1_max_seq_len",  type=int,   default=128)
    train_g.add_argument("--phase2_max_seq_len",  type=int,   default=512)
    train_g.add_argument(
        "--dataloader_num_workers", type=int, default=4,
        help="Worker processes for the DataLoader.",
    )

    # ----- MDLM / Diffusion -------------------------------------------- #
    diff_g = parser.add_argument_group("MDLM / Diffusion pre-training")
    diff_g.add_argument(
        "--training_objective",
        type=str, default="standard", choices=["standard", "mdlm"],
        help=(
            "Pre-training objective. "
            "'standard' = classic MLM+NSP (Devlin et al. 2019). "
            "'mdlm' = Masked Diffusion Language Modeling (Shi et al. 2024)."
        ),
    )
    diff_g.add_argument(
        "--noise_schedule",
        type=str, default="linear", choices=["linear", "cosine"],
        help="Noise schedule α(t) for MDLM. Ignored for standard training.",
    )
    diff_g.add_argument(
        "--time_epsilon",
        type=float, default=1e-3,
        help="Minimum timestep ε (avoids degenerate masking near t=0). MDLM only.",
    )
    diff_g.add_argument(
        "--loss_norm_type",
        type=str, default="token", choices=["token", "sequence", "batch"],
        help="Loss normalisation strategy for MDLM. MDLM only.",
    )

    # ----- Epoch-based shortcut ---------------------------------------- #
    epoch_g = parser.add_argument_group("Epoch-based training (alternative to --num_train_steps)")
    epoch_g.add_argument(
        "--num_epochs", type=int, default=None,
        help=(
            "Train for N full passes over the dataset. "
            "When set, overrides --num_train_steps and auto-adjusts "
            "--warmup_steps (6 %% of total) and --save_steps (1 per epoch)."
        ),
    )

    # ----- Checkpointing / logging ------------------------------------- #
    ckpt = parser.add_argument_group("Checkpointing")
    ckpt.add_argument("--output_dir",             type=str, default="./output")
    ckpt.add_argument("--save_steps",             type=int, default=10_000)
    ckpt.add_argument("--logging_steps",          type=int, default=100)
    ckpt.add_argument("--resume_from_checkpoint", type=str, default=None)

    # ----- Google Drive ------------------------------------------------ #
    gdrive = parser.add_argument_group("Google Drive checkpoint upload")
    gdrive.add_argument(
        "--gdrive_folder_id", type=str, default=None,
        help="Google Drive folder ID to upload checkpoints to. "
             "Get it from the folder URL: drive.google.com/drive/folders/<ID>",
    )
    gdrive.add_argument(
        "--gdrive_token_path", type=str, default="token.pickle",
        help="Path to the OAuth token file generated by setup_gdrive.py.",
    )

    # ----- Hugging Face Hub -------------------------------------------- #
    hf = parser.add_argument_group("Hugging Face Hub checkpoint upload")
    hf.add_argument(
        "--hf_repo_id", type=str, default=None,
        help="HF Hub repo id (e.g. 'username/ur-mdlm-mini'). "
             "Auto-created if missing. Each checkpoint is uploaded as "
             "checkpoint_<tag>.pt, preserving all intermediate checkpoints.",
    )
    hf.add_argument(
        "--hf_private", action="store_true",
        help="When creating the HF repo, make it private (recommended).",
    )
    hf.add_argument(
        "--hf_token", type=str, default=None,
        help="HF token. Defaults to the cached login from huggingface-cli "
             "login or huggingface_hub.login(token=…).",
    )

    # ----- Misc -------------------------------------------------------- #
    misc = parser.add_argument_group("Misc")
    misc.add_argument("--seed",   type=int,  default=42)
    misc.add_argument("--no_amp", action="store_true",
                      help="Disable automatic mixed-precision (FP16).")

    return parser.parse_args()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------------------- #
    # Apply model-size preset (if any) — overrides matching CLI args
    # ---------------------------------------------------------------- #
    if args.model_size is not None:
        preset = MODEL_SIZE_PRESETS[args.model_size]
        logger.info(f"Applying model-size preset '{args.model_size}': {preset}")
        for k, v in preset.items():
            setattr(args, k, v)

    # ---------------------------------------------------------------- #
    # Build BertConfig from CLI args
    # ---------------------------------------------------------------- #
    config = BertConfig(
        vocab_size              = args.vocab_size,
        hidden_size             = args.hidden_size,
        num_hidden_layers       = args.num_hidden_layers,
        num_attention_heads     = args.num_attention_heads,
        intermediate_size       = args.intermediate_size,
        max_position_embeddings = args.max_position_embeddings,
        mlm_probability         = args.mlm_probability,
        phase1_max_seq_length   = args.phase1_max_seq_len,
        phase2_max_seq_length   = args.phase2_max_seq_len,
        phase1_ratio            = args.phase1_ratio,
        learning_rate           = args.learning_rate,
        adam_epsilon            = 1e-6,
        weight_decay            = args.weight_decay,
        max_grad_norm           = args.max_grad_norm,
        num_train_steps         = args.num_train_steps,
        warmup_steps            = args.warmup_steps,
        train_batch_size        = args.batch_size,
        # MDLM settings
        training_objective      = args.training_objective,
        noise_schedule          = args.noise_schedule,
        time_epsilon            = args.time_epsilon,
        loss_norm_type          = args.loss_norm_type,
    )

    logger.info(f"Training objective: {config.training_objective.upper()}")
    logger.info("Configuration:")
    for k, v in vars(config).items():
        logger.info(f"  {k} = {v}")

    # ---------------------------------------------------------------- #
    # Tokenizer
    # ---------------------------------------------------------------- #
    if args.tokenizer_name is None:
        args.tokenizer_name = (
            "bert-base-uncased"
            if args.language == "en"
            else "jhu-clsp/mmBERT-base"
        )
    logger.info(f"Loading tokenizer ({args.tokenizer_name}) …")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    config.vocab_size = tokenizer.vocab_size
    # Store mask_token_id on config for use in diffusion trainer
    config.mask_token_id = tokenizer.mask_token_id
    logger.info(
        f"Vocabulary size: {config.vocab_size:,}  |  [MASK] id: {config.mask_token_id}"
    )

    # ---------------------------------------------------------------- #
    # Load corpus
    # ---------------------------------------------------------------- #
    logger.info(f"Loading text corpus (language={args.language}) …")
    corpus = TextCorpus(
        language       = args.language,
        use_bookcorpus = not args.no_bookcorpus,
        use_indiccorp  = not args.no_indiccorp,
        max_docs       = args.max_docs,
        wikipedia_date = args.wikipedia_date,
    )

    # ---------------------------------------------------------------- #
    # Build dataset
    # (The NSP-pair dataset is shared by both objectives; the MDLM
    #  collator simply drops the next_sentence_label key.)
    # ---------------------------------------------------------------- #
    logger.info("Creating pre-training dataset …")
    dataset = BertPretrainingDataset(
        corpus         = corpus,
        tokenizer      = tokenizer,
        max_seq_length = config.phase2_max_seq_length,  # 512 (Phase 1 trims on-the-fly)
        max_examples   = args.max_examples,
        seed           = args.seed,
    )

    # ---------------------------------------------------------------- #
    # Epoch-based training: convert epochs → steps
    # ---------------------------------------------------------------- #
    if args.num_epochs is not None:
        steps_per_epoch = max(1, len(dataset) // args.batch_size)
        args.num_train_steps = args.num_epochs * steps_per_epoch
        # Warmup = 6 % of total steps (sensible for short runs)
        if args.warmup_steps == 10_000:          # only override the default
            args.warmup_steps = max(100, args.num_train_steps // 16)
        args.save_steps   = steps_per_epoch      # save exactly once per epoch
        args.logging_steps = max(10, steps_per_epoch // 20)
        logger.info(
            f"Epoch mode: {args.num_epochs} epochs  "
            f"× {steps_per_epoch:,} steps/epoch  "
            f"= {args.num_train_steps:,} total steps  |  "
            f"warmup={args.warmup_steps}  save_every={args.save_steps}"
        )

    # ---------------------------------------------------------------- #
    # Post-save callback: GDrive OR Hugging Face Hub (mutually exclusive)
    # ---------------------------------------------------------------- #
    save_callback = None
    if args.gdrive_folder_id and args.hf_repo_id:
        raise ValueError(
            "Pass either --gdrive_folder_id OR --hf_repo_id, not both."
        )

    if args.gdrive_folder_id:
        from gdrive_uploader import upload_checkpoint as _gd_upload
        save_callback = lambda ckpt_dir: _gd_upload(
            ckpt_dir,
            folder_id  = args.gdrive_folder_id,
            token_path = args.gdrive_token_path,
        )
        logger.info(
            f"Google Drive upload enabled → folder {args.gdrive_folder_id} "
            f"(token: {args.gdrive_token_path})"
        )

    elif args.hf_repo_id:
        import os as _os
        import shutil as _shutil
        from hf_uploader import upload_checkpoint as _hf_upload

        def _hf_callback(ckpt_dir: str) -> None:
            # 1) Upload to HF Hub
            _hf_upload(
                ckpt_dir,
                repo_id  = args.hf_repo_id,
                hf_token = args.hf_token,
                private  = args.hf_private,
            )
            # 2) Keep a local copy in output_dir (the trainer will delete
            #    its tempdir right after this callback returns).
            tag = _os.path.basename(ckpt_dir.rstrip("/"))
            tgt = _os.path.join(args.output_dir, tag)
            if _os.path.abspath(ckpt_dir) != _os.path.abspath(tgt):
                _os.makedirs(args.output_dir, exist_ok=True)
                if _os.path.exists(tgt):
                    _shutil.rmtree(tgt, ignore_errors=True)
                _shutil.copytree(ckpt_dir, tgt)

        save_callback = _hf_callback
        logger.info(
            f"Hugging Face Hub upload enabled → "
            f"{args.hf_repo_id} (private={args.hf_private}). "
            f"Local copy kept at {args.output_dir}/checkpoint_*."
        )

    gdrive_callback = save_callback   # preserve old variable name used below

    # ================================================================ #
    # Branch: Standard BERT (MLM + NSP)  vs.  MDLM (diffusion)
    # ================================================================ #

    if config.training_objective == "standard":
        # ------------------------------------------------------------ #
        # Standard MLM + NSP
        # ------------------------------------------------------------ #
        logger.info("Setting up standard MLM+NSP pre-training …")

        collator = DataCollatorForLanguageModeling(
            tokenizer       = tokenizer,
            mlm_probability = config.mlm_probability,
        )

        bert  = BertModel(config)
        model = BertForPreTraining(config, bert)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {num_params:,}")

        trainer = BertPreTrainer(
            model               = model,
            config              = config,
            train_dataset       = dataset,
            collator            = collator,
            output_dir          = args.output_dir,
            use_amp             = not args.no_amp,
            post_save_callback  = gdrive_callback,
        )
        trainer.train(
            num_train_steps             = args.num_train_steps,
            batch_size                  = args.batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            learning_rate               = args.learning_rate,
            warmup_steps                = args.warmup_steps,
            weight_decay                = args.weight_decay,
            max_grad_norm               = args.max_grad_norm,
            save_steps                  = args.save_steps,
            logging_steps               = args.logging_steps,
            phase1_ratio                = args.phase1_ratio,
            phase1_max_seq_len          = args.phase1_max_seq_len,
            phase2_max_seq_len          = args.phase2_max_seq_len,
            dataloader_num_workers      = args.dataloader_num_workers,
            resume_from_checkpoint      = args.resume_from_checkpoint,
        )

    else:
        # ------------------------------------------------------------ #
        # MDLM diffusion pre-training
        # ------------------------------------------------------------ #
        logger.info(
            f"Setting up MDLM diffusion pre-training "
            f"(schedule={config.noise_schedule}, ε={config.time_epsilon}, "
            f"norm={config.loss_norm_type}) …"
        )

        from data.diffusion_collator     import DiffusionCollator
        from model.diffusion_heads       import BertForDiffusion
        from training.diffusion_trainer  import MDLMPreTrainer
        from training.noise_schedule     import get_noise_scheduler

        noise_scheduler = get_noise_scheduler(config.noise_schedule)
        logger.info(f"Noise scheduler: {noise_scheduler.__class__.__name__}")

        special_ids = set(tokenizer.all_special_ids)
        collator = DiffusionCollator(
            special_token_ids = special_ids,
            pad_token_id      = tokenizer.pad_token_id,
        )

        bert  = BertModel(config)
        model = BertForDiffusion(config, bert)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {num_params:,}")

        trainer = MDLMPreTrainer(
            model               = model,
            config              = config,
            train_dataset       = dataset,
            collator            = collator,
            scheduler           = noise_scheduler,
            output_dir          = args.output_dir,
            use_amp             = not args.no_amp,
            post_save_callback  = gdrive_callback,
        )
        trainer.train(
            num_train_steps             = args.num_train_steps,
            batch_size                  = args.batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            learning_rate               = args.learning_rate,
            warmup_steps                = args.warmup_steps,
            weight_decay                = args.weight_decay,
            max_grad_norm               = args.max_grad_norm,
            save_steps                  = args.save_steps,
            logging_steps               = args.logging_steps,
            phase1_ratio                = args.phase1_ratio,
            phase1_max_seq_len          = args.phase1_max_seq_len,
            phase2_max_seq_len          = args.phase2_max_seq_len,
            dataloader_num_workers      = args.dataloader_num_workers,
            resume_from_checkpoint      = args.resume_from_checkpoint,
        )


if __name__ == "__main__":
    main()
