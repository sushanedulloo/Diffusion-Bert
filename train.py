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
import sys

import torch
from transformers import BertTokenizer

from config import BertConfig
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
        "--no_bookcorpus", action="store_true",
        help="Skip the BooksCorpus dataset (English).",
    )
    data.add_argument(
        "--max_docs", type=int, default=None,
        help="Maximum Wikipedia articles to load (None = all).",
    )
    data.add_argument(
        "--max_examples", type=int, default=None,
        help="Maximum total NSP examples (None = all). Useful for quick tests.",
    )
    data.add_argument(
        "--wikipedia_date", type=str, default="20220301",
        help="Wikipedia dump date string used in the HuggingFace dataset id.",
    )

    # ----- Model ------------------------------------------------------- #
    model_g = parser.add_argument_group("Model (BERT-Base defaults)")
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

    # ----- Checkpointing / logging ------------------------------------- #
    ckpt = parser.add_argument_group("Checkpointing")
    ckpt.add_argument("--output_dir",             type=str, default="./output")
    ckpt.add_argument("--save_steps",             type=int, default=10_000)
    ckpt.add_argument("--logging_steps",          type=int, default=100)
    ckpt.add_argument("--resume_from_checkpoint", type=str, default=None)

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
    logger.info("Loading tokenizer (bert-base-uncased) …")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config.vocab_size = tokenizer.vocab_size
    # Store mask_token_id on config for use in diffusion trainer
    config.mask_token_id = tokenizer.mask_token_id
    logger.info(f"Vocabulary size: {config.vocab_size:,}  |  [MASK] id: {config.mask_token_id}")

    # ---------------------------------------------------------------- #
    # Load corpus
    # ---------------------------------------------------------------- #
    logger.info("Loading text corpus …")
    corpus = TextCorpus(
        use_bookcorpus = not args.no_bookcorpus,
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
            model         = model,
            config        = config,
            train_dataset = dataset,
            collator      = collator,
            output_dir    = args.output_dir,
            use_amp       = not args.no_amp,
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
            model        = model,
            config       = config,
            train_dataset = dataset,
            collator     = collator,
            scheduler    = noise_scheduler,
            output_dir   = args.output_dir,
            use_amp      = not args.no_amp,
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
