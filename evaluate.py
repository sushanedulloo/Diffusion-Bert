"""
evaluate.py — Unified entry point for all BERT downstream evaluations.

Evaluations
───────────
BERT paper tasks (Devlin et al., 2019):
  glue         Fine-tune + evaluate on any of the 9 GLUE tasks
  squad        Fine-tune + evaluate on SQuAD v1.1 or v2.0
  ner          Fine-tune + evaluate on CoNLL-2003 NER
  swag         Fine-tune + evaluate on SWAG 4-way commonsense MC

GPT paper tasks (Radford et al., 2018):
  snli         Fine-tune + evaluate on SNLI 3-way NLI (570K pairs)
  race         Fine-tune + evaluate on RACE reading comprehension (4-way MC)
  story_cloze  Fine-tune + evaluate on Story Cloze (2-way commonsense MC)
  scitail      Fine-tune + evaluate on SciTail binary NLI (science domain)

MDLM diffusion-specific (no fine-tuning):
  perplexity   PLL perplexity on Wikitext-103, Lambada, PTB,
               AG News, PubMed, ArXiv
  generation   MDLM text generation quality: Generative PPL + MAUVE + Entropy

  all          Run all fine-tuning benchmarks (GLUE, SQuAD, NER, SWAG,
               SNLI, RACE, Story Cloze, SciTail) then perplexity and generation

Usage examples:
  python evaluate.py glue        --task cola --checkpoint ./ckpt --output_dir ./results
  python evaluate.py squad       --version 1 --checkpoint ./ckpt --output_dir ./results
  python evaluate.py ner         --checkpoint ./ckpt --output_dir ./results
  python evaluate.py swag        --checkpoint ./ckpt --output_dir ./results
  python evaluate.py snli        --checkpoint ./ckpt --output_dir ./results
  python evaluate.py race        --checkpoint ./ckpt --output_dir ./results
  python evaluate.py story_cloze --checkpoint ./ckpt --output_dir ./results
  python evaluate.py scitail     --checkpoint ./ckpt --output_dir ./results
  python evaluate.py perplexity  --checkpoint ./ckpt --output_dir ./results
  python evaluate.py generation  --checkpoint ./ckpt --output_dir ./results
  python evaluate.py all         --checkpoint ./ckpt --output_dir ./results
"""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Sub-command parsers
# ------------------------------------------------------------------ #

def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared across all sub-commands."""
    p.add_argument("--checkpoint",     required=True,
                   help="Path to checkpoint.pt file or its parent directory")
    p.add_argument("--output_dir",     default="./results",
                   help="Directory to write results and fine-tuned models")
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int,   default=128)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_amp",         action="store_true",
                   help="Disable automatic mixed precision (AMP/FP16)")


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="evaluate.py",
        description="BERT downstream evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ── GLUE ─────────────────────────────────────────────────────────
    p_glue = sub.add_parser("glue", help="GLUE benchmark (9 tasks)")
    _add_common_args(p_glue)
    p_glue.add_argument("--task", required=True,
                        choices=["cola", "sst2", "mrpc", "stsb", "qqp",
                                 "mnli", "qnli", "rte", "wnli"],
                        help="Which GLUE task to evaluate")
    p_glue.add_argument("--eval_mode",
                        choices=["discriminative", "generative"],
                        default="discriminative",
                        help="discriminative: BERT-style classification head. "
                             "generative: instruction-prompted diffusion generation.")
    p_glue.add_argument("--num_steps",   type=int,   default=200,
                        help="Reverse diffusion steps (generative mode only)")
    p_glue.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (generative mode only)")

    # ── SQuAD ────────────────────────────────────────────────────────
    p_sq = sub.add_parser("squad", help="SQuAD v1.1 / v2.0")
    _add_common_args(p_sq)
    p_sq.add_argument("--version", type=int, required=True, choices=[1, 2],
                      help="SQuAD version (1 or 2)")
    p_sq.add_argument("--batch_size", type=int, default=12,  # override default — SQuAD is heavier
                      help="Train batch size (default 12 for SQuAD)")
    p_sq.add_argument("--eval_mode",
                      choices=["discriminative", "generative"],
                      default="discriminative",
                      help="discriminative: span-extraction head. "
                           "generative: instruction-prompted diffusion generation.")
    p_sq.add_argument("--num_steps",   type=int,   default=200,
                      help="Reverse diffusion steps (generative mode only)")
    p_sq.add_argument("--temperature", type=float, default=1.0,
                      help="Sampling temperature (generative mode only)")

    # ── NER ──────────────────────────────────────────────────────────
    p_ner = sub.add_parser("ner", help="CoNLL-2003 NER")
    _add_common_args(p_ner)
    p_ner.add_argument("--eval_mode",
                       choices=["discriminative", "generative"],
                       default="discriminative",
                       help="discriminative: per-token classification. "
                            "generative: instruction-prompted diffusion generation.")
    p_ner.add_argument("--num_steps",   type=int,   default=200,
                       help="Reverse diffusion steps (generative mode only)")
    p_ner.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (generative mode only)")

    # ── SWAG ─────────────────────────────────────────────────────────
    p_swag = sub.add_parser("swag", help="SWAG commonsense multiple choice")
    _add_common_args(p_swag)
    p_swag.add_argument("--batch_size", type=int, default=16)
    p_swag.add_argument("--eval_mode",
                        choices=["discriminative", "generative"],
                        default="discriminative",
                        help="discriminative: multiple-choice classification head. "
                             "generative: zero-shot PLL scoring (no fine-tuning).")

    # ── XNLI ─────────────────────────────────────────────────────────
    p_xnli = sub.add_parser("xnli", help="XNLI zero-shot cross-lingual NLI")
    _add_common_args(p_xnli)
    p_xnli.add_argument("--languages", nargs="+", default=None,
                        help="XNLI language codes to evaluate (default: all 15)")

    # ── Cross-lingual NER ────────────────────────────────────────────
    p_xlner = sub.add_parser("xl_ner", help="Cross-lingual NER (WikiANN)")
    _add_common_args(p_xlner)
    p_xlner.add_argument("--languages", nargs="+", default=None)

    # ── POS tagging ──────────────────────────────────────────────────
    p_pos = sub.add_parser("pos", help="Cross-lingual POS tagging (Universal Dependencies)")
    _add_common_args(p_pos)
    p_pos.add_argument("--languages", nargs="+", default=None)

    # ── MLDoc ────────────────────────────────────────────────────────
    p_ml = sub.add_parser("mldoc", help="MLDoc cross-lingual document classification")
    _add_common_args(p_ml)
    p_ml.add_argument("--data_dir", required=True,
                      help="Path to MLDoc TSV files (see evaluate/mldoc.py for download instructions)")
    p_ml.add_argument("--languages",  nargs="+", default=None)
    p_ml.add_argument("--train_size", type=int,  default=1000, choices=[1000, 2000, 5000, 10000])

    # ── Dependency parsing ───────────────────────────────────────────
    p_dep = sub.add_parser("dep", help="Cross-lingual dependency parsing (Universal Dependencies)")
    _add_common_args(p_dep)
    p_dep.add_argument("--languages", nargs="+", default=None)

    # ── SNLI ─────────────────────────────────────────────────────────
    p_snli = sub.add_parser("snli", help="SNLI 3-way NLI (570K premise-hypothesis pairs)")
    _add_common_args(p_snli)

    # ── RACE ─────────────────────────────────────────────────────────
    p_race = sub.add_parser("race", help="RACE reading comprehension (4-way MC)")
    _add_common_args(p_race)
    p_race.add_argument("--split", default="all",
                        choices=["all", "middle", "high"],
                        help="RACE sub-split (default: all)")
    p_race.add_argument("--max_seq_length", type=int, default=512)

    # ── Story Cloze ──────────────────────────────────────────────────
    p_sc = sub.add_parser("story_cloze",
                          help="Story Cloze 2-way commonsense MC")
    _add_common_args(p_sc)
    p_sc.add_argument("--data_dir", default=None,
                      help="Directory with Story Cloze TSV files "
                           "(optional; tries HuggingFace if omitted)")

    # ── SciTail ──────────────────────────────────────────────────────
    p_sci = sub.add_parser("scitail", help="SciTail binary NLI (science domain)")
    _add_common_args(p_sci)

    # ── PLL Perplexity ───────────────────────────────────────────────
    p_ppl = sub.add_parser("perplexity",
                           help="PLL perplexity on Wikitext-103, Lambada, PTB "
                                "(no fine-tuning; uses MDLM checkpoint directly)")
    p_ppl.add_argument("--checkpoint",     required=True)
    p_ppl.add_argument("--output_dir",     default="./results")
    p_ppl.add_argument("--datasets",       nargs="+", default=None,
                       choices=["wikitext", "lambada", "ptb",
                                "ag_news", "pubmed", "arxiv"],
                       help="Datasets to evaluate (default: all six)")
    p_ppl.add_argument("--max_seq_length", type=int, default=128)
    p_ppl.add_argument("--batch_size",     type=int, default=16)
    p_ppl.add_argument("--max_batches",    type=int, default=None)
    p_ppl.add_argument("--seed",           type=int, default=42)

    # ── Generation Quality ───────────────────────────────────────────
    p_gen = sub.add_parser("generation",
                           help="MDLM generation quality: Generative PPL + MAUVE + Entropy "
                                "(no fine-tuning; uses MDLM checkpoint directly)")
    p_gen.add_argument("--checkpoint",     required=True)
    p_gen.add_argument("--output_dir",     default="./results")
    p_gen.add_argument("--num_samples",    type=int,   default=512)
    p_gen.add_argument("--seq_len",        type=int,   default=128)
    p_gen.add_argument("--num_steps",      type=int,   default=1000)
    p_gen.add_argument("--noise_schedule", default="linear",
                       choices=["linear", "cosine"])
    p_gen.add_argument("--temperature",    type=float, default=1.0)
    p_gen.add_argument("--batch_size",     type=int,   default=64)
    p_gen.add_argument("--skip_mauve",     action="store_true")
    p_gen.add_argument("--seed",           type=int,   default=42)

    # ── All ──────────────────────────────────────────────────────────
    p_all = sub.add_parser("all",
                           help="Run all benchmarks: GLUE, SQuAD, NER, SWAG, "
                                "SNLI, RACE, Story Cloze, SciTail, "
                                "PLL Perplexity (6 datasets), Generation Quality")
    _add_common_args(p_all)
    p_all.add_argument("--glue_tasks", nargs="+",
                       default=["cola", "sst2", "mrpc", "stsb", "qqp",
                                "mnli", "qnli", "rte"],
                       help="GLUE tasks to include (default: all except WNLI)")
    p_all.add_argument("--ppl_datasets", nargs="+", default=None,
                       help="Perplexity datasets (default: all)")
    p_all.add_argument("--num_gen_samples", type=int, default=512,
                       help="Samples for generation quality eval")
    p_all.add_argument("--skip_mauve",  action="store_true")

    return root


# ------------------------------------------------------------------ #
# Dispatch
# ------------------------------------------------------------------ #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args   = parser.parse_args()
    use_amp = not args.no_amp

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # ------------------------------------------------------------------ #
    if args.command == "glue":
        eval_mode = getattr(args, "eval_mode", "discriminative")
        if eval_mode == "generative":
            from evaluate.diffusion_glue import run as glue_run
            results = glue_run(
                task_name=args.task,
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
                use_amp=use_amp,
                num_steps=getattr(args, "num_steps", 200),
                temperature=getattr(args, "temperature", 1.0),
            )
        else:
            from evaluate.glue import run as glue_run
            results = glue_run(
                task_name=args.task,
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
                use_amp=use_amp,
            )
        all_results["glue"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "squad":
        eval_mode = getattr(args, "eval_mode", "discriminative")
        if eval_mode == "generative":
            from evaluate.diffusion_squad import run as squad_run
            results = squad_run(
                version=args.version,
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
                use_amp=use_amp,
                num_steps=getattr(args, "num_steps", 200),
                temperature=getattr(args, "temperature", 1.0),
            )
        else:
            from evaluate.squad import run as squad_run
            results = squad_run(
                version=args.version,
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
                use_amp=use_amp,
            )
        all_results["squad"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "ner":
        eval_mode = getattr(args, "eval_mode", "discriminative")
        if eval_mode == "generative":
            from evaluate.diffusion_ner import run as ner_run
            results = ner_run(
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
                use_amp=use_amp,
                num_steps=getattr(args, "num_steps", 200),
                temperature=getattr(args, "temperature", 1.0),
            )
        else:
            from evaluate.ner import run as ner_run
            results = ner_run(
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
                use_amp=use_amp,
            )
        all_results["ner"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "swag":
        eval_mode = getattr(args, "eval_mode", "discriminative")
        if eval_mode == "generative":
            from evaluate.diffusion_swag import run as swag_run
            results = swag_run(
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        else:
            from evaluate.swag import run as swag_run
            results = swag_run(
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed,
                use_amp=use_amp,
            )
        all_results["swag"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "xnli":
        from evaluate.xnli import run as xnli_run
        results = xnli_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            languages=args.languages,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )
        all_results["xnli"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "xl_ner":
        from evaluate.cross_lingual_ner import run as xlner_run
        results = xlner_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            languages=args.languages,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )
        all_results["xl_ner"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "pos":
        from evaluate.pos_tagging import run as pos_run
        results = pos_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            languages=args.languages,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )
        all_results["pos"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "mldoc":
        from evaluate.mldoc import run as mldoc_run
        results = mldoc_run(
            checkpoint_dir=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            languages=args.languages,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )
        all_results["mldoc"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "dep":
        from evaluate.dependency_parsing import run as dep_run
        results = dep_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            languages=args.languages,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )
        all_results["dep"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "snli":
        from evaluate.snli import run as snli_run
        results = snli_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )
        all_results["snli"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "race":
        from evaluate.race import run as race_run
        results = race_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )
        all_results["race"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "story_cloze":
        from evaluate.story_cloze import run as sc_run
        results = sc_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )
        all_results["story_cloze"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "scitail":
        from evaluate.scitail import run as scitail_run
        results = scitail_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )
        all_results["scitail"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "perplexity":
        from evaluate.perplexity import run as ppl_run
        results = ppl_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            datasets=args.datasets,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            seed=args.seed,
        )
        all_results["perplexity"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "generation":
        from evaluate.generation_quality import run as gen_run
        results = gen_run(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            noise_schedule=args.noise_schedule,
            temperature=args.temperature,
            batch_size=args.batch_size,
            seed=args.seed,
            skip_mauve=args.skip_mauve,
        )
        all_results["generation"] = results

    # ------------------------------------------------------------------ #
    elif args.command == "all":
        from evaluate.glue               import run as glue_run
        from evaluate.squad              import run as squad_run
        from evaluate.ner                import run as ner_run
        from evaluate.swag               import run as swag_run
        from evaluate.snli               import run as snli_run
        from evaluate.race               import run as race_run
        from evaluate.story_cloze        import run as sc_run
        from evaluate.scitail            import run as scitail_run
        from evaluate.perplexity         import run as ppl_run
        from evaluate.generation_quality import run as gen_run

        common = dict(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_amp=use_amp,
        )

        # ── BERT paper tasks ──────────────────────────────────────────
        glue_results = {}
        for task in args.glue_tasks:
            logger.info(f"\n{'='*60}\nGLUE: {task.upper()}\n{'='*60}")
            try:
                glue_results[task] = glue_run(task_name=task, **common)
            except Exception as e:
                logger.error(f"GLUE {task} failed: {e}")
        all_results["glue"] = glue_results

        for v in (1, 2):
            logger.info(f"\n{'='*60}\nSQuAD v{v}\n{'='*60}")
            try:
                all_results[f"squad_v{v}"] = squad_run(
                    version=v, batch_size=12,
                    **{k: common[k] for k in common if k != "batch_size"},
                )
            except Exception as e:
                logger.error(f"SQuAD v{v} failed: {e}")

        logger.info(f"\n{'='*60}\nCoNLL-2003 NER\n{'='*60}")
        try:
            all_results["ner"] = ner_run(**common)
        except Exception as e:
            logger.error(f"NER failed: {e}")

        logger.info(f"\n{'='*60}\nSWAG\n{'='*60}")
        try:
            all_results["swag"] = swag_run(**{**common, "batch_size": 16})
        except Exception as e:
            logger.error(f"SWAG failed: {e}")

        # ── GPT paper tasks ───────────────────────────────────────────
        logger.info(f"\n{'='*60}\nSNLI\n{'='*60}")
        try:
            all_results["snli"] = snli_run(**common)
        except Exception as e:
            logger.error(f"SNLI failed: {e}")

        logger.info(f"\n{'='*60}\nRACE\n{'='*60}")
        try:
            all_results["race"] = race_run(
                **{**common, "batch_size": 8, "max_seq_length": 512}
            )
        except Exception as e:
            logger.error(f"RACE failed: {e}")

        logger.info(f"\n{'='*60}\nStory Cloze\n{'='*60}")
        try:
            all_results["story_cloze"] = sc_run(**common)
        except Exception as e:
            logger.error(f"Story Cloze failed: {e}")

        logger.info(f"\n{'='*60}\nSciTail\n{'='*60}")
        try:
            all_results["scitail"] = scitail_run(**common)
        except Exception as e:
            logger.error(f"SciTail failed: {e}")

        # ── MDLM diffusion evaluations ────────────────────────────────
        logger.info(f"\n{'='*60}\nPLL Perplexity\n{'='*60}")
        try:
            all_results["perplexity"] = ppl_run(
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                datasets=getattr(args, "ppl_datasets", None),
                seed=args.seed,
            )
        except Exception as e:
            logger.error(f"Perplexity failed: {e}")

        logger.info(f"\n{'='*60}\nGeneration Quality\n{'='*60}")
        try:
            all_results["generation"] = gen_run(
                checkpoint_dir=args.checkpoint,
                output_dir=args.output_dir,
                num_samples=getattr(args, "num_gen_samples", 512),
                seed=args.seed,
                skip_mauve=getattr(args, "skip_mauve", False),
            )
        except Exception as e:
            logger.error(f"Generation quality failed: {e}")

    # ------------------------------------------------------------------ #
    # Write summary
    # ------------------------------------------------------------------ #
    import json
    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results written to {summary_path}")


if __name__ == "__main__":
    main()
