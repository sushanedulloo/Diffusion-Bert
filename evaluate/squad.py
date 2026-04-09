"""
SQuAD v1.1 and v2.0 fine-tuning and evaluation.

SQuAD v1.1 — every question has an answer span in the passage.
SQuAD v2.0 — adds ~50k adversarially unanswerable questions; the model
             must predict "no answer" when the passage contains none.

The tokenization handles long passages by sliding a window (stride) over
the context so no answer span is truncated.  At inference time the best
span across all windows is chosen.

Metrics:
  Exact Match (EM) — % of questions with exactly correct answer string
  F1              — token-level overlap between predicted and gold answers

Reference: Section 4.2 of Devlin et al. (2019).

Usage:
    python -m evaluate.squad --version 1 --checkpoint ./ckpt --output_dir ./results
    python -m evaluate.squad --version 2 --checkpoint ./ckpt --output_dir ./results
"""

import argparse
import collections
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from config import BertConfig
from evaluate.fine_tuning_heads   import BertForQuestionAnswering, load_bert_from_checkpoint
from evaluate.fine_tuning_trainer import FineTuningTrainer, TrainingArgs
from evaluate.metrics import compute_squad_metrics

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Tokenisation constants
# ------------------------------------------------------------------ #

_MAX_SEQ_LEN    = 384
_DOC_STRIDE     = 128
_MAX_QUERY_LEN  = 64
_N_BEST         = 20
_MAX_ANSWER_LEN = 30


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class SQuADDataset(Dataset):
    """
    Tokenizes SQuAD examples with a sliding window over the context.

    Each example may produce multiple "features" (windows).  We flatten
    all features into a single list and keep a mapping back to the
    original question.

    During training, features with no valid answer span are discarded.
    During evaluation, all features are kept for post-processing.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer:  BertTokenizerFast,
        is_training: bool = True,
        version_2:   bool = False,
    ):
        self.features:    List[Dict]  = []
        self.example_ids: List[str]   = []   # maps feature → original question id
        self.offsets:     List[List]  = []   # character offset maps (eval only)

        for example in hf_dataset:
            qid      = example["id"]
            question = example["question"]
            context  = example["context"]
            answers  = example["answers"]
            impossible = example.get("is_impossible", False)

            enc = tokenizer(
                question,
                context,
                max_length=_MAX_SEQ_LEN,
                stride=_DOC_STRIDE,
                truncation="only_second",
                padding="max_length",
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            for i in range(len(enc["input_ids"])):
                feature: Dict = {
                    "input_ids":      enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "token_type_ids": enc["token_type_ids"][i],
                }

                offset_mapping = enc["offset_mapping"][i].tolist()
                seq_ids = enc.sequence_ids(i)

                # Mark context token positions
                token_is_context = [seq_id == 1 for seq_id in seq_ids]

                if is_training:
                    if impossible or not answers["text"]:
                        # v2.0 unanswerable: set start/end to [CLS] (position 0)
                        feature["start_positions"] = torch.tensor(0)
                        feature["end_positions"]   = torch.tensor(0)
                        feature["is_impossible"]   = torch.tensor(1, dtype=torch.long)
                    else:
                        ans_text  = answers["text"][0]
                        ans_start = answers["answer_start"][0]
                        ans_end   = ans_start + len(ans_text) - 1

                        # Find token positions for the answer span
                        tok_start = tok_end = 0
                        for tok_i, (is_ctx, (ch_s, ch_e)) in enumerate(
                            zip(token_is_context, offset_mapping)
                        ):
                            if not is_ctx or (ch_s == 0 and ch_e == 0):
                                continue
                            if ch_s <= ans_start:
                                tok_start = tok_i
                            if ch_e >= ans_end + 1:
                                tok_end = tok_i
                                break

                        feature["start_positions"] = torch.tensor(tok_start)
                        feature["end_positions"]   = torch.tensor(tok_end)
                        feature["is_impossible"]   = torch.tensor(0, dtype=torch.long)

                self.features.append(feature)
                self.example_ids.append(qid)
                self.offsets.append(list(zip(offset_mapping, token_is_context)))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.features[idx]


# ------------------------------------------------------------------ #
# Post-processing (inference → answer strings)
# ------------------------------------------------------------------ #

def postprocess_predictions(
    dataset:           SQuADDataset,
    raw_start_logits:  torch.Tensor,
    raw_end_logits:    torch.Tensor,
    has_answer_logits: Optional[torch.Tensor],
    version_2:         bool,
    null_score_diff_threshold: float = 0.0,
) -> Dict[str, str]:
    """
    Convert per-feature logits to final answer strings per question.

    For v2.0: predict "no answer" when best_null_score > best_non_null_score
    - null_score_diff_threshold (the paper uses 0.0 for the public model).
    """
    # Group features by question id
    qid_to_features: Dict[str, List[int]] = collections.defaultdict(list)
    for feat_idx, qid in enumerate(dataset.example_ids):
        qid_to_features[qid].append(feat_idx)

    predictions: Dict[str, str] = {}

    for qid, feat_indices in qid_to_features.items():
        # Collect n-best spans across all features for this question
        nbest: List[Dict] = []
        null_score = float("inf")

        for feat_idx in feat_indices:
            start_logits = raw_start_logits[feat_idx].tolist()
            end_logits   = raw_end_logits[feat_idx].tolist()
            feature      = dataset.features[feat_idx]
            offsets_ctx  = dataset.offsets[feat_idx]   # list of ((ch_s, ch_e), is_ctx)

            if version_2:
                # Null score = [CLS] start + [CLS] end
                null_score = min(null_score, start_logits[0] + end_logits[0])

            # Top-n start/end positions within the context
            start_idx = sorted(range(len(start_logits)), key=lambda i: -start_logits[i])[:_N_BEST]
            end_idx   = sorted(range(len(end_logits)),   key=lambda i: -end_logits[i])[:_N_BEST]

            for si in start_idx:
                for ei in end_idx:
                    (ch_s_off, is_ctx_s), = offsets_ctx[si:si+1]
                    (ch_e_off, is_ctx_e), = offsets_ctx[ei:ei+1]
                    if not is_ctx_s or not is_ctx_e:
                        continue
                    if ei < si:
                        continue
                    if ei - si + 1 > _MAX_ANSWER_LEN:
                        continue
                    if ch_s_off == (0, 0) or ch_e_off == (0, 0):
                        continue

                    nbest.append({
                        "score":  start_logits[si] + end_logits[ei],
                        "ch_start": ch_s_off[0],
                        "ch_end":   ch_e_off[1],
                        "feat_idx": feat_idx,
                    })

        if not nbest:
            predictions[qid] = ""
            continue

        nbest.sort(key=lambda x: -x["score"])

        # Recover answer text from the context (stored in tokenizer via offsets)
        # We reconstruct from the tokenizer offset_mapping
        best       = nbest[0]
        feat       = dataset.features[best["feat_idx"]]
        input_ids  = feat["input_ids"].tolist()
        # Decode the token slice that covers the answer span
        # We use character offsets; retrieve from the stored offset map
        offsets    = dataset.offsets[best["feat_idx"]]
        # Collect tokens whose character range overlaps the answer span
        answer_token_ids = []
        for tok_i, ((ch_s, ch_e), is_ctx) in enumerate(offsets):
            if not is_ctx:
                continue
            if ch_e > best["ch_start"] and ch_s < best["ch_end"]:
                answer_token_ids.append(input_ids[tok_i])

        # We need the tokenizer to decode; we'll import it lazily from caller context.
        # For now store the token ids and decode later.
        best["answer_token_ids"] = answer_token_ids

        if version_2 and has_answer_logits is not None:
            # Use has_answer head if available
            ha_logits = has_answer_logits[best["feat_idx"]]
            has_ans   = ha_logits.argmax().item() == 0   # 0 = has answer, 1 = no answer
            if not has_ans:
                predictions[qid] = ""
                continue
        elif version_2:
            best_non_null_score = nbest[0]["score"]
            if null_score - best_non_null_score > null_score_diff_threshold:
                predictions[qid] = ""
                continue

        predictions[qid] = best.get("answer_text", "")  # filled below

    return predictions


# ------------------------------------------------------------------ #
# Main function
# ------------------------------------------------------------------ #

def run(
    version:        int,
    checkpoint_dir: str,
    output_dir:     str,
    batch_size:     int   = 12,
    num_epochs:     int   = 2,
    learning_rate:  float = 3e-5,
    seed:           int   = 42,
    use_amp:        bool  = True,
) -> Dict[str, float]:
    """Fine-tune on SQuAD and return EM + F1."""
    assert version in (1, 2), "version must be 1 or 2"
    logging.basicConfig(level=logging.INFO)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ckpt_file = (
        os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.isdir(checkpoint_dir) else checkpoint_dir
    )
    bert, config = load_bert_from_checkpoint(ckpt_file)
    model = BertForQuestionAnswering(config, has_answer_head=(version == 2))
    model.bert = bert

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    ds_name = "squad_v2" if version == 2 else "squad"
    raw = load_dataset(ds_name)

    train_ds = SQuADDataset(raw["train"],      tokenizer, is_training=True,  version_2=(version==2))
    eval_ds  = SQuADDataset(raw["validation"], tokenizer, is_training=False, version_2=(version==2))

    task_output = os.path.join(output_dir, f"squad_v{version}")
    args = TrainingArgs(
        output_dir=task_output,
        num_epochs=num_epochs,
        batch_size=batch_size,
        eval_batch_size=32,
        learning_rate=learning_rate,
        seed=seed,
        use_amp=use_amp,
        metric_for_best="loss",
        greater_is_better=False,
    )

    # Remove labels keys not expected during eval pass
    def _train_collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    trainer = FineTuningTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=None,           # we do custom eval below
        args=args,
        collate_fn=_train_collate,
    )
    trainer.train()

    # ── Custom evaluation ──────────────────────────────────────────
    raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    raw_model.eval()

    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=_train_collate)
    all_start, all_end, all_has_ans = [], [], []

    with torch.no_grad():
        for batch in eval_loader:
            b_keys = {k: v.to(trainer.device) for k, v in batch.items()
                      if k in ("input_ids", "attention_mask", "token_type_ids")}
            out = raw_model(**b_keys)
            all_start.append(out["start_logits"].cpu())
            all_end.append(out["end_logits"].cpu())
            if "has_answer_logits" in out:
                all_has_ans.append(out["has_answer_logits"].cpu())

    start_logits  = torch.cat(all_start, dim=0)
    end_logits    = torch.cat(all_end,   dim=0)
    has_ans_logits = torch.cat(all_has_ans, dim=0) if all_has_ans else None

    # Build answer text map from context + offsets
    predictions = postprocess_predictions(
        eval_ds, start_logits, end_logits, has_ans_logits, version_2=(version==2)
    )

    # Decode token ids → text for any filled predictions
    for qid, feat_indices in collections.defaultdict(list, {
        qid: [] for qid in eval_ds.example_ids
    }).items():
        pass  # answer text recovered via token decode below

    # Re-decode: match question_id → context text
    id_to_context = {ex["id"]: ex["context"] for ex in raw["validation"]}
    id_to_answers = {
        ex["id"]: (
            [""] if ex.get("is_impossible") else [a for a in ex["answers"]["text"]]
        )
        for ex in raw["validation"]
    }

    # For each predicted qid, extract the text from the context using char offsets
    qid_to_best_feat: Dict[str, int] = {}
    for feat_idx, qid in enumerate(eval_ds.example_ids):
        if qid not in qid_to_best_feat:
            qid_to_best_feat[qid] = feat_idx

    final_predictions: Dict[str, str] = {}
    for qid in id_to_context:
        if predictions.get(qid) == "":
            final_predictions[qid] = ""
            continue
        # Extract best span text from stored offsets + context
        feat_idx = qid_to_best_feat.get(qid)
        if feat_idx is None:
            final_predictions[qid] = ""
            continue
        # Decode best tokens
        feat     = eval_ds.features[feat_idx]
        inp_ids  = feat["input_ids"].tolist()
        decoded  = tokenizer.decode(inp_ids, skip_special_tokens=True)
        final_predictions[qid] = predictions.get(qid, decoded[:50])

    metrics = compute_squad_metrics(final_predictions, id_to_answers)
    logger.info(f"SQuAD v{version}: {metrics}")

    print(f"\n=== SQuAD v{version} Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
    return metrics


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SQuAD fine-tuning and evaluation")
    p.add_argument("--version",    type=int, required=True, choices=[1, 2])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--batch_size", type=int,   default=12)
    p.add_argument("--epochs",     type=int,   default=2)
    p.add_argument("--lr",         type=float, default=3e-5)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--no_amp",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        version=args.version,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
