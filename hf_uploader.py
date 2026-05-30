"""
hf_uploader.py — Upload BERT checkpoints to a Hugging Face Hub model repo.

Mirrors gdrive_uploader.py but uses the HF Hub. Each call uploads the
``checkpoint.pt`` inside a checkpoint sub-directory to the repo as
``checkpoint_<tag>.pt``, where ``<tag>`` is the directory's basename
(e.g. ``checkpoint_step_0001000`` → ``checkpoint_step_0001000.pt``).

Authentication: uses the cached HF token from ``huggingface_hub.login()``
unless ``hf_token`` is passed explicitly.

Usage
-----
1. ``huggingface-cli login`` (one-time) — or call ``login(token=…)``
   inside a notebook.
2. Pass ``--hf_repo_id <username>/<repo-name>`` to ``train.py``.
   The repo is auto-created (private by default) if it doesn't exist.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def upload_checkpoint(
    checkpoint_dir: str,
    repo_id:        str,
    hf_token:       Optional[str] = None,
    private:        bool          = True,
) -> None:
    """
    Upload ``checkpoint_dir/checkpoint.pt`` to ``repo_id`` on the HF Hub
    as ``checkpoint_<tag>.pt``.  Auto-creates the repo if missing.

    Args:
        checkpoint_dir: local path containing checkpoint.pt
        repo_id:        HF repo id, e.g. ``"username/ur-mdlm-mini"``
        hf_token:       optional explicit token (otherwise uses cached login)
        private:        create the repo as private when first creating it

    Failures are logged as warnings — training is never interrupted.
    """
    try:
        from huggingface_hub import HfApi, create_repo

        ckpt_file = os.path.join(checkpoint_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_file):
            logger.warning(f"HF upload skipped — file not found: {ckpt_file}")
            return

        tag           = os.path.basename(checkpoint_dir.rstrip("/"))
        file_size_mb  = os.path.getsize(ckpt_file) / 1e6
        path_in_repo  = f"{tag}.pt"

        # Create the repo if it doesn't exist (no-op if it does).
        create_repo(
            repo_id    = repo_id,
            token      = hf_token,
            private    = private,
            exist_ok   = True,
            repo_type  = "model",
        )

        logger.info(
            f"HF: uploading {path_in_repo} ({file_size_mb:.0f} MB) "
            f"to {repo_id} …"
        )

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj = ckpt_file,
            path_in_repo    = path_in_repo,
            repo_id         = repo_id,
            repo_type       = "model",
            commit_message  = f"Upload {tag}",
        )

        logger.info(f"HF upload done: {repo_id}/{path_in_repo}")

    except Exception as exc:
        logger.warning(f"HF upload failed (training continues): {exc}")


def upload_results_dir(
    results_dir:  str,
    repo_id:      str,
    hf_token:     Optional[str] = None,
    private:      bool          = True,
    path_prefix:  str           = "results",
) -> None:
    """
    Upload every file in ``results_dir`` to ``repo_id`` under
    ``<path_prefix>/<filename>``.  Useful for pushing eval JSONs alongside
    the checkpoints.
    """
    try:
        from huggingface_hub import HfApi, create_repo

        if not os.path.isdir(results_dir):
            logger.warning(f"HF upload_results skipped — not a dir: {results_dir}")
            return

        create_repo(
            repo_id   = repo_id,
            token     = hf_token,
            private   = private,
            exist_ok  = True,
            repo_type = "model",
        )

        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path    = results_dir,
            path_in_repo   = path_prefix,
            repo_id        = repo_id,
            repo_type      = "model",
            commit_message = f"Upload {path_prefix}/",
        )
        logger.info(f"HF: uploaded {results_dir}/ → {repo_id}/{path_prefix}/")

    except Exception as exc:
        logger.warning(f"HF upload_results failed: {exc}")
