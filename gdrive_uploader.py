"""
gdrive_uploader.py — Upload BERT checkpoints to Google Drive.

Always uploads as 'latest_checkpoint.pt', overwriting the previous file
in Drive so only one checkpoint exists there at any time.

Usage
-----
1. Run setup_gdrive.py ONCE on a machine with a browser to get token.pickle.
2. Copy token.pickle to your server alongside this file.
3. Pass --gdrive_folder_id and --gdrive_token_path to train.py.

The folder ID is the last part of the Google Drive folder URL:
    https://drive.google.com/drive/folders/<FOLDER_ID>
"""

import os
import pickle
import logging

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_FILENAME = "latest_checkpoint.pt"


def _get_service(token_path: str):
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    if not os.path.exists(token_path):
        raise FileNotFoundError(
            f"Google Drive token not found at '{token_path}'. "
            "Run setup_gdrive.py on a machine with a browser first."
        )

    with open(token_path, "rb") as f:
        creds = pickle.load(f)

    if creds.expired and creds.refresh_token:
        logger.info("Refreshing Google Drive token ...")
        creds.refresh(Request())
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    if not creds.valid:
        raise RuntimeError(
            f"Google Drive credentials at '{token_path}' are invalid. "
            "Re-run setup_gdrive.py to generate a fresh token."
        )

    return build("drive", "v3", credentials=creds)


def upload_checkpoint(
    checkpoint_dir: str,
    folder_id: str,
    token_path: str = "token.pickle",
) -> None:
    """
    Upload checkpoint.pt from checkpoint_dir to Google Drive as
    'latest_checkpoint.pt', overwriting any previous version.

    Only one file ever exists in the Drive folder, so storage stays minimal.
    Failures are logged as warnings — training is never interrupted.
    """
    try:
        from googleapiclient.http import MediaFileUpload

        ckpt_file = os.path.join(checkpoint_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_file):
            logger.warning(f"GDrive upload skipped — file not found: {ckpt_file}")
            return

        file_size_mb = os.path.getsize(ckpt_file) / 1e6
        step_tag = os.path.basename(checkpoint_dir.rstrip("/"))
        service = _get_service(token_path)

        # Find existing 'latest_checkpoint.pt' in the folder (if any)
        query = (
            f"name='{DRIVE_FILENAME}' and '{folder_id}' in parents and trashed=false"
        )
        existing = service.files().list(q=query, fields="files(id,name)").execute()

        media = MediaFileUpload(
            ckpt_file,
            mimetype="application/octet-stream",
            resumable=True,
            chunksize=16 * 1024 * 1024,  # 16 MB chunks
        )

        if existing["files"]:
            file_id = existing["files"][0]["id"]
            logger.info(
                f"GDrive: overwriting {DRIVE_FILENAME} with {step_tag} "
                f"({file_size_mb:.0f} MB) ..."
            )
            service.files().update(
                fileId=file_id,
                media_body=media,
            ).execute()
        else:
            logger.info(
                f"GDrive: uploading {DRIVE_FILENAME} [{step_tag}] "
                f"({file_size_mb:.0f} MB) ..."
            )
            metadata = {"name": DRIVE_FILENAME, "parents": [folder_id]}
            service.files().create(
                body=metadata,
                media_body=media,
                fields="id",
            ).execute()

        logger.info(f"GDrive upload done: {DRIVE_FILENAME} [{step_tag}]")

    except Exception as exc:
        logger.warning(f"GDrive upload failed (training continues): {exc}")
