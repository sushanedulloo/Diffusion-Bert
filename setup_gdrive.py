"""
setup_gdrive.py — One-time Google Drive OAuth setup.

Run this script ONCE on a machine that has a web browser (your laptop/desktop).
It will open a browser tab, ask you to sign in with Google, grant Drive access,
then save a token.pickle file.  Copy that file to your college server.

Steps
-----
1.  Go to https://console.cloud.google.com/
2.  Create a project (or pick an existing one).
3.  Enable the Google Drive API.
4.  Create OAuth 2.0 credentials → Desktop app → download credentials.json.
5.  Run:  python setup_gdrive.py
6.  Copy the generated token.pickle to your server.

Usage on server
---------------
python train.py \\
    --gdrive_folder_id <YOUR_FOLDER_ID> \\
    --gdrive_token_path /path/to/token.pickle \\
    ...other args...
"""

import os
import pickle
import sys

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Missing dependency.  Run:  pip install google-auth-oauthlib")
        sys.exit(1)

    credentials_path = (
        input("Path to credentials.json [credentials.json]: ").strip()
        or "credentials.json"
    )
    if not os.path.exists(credentials_path):
        print(f"ERROR: File not found: {credentials_path}")
        sys.exit(1)

    token_path = (
        input("Where to save token.pickle [token.pickle]: ").strip()
        or "token.pickle"
    )

    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
    creds = flow.run_local_server(port=0)

    with open(token_path, "wb") as f:
        pickle.dump(creds, f)

    print(f"\nDone!  Token saved to: {token_path}")
    print("\nNext steps:")
    print(f"  1. Copy '{token_path}' to your college server.")
    print("  2. Get your Google Drive folder ID from the browser URL:")
    print("       https://drive.google.com/drive/folders/<FOLDER_ID>")
    print("  3. Run training with:")
    print("       python train.py \\")
    print(f"           --gdrive_folder_id <FOLDER_ID> \\")
    print(f"           --gdrive_token_path {token_path} \\")
    print("           --num_epochs 5 \\")
    print("           ... other args ...")


if __name__ == "__main__":
    main()
