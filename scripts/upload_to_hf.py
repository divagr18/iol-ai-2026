"""
Upload submission to HuggingFace Hub.
Usage:
    export HF_TOKEN=hf_xxxx
    python scripts/upload_to_hf.py
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Installing huggingface_hub...")
    os.system("pip install -q huggingface_hub")
    from huggingface_hub import HfApi, create_repo

REPO_ID = os.environ.get("REPO_ID", "divagr18/iol-ai-2026-submission")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN environment variable")
    print("  export HF_TOKEN=hf_xxxx")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

print(f"Creating repo: {REPO_ID}")
create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)

print("Uploading script.py...")
api.upload_file(
    path_or_fileobj="script.py",
    path_in_repo="script.py",
    repo_id=REPO_ID,
    repo_type="model",
)

print("Uploading model weights (this may take a few minutes)...")
api.upload_folder(
    folder_path="qwen2.5-14b-awq",
    path_in_repo="qwen2.5-14b-awq",
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"\n✓ Upload complete!")
print(f"Repo: https://huggingface.co/{REPO_ID}")
print(f"\nNext step: Go to the IOL-AI 2026 competition Space")
print(f"Submit this repo ID: {REPO_ID}")
