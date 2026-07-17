"""
HF Repo Setup Helper

1. Creates a public HF model repo (or uses existing)
2. Downloads model weights to local cache
3. Copies script.py + src modules + weights into a staging dir
4. Uploads to HF Hub using huggingface_hub upload_folder

Usage:
    python scripts/setup_hf_repo.py --repo_id <your-username>/iol-ai-2026-submission \
        --model_id Qwen/Qwen2.5-1.5B-Instruct --quant "" \
        --script script.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo, snapshot_download, upload_folder


def setup_repo(repo_id: str, model_id: str, script_path: Path, src_dir: Path, token: str = None):
    """Create or verify HF repo, download model, stage files, upload."""
    api = HfApi(token=token)

    # 1. Create repo if not exists
    try:
        create_repo(repo_id, repo_type="model", private=False, token=token, exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repo creation/check failed: {e}")
        sys.exit(1)

    # 2. Download model weights
    print(f"Downloading model {model_id} ...")
    local_model = snapshot_download(
        repo_id=model_id,
        token=token,
        local_dir=os.path.join(tempfile.gettempdir(), "iol_model_download"),
        local_dir_use_symlinks=False,
    )
    print(f"Model cached at {local_model}")

    # 3. Stage directory
    staging = Path(tempfile.mkdtemp(prefix="iol_staging_"))
    print(f"Staging directory: {staging}")

    # Copy model weights
    shutil.copytree(local_model, staging / "model", dirs_exist_ok=True)

    # Copy script.py
    shutil.copy2(script_path, staging / "script.py")

    # Copy src analyzers/verifiers/explanation (needed by script.py at runtime)
    if src_dir.exists():
        shutil.copytree(src_dir, staging / "src", dirs_exist_ok=True)

    # Ensure .gitattributes for LFS
    gitattrs = staging / ".gitattributes"
    if not gitattrs.exists():
        gitattrs.write_text(
            "*.bin filter=lfs diff=lfs merge=lfs -text\n"
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
            "*.pt filter=lfs diff=lfs merge=lfs -text\n"
            "*.pth filter=lfs diff=lfs merge=lfs -text\n"
            "*.ckpt filter=lfs diff=lfs merge=lfs -text\n"
        )

    # 4. Upload
    print("Uploading to Hugging Face Hub...")
    upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="IOL-AI 2026 submission update",
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")

    # 5. Cleanup staging
    shutil.rmtree(staging, ignore_errors=True)
    print("Staging cleaned up.")

    return repo_id


def main():
    parser = argparse.ArgumentParser(description="Setup HF submission repo")
    parser.add_argument("--repo_id", required=True, help="HF repo ID, e.g., username/iol-submission")
    parser.add_argument("--model_id", required=True, help="Model ID to download and ship (e.g., cyankiwi/Qwen3.5-4B-AWQ-4bit)")
    parser.add_argument("--script", type=Path, default=Path("script.py"))
    parser.add_argument("--src_dir", type=Path, default=Path("src"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF API token")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: Provide --token or set HF_TOKEN env var")
        sys.exit(1)

    setup_repo(args.repo_id, args.model_id, args.script, args.src_dir, args.token)


if __name__ == "__main__":
    main()
