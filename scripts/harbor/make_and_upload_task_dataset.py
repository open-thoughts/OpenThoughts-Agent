#!/usr/bin/env python3
"""
Convert a local directory of Harbor tasks into a Parquet dataset and upload it
to the Hugging Face Hub as a dataset repository.

Usage example:

  python -m scripts.harbor.make_and_upload_task_dataset \
    --repo_id my-org/my-tasks-dataset \
    --tasks_dir /path/to/tasks \
    --private

Requirements:
- The directory at --tasks_dir should contain one task per subdirectory.
- Each task directory should include an instruction.md (task marker).
- `huggingface_hub` and `pyarrow` must be installed.
- Authentication via HF_TOKEN env var or --token.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# Local converter utilities
from scripts.harbor import tasks_parquet_converter as tpc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make a Parquet dataset from tasks and upload to Hugging Face Hub",
    )
    p.add_argument(
        "--repo_id",
        required=True,
        help="Target HF dataset repo (e.g. 'org/name')",
    )
    p.add_argument(
        "--tasks_dir",
        required=True,
        help="Local path to directory containing task subdirectories",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF access token (otherwise uses HF_TOKEN env var)",
    )
    p.add_argument(
        "--commit_message",
        default="Upload tasks parquet",
        help="Commit message used for the upload",
    )
    p.add_argument(
        "--delete_existing",
        action="store_true",
        help="Delete existing files in the repo before upload",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tasks_dir = Path(args.tasks_dir).expanduser().resolve()
    if not tasks_dir.exists() or not tasks_dir.is_dir():
        raise FileNotFoundError(f"tasks_dir does not exist or is not a directory: {tasks_dir}")

    # 1) Build a temporary folder with tasks.parquet
    print(f"[make-upload] Converting tasks at: {tasks_dir}")
    parquet_tmp_dir = Path(tpc.convert_to_parquet(str(tasks_dir)))
    parquet_file = parquet_tmp_dir / "tasks.parquet"
    if not parquet_file.exists():
        raise RuntimeError(f"Parquet conversion did not produce tasks.parquet at {parquet_file}")
    print(f"[make-upload] Parquet created: {parquet_file}")

    # 2) Create (or reuse) the HF dataset repo
    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=bool(args.private),
        token=token,
        exist_ok=True,
    )
    print(f"[make-upload] HF repo ready: https://huggingface.co/datasets/{args.repo_id}")

    # 3) Upload the parquet folder (single file) to the repo
    print("[make-upload] Uploading parquet to HF Hub...")
    delete_patterns = "*" if args.delete_existing else None
    result = api.upload_folder(
        folder_path=str(parquet_tmp_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
        delete_patterns=delete_patterns,
    )
    print(f"[make-upload] Upload complete: {result}")
    print(f"[make-upload] Dataset URL: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

