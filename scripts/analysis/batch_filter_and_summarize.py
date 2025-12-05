#!/usr/bin/env python3
"""
Batch helper: iterate subdirectories under a root, derive a HF repo id from
each directory name, run filter_latest_episodes.py to produce a JSONL per repo,
then summarize each JSONL via summarize_conversations.py.

Defaults:
- root: /scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/jobs/to_upload
- out_dir: ../eval-jsonl

Repo id mapping:
- By default uses the folder name directly. If that fails to load, it tries
  replacing the first underscore with a slash (org_repo -> org/repo).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import os

from datasets import load_dataset, get_dataset_split_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch filter + summarize across subdirectories")
    p.add_argument(
        "--root",
        default="/scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/jobs/to_upload",
        help="Root directory containing subdirectories to process",
    )
    p.add_argument(
        "--out_dir",
        default="../eval-jsonl",
        help="Directory where JSONL outputs will be written",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip JSONL generation when the output file already exists",
    )
    p.add_argument(
        "--use_job_dir",
        action="store_true",
        default=True,
        help=(
            "When set (default), pass each subdirectory as --job-dir to filter_latest_episodes.py. "
            "Disable to read from Hub by derived repo id."
        ),
    )
    return p.parse_args()


def _guess_repo_id(name: str) -> str | None:
    """Return a probable HF repo id for a directory name.

    Strategy:
      1) Try as-is
      2) If that fails: replace first underscore with a slash
    """
    # Try as-is
    try:
        get_dataset_split_names(name)
        return name
    except Exception:
        pass

    # Try org_repo -> org/repo
    if "_" in name:
        candidate = name.replace("_", "/", 1)
        try:
            get_dataset_split_names(candidate)
            return candidate
        except Exception:
            pass

    return None


def _pick_split(repo_id: str) -> str | None:
    try:
        splits = get_dataset_split_names(repo_id)
    except Exception:
        return None
    if "train" in splits:
        return "train"
    return splits[0] if splits else None


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[batch] Root: {root}")
    print(f"[batch] Out:  {out_dir}")

    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        name = d.name
        out_jsonl = out_dir / f"{name}.jsonl"
        if args.skip_existing and out_jsonl.exists():
            print(f"[skip] Exists: {out_jsonl}")
        elif args.use_job_dir:
            cmd = [
                "python",
                "scripts/analysis/filter_latest_episodes.py",
                "--job-dir",
                str(d),
                "--output-jsonl",
                str(out_jsonl),
            ]
            print(f"[run local] {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[error] filter_latest_episodes failed for job dir {d}: {e}")
                continue
        else:
            repo_id = _guess_repo_id(name)
            if not repo_id:
                print(f"[warn] Could not derive repo id for {name}; skipping")
                continue
            split = _pick_split(repo_id)
            if not split:
                print(f"[warn] No splits available for {repo_id}; skipping")
                continue
            cmd = [
                "python",
                "scripts/analysis/filter_latest_episodes.py",
                repo_id,
                "--split",
                split,
                "--output-jsonl",
                str(out_jsonl),
            ]
            print(f"[run hub] {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[error] filter_latest_episodes failed for {repo_id}: {e}")
                continue

    # Summarize each JSONL
    # Prepare env for tokenizers to avoid HPC thread pool issues
    summarize_env = os.environ.copy()
    summarize_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    summarize_env.setdefault("RAYON_NUM_THREADS", "1")

    for f in sorted(out_dir.glob("*.jsonl")):
        print(f"[summary] {f}")
        subprocess.run([
            "python", "scripts/analysis/summarize_conversations.py", str(f)
        ], check=False, env=summarize_env)

    print("[batch] Done.")


if __name__ == "__main__":
    main()
