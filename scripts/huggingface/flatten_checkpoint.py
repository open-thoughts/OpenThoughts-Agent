#!/usr/bin/env python3
"""Flatten a HuggingFace model repo that stores weights under checkpoints/.

Clones the repo, copies the latest checkpoint (highest step number) to the
repo root, deletes the checkpoints/ directory, commits, pushes, and cleans
up the local clone.

Usage:
    python scripts/huggingface/flatten_checkpoint.py laion/rl__24GPU_base__exp_rpt_codeelo-v2__Qwen3-8B

    # Specify a particular checkpoint step
    python scripts/huggingface/flatten_checkpoint.py laion/my-model --step 5

    # Keep local clone after pushing
    python scripts/huggingface/flatten_checkpoint.py laion/my-model --keep-local

    # Dry run — clone and flatten but don't push or delete
    python scripts/huggingface/flatten_checkpoint.py laion/my-model --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    import os

    print(f"  $ {' '.join(cmd)}")
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=False, env=run_env)


def find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    """Return the checkpoint subdirectory with the highest step number."""
    step_dirs = []
    for child in checkpoints_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.search(r"(\d+)", child.name)
        if match:
            step_dirs.append((int(match.group(1)), child))
    if not step_dirs:
        raise FileNotFoundError(f"No checkpoint step directories found in {checkpoints_dir}")
    step_dirs.sort(key=lambda x: x[0])
    return step_dirs[-1][1]


def flatten_checkpoint(
    repo_id: str,
    step: int | None = None,
    keep_local: bool = False,
    dry_run: bool = False,
    clone_dir: Path | None = None,
) -> None:
    repo_url = f"https://huggingface.co/{repo_id}"
    repo_name = repo_id.split("/")[-1]

    # Determine clone location
    if clone_dir is not None:
        work_dir = clone_dir / repo_name
    else:
        work_dir = Path(tempfile.mkdtemp()) / repo_name

    print(f"Cloning {repo_url} (LFS pointers only)...")
    run(["git", "clone", repo_url, str(work_dir)],
        check=True, env={"GIT_LFS_SKIP_SMUDGE": "1"})

    # Set GIT_LFS_SKIP_SMUDGE for the clone, then pull selectively
    checkpoints_dir = work_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        print(f"Error: No checkpoints/ directory found in {repo_id}")
        print("Repo root contents:", [p.name for p in work_dir.iterdir()])
        sys.exit(1)

    # Find target checkpoint
    if step is not None:
        target = None
        for child in checkpoints_dir.iterdir():
            if child.is_dir() and str(step) in child.name:
                target = child
                break
        if target is None:
            available = sorted(p.name for p in checkpoints_dir.iterdir() if p.is_dir())
            print(f"Error: step {step} not found. Available: {available}")
            sys.exit(1)
    else:
        target = find_latest_checkpoint(checkpoints_dir)

    print(f"Using checkpoint: {target.name}")

    # Pull LFS files for the target checkpoint only
    print(f"Pulling LFS files for {target.name}...")
    run(["git", "lfs", "pull", f"--include=checkpoints/{target.name}/*"], cwd=work_dir)

    # Copy checkpoint files to repo root
    print("Copying checkpoint files to repo root...")
    for src_file in target.iterdir():
        if src_file.is_file():
            dst = work_dir / src_file.name
            shutil.copy2(src_file, dst)
            print(f"  {src_file.name}")

    # Ensure large files at root are LFS-tracked (HF rejects >10MB non-LFS)
    print("Checking for large files that need LFS tracking...")
    for src_file in target.iterdir():
        if src_file.is_file() and src_file.stat().st_size > 10 * 1024 * 1024:
            run(["git", "lfs", "track", src_file.name], cwd=work_dir)

    # Remove checkpoints directory
    print("Removing checkpoints/ from git...")
    run(["git", "rm", "-rf", "checkpoints/"], cwd=work_dir)

    # Stage new files (including .gitattributes if modified)
    print("Staging new files...")
    run(["git", "add", "."], cwd=work_dir)

    # Commit
    commit_msg = f"Flatten: move {target.name} checkpoint to repo root, remove checkpoints/"
    run(["git", "commit", "-m", commit_msg], cwd=work_dir)

    if dry_run:
        print(f"\nDRY RUN — not pushing. Local clone at: {work_dir}")
        return

    # Push
    print("Pushing to HuggingFace (this may take a while for large models)...")
    run(["git", "push", "origin", "main"], cwd=work_dir)
    print(f"\nDone! {repo_url} updated.")

    # Cleanup
    if not keep_local:
        print(f"Cleaning up local clone at {work_dir}...")
        shutil.rmtree(work_dir.parent if clone_dir is None else work_dir)
        print("Cleanup complete.")
    else:
        print(f"Local clone kept at: {work_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten a HuggingFace model repo by moving the latest checkpoint to the root.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g. laion/my-model)")
    parser.add_argument("--step", type=int, default=None, help="Specific checkpoint step to use (default: latest)")
    parser.add_argument("--keep-local", action="store_true", help="Don't delete the local clone after pushing")
    parser.add_argument("--dry-run", action="store_true", help="Clone and flatten but don't push")
    parser.add_argument("--clone-dir", type=str, default=None, help="Directory to clone into (default: temp dir)")

    args = parser.parse_args()
    clone_dir = Path(args.clone_dir) if args.clone_dir else None

    flatten_checkpoint(
        repo_id=args.repo_id,
        step=args.step,
        keep_local=args.keep_local,
        dry_run=args.dry_run,
        clone_dir=clone_dir,
    )


if __name__ == "__main__":
    main()
