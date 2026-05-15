#!/usr/bin/env python3
"""Patch freelancer tasks to fix /testbed, miniconda, and test.sh issues.

The freelancer dataset (DCAgent/freelancer-projects-sandboxes-ta-rl-gpt-5-nano)
ships with Dockerfiles that use WORKDIR /app, but all test.sh scripts expect
/testbed to exist and many unconditionally run `source /opt/miniconda3/bin/activate`.

Additionally, test.sh scripts use `set -euo pipefail` which causes the script
to abort when an inline Python checker exits non-zero, before the parser.py
block can write reward.txt.

This script patches every task to:
1. Dockerfile: Use WORKDIR /testbed + install miniforge at /opt/miniconda3
2. test.sh: Replace `set -euo pipefail` with `set -uo pipefail` so the
   reward-writing parser.py block always runs

Usage:
    python scripts/daytona/patch_freelancer_testbed.py /path/to/tasks [--dry-run]

    # Or fetch from HuggingFace first:
    python scripts/daytona/patch_freelancer_testbed.py --from-hf DCAgent/freelancer-projects-sandboxes-ta-rl-gpt-5-nano /path/to/output
"""

import argparse
import os
import re
import sys
from pathlib import Path

# New Dockerfile that fixes /testbed and miniconda
NEW_DOCKERFILE = """\
FROM ubuntu:24.04

# System packages
RUN apt-get update && \\
    apt-get install -y python3 python3-pip python3-venv curl git wget && \\
    rm -rf /var/lib/apt/lists/*

# Install miniforge3 at /opt/miniconda3 (many test.sh scripts expect this path)
# Uses conda-forge channel — no Anaconda TOS acceptance needed
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \\
    bash /tmp/miniforge.sh -b -p /opt/miniconda3 && \\
    rm /tmp/miniforge.sh && \\
    /opt/miniconda3/bin/conda create -n testbed python=3.12 -y && \\
    /opt/miniconda3/bin/conda clean -afy

# Create /testbed (all test.sh scripts do cd /testbed)
RUN mkdir -p /testbed /logs/verifier

# Ubuntu 24.04 only has python3; many test.sh scripts call bare 'python'
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /testbed
"""


def patch_test_sh(test_sh: Path) -> bool:
    """Remove -e from set -euo pipefail so reward.txt always gets written.

    Returns True if the file was modified.
    """
    content = test_sh.read_text()
    # Replace 'set -euo pipefail' with 'set -uo pipefail'
    new_content = content.replace("set -euo pipefail", "set -uo pipefail")
    if new_content != content:
        test_sh.write_text(new_content)
        return True
    return False


def patch_tasks(tasks_dir: Path, dry_run: bool = False) -> dict:
    """Patch all task Dockerfiles and test.sh scripts in tasks_dir."""
    stats = {"dockerfile_patched": 0, "testsh_patched": 0, "skipped": 0, "errors": 0, "total": 0}

    task_dirs = sorted(
        d for d in tasks_dir.iterdir()
        if d.is_dir() and d.name.startswith("freelancer_ta_rl_nano-")
    )
    stats["total"] = len(task_dirs)

    for task_dir in task_dirs:
        # Patch Dockerfile
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.exists():
            stats["errors"] += 1
            print(f"  SKIP (no Dockerfile): {task_dir.name}", file=sys.stderr)
            continue

        current = dockerfile.read_text()
        if not ("WORKDIR /testbed" in current and "/opt/miniconda3" in current):
            if dry_run:
                print(f"  WOULD PATCH Dockerfile: {task_dir.name}")
            else:
                dockerfile.write_text(NEW_DOCKERFILE)
            stats["dockerfile_patched"] += 1

        # Patch test.sh — remove -e from set -euo pipefail
        test_sh = task_dir / "tests" / "test.sh"
        if test_sh.exists():
            content = test_sh.read_text()
            if "set -euo pipefail" in content:
                if dry_run:
                    print(f"  WOULD PATCH test.sh: {task_dir.name}")
                else:
                    patch_test_sh(test_sh)
                stats["testsh_patched"] += 1

    return stats


def fetch_from_hf(repo_id: str, output_dir: Path):
    """Download dataset from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {repo_id} to {output_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
    )
    print(f"Downloaded to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Patch freelancer tasks for /testbed + miniconda")
    parser.add_argument("tasks_dir", type=Path, help="Path to tasks directory")
    parser.add_argument("--from-hf", type=str, default=None,
                        help="HuggingFace repo ID to download first")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be patched without writing")
    args = parser.parse_args()

    if args.from_hf:
        fetch_from_hf(args.from_hf, args.tasks_dir)

    if not args.tasks_dir.exists():
        print(f"ERROR: {args.tasks_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Patching tasks in {args.tasks_dir}...")
    stats = patch_tasks(args.tasks_dir, dry_run=args.dry_run)

    print(f"\nResults: {stats['dockerfile_patched']} Dockerfiles patched, "
          f"{stats['testsh_patched']} test.sh patched, "
          f"{stats['errors']} errors, {stats['total']} total")

    if args.dry_run:
        print("(dry run — no files were modified)")


if __name__ == "__main__":
    main()
