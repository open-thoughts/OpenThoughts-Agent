#!/usr/bin/env python3
"""
Generate patched R2E-Gym tasks directly from HuggingFace → task directories.

Combines generation + patching in one step. Loads R2E-Gym/R2E-Gym-Subset from
HuggingFace and writes Harbor-format task directories using 10 shared Dockerfiles
(one per repo) instead of 4,578 unique pre-built images.

Does NOT require GCS credentials or h5py — runs locally with just:
    pip install datasets

Agent setup per task is reduced to a single fast step:
    git checkout {commit_hash}   (repo + deps already in image)

Test infrastructure (r2e_tests/, run_tests.sh) must be extracted separately
via the --extract-tests flag in patch_r2egym_tasks.py (requires docker).
Until then, test.sh falls back to running tests by name directly via pytest.

Usage:
    python generate_r2egym_patched.py --output-dir ./r2egym_patched
    python generate_r2egym_patched.py --output-dir ./r2egym_patched --limit 100
    python generate_r2egym_patched.py --output-dir ./r2egym_patched --repo pandas
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-use Dockerfile configs from the patcher
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))
from patch_r2egym_tasks import (
    _build_dockerfile,
    _repo_name_from_metadata,
    _REPO_GITHUB_URL,
    SETUP_PREAMBLE_TEMPLATE,
    TEST_SH_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Task file templates
# ---------------------------------------------------------------------------

TASK_TOML = """\
version = "1.0"

[agent]
timeout_sec = 900.0

[metadata]
author_name = "OpenThoughts-Agent"
author_email = "r2egym@openthoughts-agent.invalid"
difficulty = "hard"
category = "software-engineering"
tags = ["r2egym", "code-repair", "bug-fixing"]

[verifier]
restart_environment = false
timeout_sec = 720.0
"""

SOLVE_SH_TEMPLATE = """\
#!/bin/bash
set -euo pipefail
# Oracle solution: checkout the specific commit (which IS the fixed state).
# R2E-Gym tasks use the fixed commit as base; the bug is introduced by the
# test infrastructure expecting pre-fix behavior.
# The oracle just ensures the environment is correctly set up.
cd /testbed && git checkout {base_commit}
"""

INSTRUCTION_TEMPLATE = """\
## Environment Setup (complete this step first)

```bash
cd /testbed && git checkout {base_commit}
```

---

<uploaded_files>
/testbed
</uploaded_files>

I've uploaded a Python code repository in the directory /testbed at commit `{base_commit}`.
Consider the following issue:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the
requirements specified in the issue are met?

The development environment is already set up — the repo is pre-cloned and all
dependencies are installed. Your only setup step is the `git checkout` above.

Follow these steps:

1. **Explore** the repository to understand the codebase and locate the issue.
2. **Reproduce** the issue by creating a script at `/testbed/reproduce_issue.py`.
3. **Fix** the issue with minimal changes to non-test files.
4. **Verify** your fix passes the existing tests.

Be thorough — quality matters more than brevity.
"""


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------

def _short_repo_name(repo_name: str) -> str:
    """pandas-dev/pandas → pandas"""
    return repo_name.split("/")[-1].lower()


def build_task(instance: dict, output_dir: Path, task_id: str) -> Path | None:
    """
    Write a single patched task directory.
    Returns the task path, or None if the repo is unrecognised.
    """
    repo_name_full = instance.get("repo_name", "")
    short = _short_repo_name(repo_name_full)

    # Map to canonical repo key used in Dockerfile configs
    repo_key = next(
        (k for k in _REPO_GITHUB_URL
         if short == k or short.startswith(k) or k.startswith(short)),
        None,
    )
    if repo_key is None:
        print(f"  SKIP {task_id}: unknown repo '{repo_name_full}'")
        return None

    base_commit = instance.get("commit_hash") or instance.get("base_commit", "HEAD")
    problem_statement = instance.get("problem_statement") or instance.get("prompt", "")
    docker_image = instance.get("docker_image", "")
    expected_output_json = instance.get("expected_output_json", "{}")

    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # --- instruction.md ---
    (task_dir / "instruction.md").write_text(
        INSTRUCTION_TEMPLATE.format(
            base_commit=base_commit,
            problem_statement=problem_statement.strip(),
        )
    )

    # --- task.toml ---
    (task_dir / "task.toml").write_text(TASK_TOML)

    # --- environment/Dockerfile (one of 10, keyed by repo) ---
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(repo_key))

    # --- setup_files/metadata.json (per-task data; uploaded by Harbor) ---
    setup_dir = task_dir / "setup_files"
    setup_dir.mkdir(exist_ok=True)
    metadata = {
        "instance_id": instance.get("instance_id") or instance.get("docker_image", task_id),
        "docker_image": docker_image,
        "base_commit": base_commit,
        "repo_name": repo_name_full,
        "problem_statement": problem_statement,
        "expected_output_json": expected_output_json,
        "source": "r2egym",
    }
    (setup_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # --- tests/test.sh ---
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(TEST_SH_TEMPLATE)
    test_sh.chmod(test_sh.stat().st_mode | stat.S_IEXEC)

    # --- solution/solve.sh (oracle) ---
    solution_dir = task_dir / "solution"
    solution_dir.mkdir(exist_ok=True)
    solve_sh = solution_dir / "solve.sh"
    solve_sh.write_text(SOLVE_SH_TEMPLATE.format(base_commit=base_commit))
    solve_sh.chmod(solve_sh.stat().st_mode | stat.S_IEXEC)

    return task_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate patched R2E-Gym tasks from HuggingFace → task dirs",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write task folders into",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Generate only the first N tasks (useful for testing)",
    )
    parser.add_argument(
        "--repo", type=str, default=None,
        help="Only generate tasks for this repo (e.g. pandas, numpy)",
    )
    parser.add_argument(
        "--dataset", type=str, default="R2E-Gym/R2E-Gym-Subset",
        help="HuggingFace dataset to load (default: R2E-Gym/R2E-Gym-Subset)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets not installed — run: pip install datasets")

    print(f"Loading {args.dataset} from HuggingFace...")
    ds = load_dataset(args.dataset, split="train")
    print(f"Loaded {len(ds)} instances")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = skipped = 0
    repo_counts: dict[str, int] = {}

    for i, instance in enumerate(ds):
        repo_name_full = instance.get("repo_name", "")
        short = _short_repo_name(repo_name_full)

        # Filter by repo if requested
        if args.repo and short != args.repo.lower():
            continue

        # Apply limit
        if args.limit and generated >= args.limit:
            break

        task_id = f"r2egym-{i:04d}"
        result = build_task(instance, args.output_dir, task_id)

        if result is None:
            skipped += 1
        else:
            generated += 1
            repo_counts[short] = repo_counts.get(short, 0) + 1

        if generated % 100 == 0 and generated > 0:
            print(f"  Generated {generated} tasks...")

    print(f"\nDone: {generated} tasks written to {args.output_dir}")
    print(f"Skipped (unknown repo): {skipped}")
    print("\nPer-repo breakdown:")
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        print(f"  {repo}: {count}")

    # Report unique Dockerfiles
    unique_dfs: set[str] = set()
    for td in args.output_dir.iterdir():
        df = td / "environment" / "Dockerfile"
        if df.exists():
            unique_dfs.add(df.read_text())
    print(f"\nUnique Dockerfiles: {len(unique_dfs)} (expected: up to 10)")
    print("\nNext step: extract r2e_tests + run_tests.sh from original docker images:")
    print(f"  python patch_r2egym_tasks.py {args.output_dir} --extract-tests --parallel 8")


if __name__ == "__main__":
    main()
