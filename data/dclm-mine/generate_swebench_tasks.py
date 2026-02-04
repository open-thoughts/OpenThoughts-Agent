#!/usr/bin/env python3
"""
Generate tasks from SWE-bench dataset (2,294 multi-file GitHub issues).

SWE-bench contains real GitHub issues from popular Python projects (Django,
Requests, Scikit-learn, etc.) that require multi-file changes to solve.
This is one of the hardest coding benchmarks - best models solve ~23%.

Dataset structure:
- instance_id: Unique identifier
- repo: Repository name (e.g., "django/django")
- base_commit: Commit to checkout before applying fix
- problem_statement: GitHub issue description
- patch: Ground truth fix (diff format)
- test_patch: Tests that verify the fix
- hints_text: Optional hints about the solution
"""

import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.commons import (
    create_harbor_task_directory_generic,
    create_generic_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"

# Custom Dockerfile for SWE-bench - needs git and ability to clone repos
SWEBENCH_DOCKERFILE = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install common Python packages
RUN pip install --no-cache-dir \\
    pytest \\
    pytest-xdist \\
    pytest-cov \\
    coverage \\
    flake8 \\
    mypy

# Create directories for task
RUN mkdir -p /workspace /tests /logs/verifier
"""


def create_swebench_test_sh(repo: str, test_cmd: str = "pytest") -> str:
    """Create test.sh for SWE-bench tasks."""
    return f'''#!/bin/bash
set -e

mkdir -p /logs/verifier

cleanup() {{
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}}
trap cleanup EXIT

cd /workspace

# Run the test patch tests
echo "Running SWE-bench tests for {repo}..."
{test_cmd} 2>&1 | tee /logs/verifier/test_output.txt

exit ${{PIPESTATUS[0]}}
'''


def load_swebench(split: str = "test", limit: int = LIMIT) -> List[Dict]:
    """
    Load SWE-bench dataset from HuggingFace.

    Available splits: train, dev, test
    Also available: SWE-bench_Lite (300 samples), SWE-bench_Verified (500 samples)
    """
    print(f"Loading SWE-bench dataset ({split})...")

    # Try different SWE-bench variants
    repo_names = [
        ("princeton-nlp/SWE-bench", split),
        ("princeton-nlp/SWE-bench_Lite", split),
        ("princeton-nlp/SWE-bench_Verified", split),
    ]

    ds = None
    for repo_name, split_name in repo_names:
        try:
            ds = load_dataset(repo_name, split=split_name, streaming=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        raise ValueError("Could not load any SWE-bench dataset variant")

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        # Validate required fields
        instance_id = sample.get("instance_id", "")
        repo = sample.get("repo", "")
        problem_statement = sample.get("problem_statement", "")
        test_patch = sample.get("test_patch", "")

        if not instance_id or not repo or not problem_statement:
            continue

        samples.append({
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": sample.get("base_commit", ""),
            "problem_statement": problem_statement,
            "patch": sample.get("patch", ""),
            "test_patch": test_patch,
            "hints_text": sample.get("hints_text", ""),
            "created_at": sample.get("created_at", ""),
            "version": sample.get("version", ""),
            "fail_to_pass": sample.get("FAIL_TO_PASS", []),
            "pass_to_pass": sample.get("PASS_TO_PASS", []),
        })

    print(f"Loaded {len(samples)} SWE-bench instances")
    return samples


def create_instruction_from_issue(sample: Dict) -> str:
    """Create task instruction from GitHub issue."""
    repo = sample.get("repo", "unknown/repo")
    problem = sample.get("problem_statement", "No problem description available.")
    hints = sample.get("hints_text", "")

    instruction = f"""# GitHub Issue: {repo}

## Problem Statement

{problem}

## Your Task

Fix this issue in the codebase. The solution should:
1. Address all aspects of the issue description
2. Pass the existing test suite
3. Not break any other functionality

The workspace contains the repository at `/workspace`. Make your changes there.
"""

    if hints:
        instruction += f"""
## Hints

{hints}
"""

    return instruction


def extract_test_files_from_patch(sample: Dict) -> Dict[str, str]:
    """Extract test files from the test_patch field."""
    test_patch = sample.get("test_patch", "")
    test_files = {}

    if not test_patch:
        # Create a placeholder test file
        fail_to_pass = sample.get("fail_to_pass", [])
        if fail_to_pass:
            test_list = "\n".join(f"    # {t}" for t in fail_to_pass[:10])
            test_files["test_info.txt"] = f"""# Tests that should pass after fix:
{test_list}
"""
        return test_files

    # Parse the diff to extract test content
    # Format: diff --git a/path/to/file b/path/to/file
    current_file = None
    current_content = []

    for line in test_patch.split("\n"):
        if line.startswith("diff --git"):
            # Save previous file
            if current_file and current_content:
                test_files[current_file] = "\n".join(current_content)

            # Extract new file path
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3].lstrip("b/").replace("/", "_")
                current_content = []
        elif line.startswith("+") and not line.startswith("+++"):
            # Added line (remove the + prefix)
            current_content.append(line[1:])
        elif line.startswith(" "):
            # Context line
            current_content.append(line[1:])

    # Save last file
    if current_file and current_content:
        test_files[current_file] = "\n".join(current_content)

    # If no files extracted, store the raw patch
    if not test_files:
        test_files["test_patch.diff"] = test_patch

    return test_files


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a SWE-bench sample."""
    instruction = create_instruction_from_issue(sample)
    test_files = extract_test_files_from_patch(sample)

    # Add test info file with fail_to_pass tests
    fail_to_pass = sample.get("fail_to_pass", [])
    if fail_to_pass:
        test_files["required_tests.txt"] = "# Tests that must pass:\n" + "\n".join(fail_to_pass)

    metadata = {
        "source": "swebench",
        "instance_id": sample.get("instance_id", ""),
        "repo": sample.get("repo", ""),
        "base_commit": sample.get("base_commit", ""),
        "version": sample.get("version", ""),
        "fail_to_pass_count": len(fail_to_pass),
    }

    # Store the ground truth patch as solution
    solution_files = None
    if sample.get("patch"):
        solution_files = {"solution.patch": sample["patch"]}

    # Create test command based on fail_to_pass tests
    if fail_to_pass:
        test_cmd = f"pytest {' '.join(fail_to_pass[:5])} -v --tb=short"
    else:
        test_cmd = "pytest -v --tb=short"

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=SWEBENCH_DOCKERFILE,
        test_sh=create_swebench_test_sh(sample.get("repo", ""), test_cmd),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(samples: List[Dict], dataset_prefix: str = "swebench") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, sample in enumerate(tqdm(samples, desc="Creating tasks")):
        create_harbor_task(temp_dir, i, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(split: str = "test", limit: int = LIMIT, upload: bool = True) -> None:
    """Main pipeline for generating SWE-bench tasks."""

    print(f"Step 1: Loading SWE-bench dataset ({split})...")
    samples = load_swebench(split=split, limit=limit)
    print(f"  -> {len(samples)} instances loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating harbor task directories...")
    dataset_prefix = "swebench"
    task_dir = generate_tasks(samples, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 3: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} SWE-bench tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from SWE-bench dataset")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test"],
                        help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")

    args = parser.parse_args()
    main(split=args.split, limit=args.limit, upload=not args.no_upload)
