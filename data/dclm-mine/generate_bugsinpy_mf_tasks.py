#!/usr/bin/env python3
"""
Generate tasks from BugsInPy-MF dataset (501 multi-fault versions).

BugsInPy-MF extends BugsInPy with multi-fault program versions where
bugs can interact and mask each other. Average 18.6 faults per version.
Much harder than single-fault benchmarks.

Source: https://github.com/DCallaz/bugsinpy-mf
"""

import itertools
import json
import os
import subprocess
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
    create_pytest_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"

BUGSINPY_MF_REPO = "https://github.com/DCallaz/bugsinpy-mf.git"

# Dockerfile for BugsInPy-MF
BUGSINPY_MF_DOCKERFILE = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install common Python packages
RUN pip install --no-cache-dir \\
    pytest \\
    pytest-timeout \\
    pytest-cov \\
    coverage

RUN mkdir -p /logs/verifier /workspace
"""


def create_bugsinpy_mf_test_sh(test_cmd: str = "pytest") -> str:
    """Create test.sh for BugsInPy-MF tasks."""
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

echo "Running BugsInPy-MF tests..."
{test_cmd} 2>&1 | tee /logs/verifier/test_output.txt

exit ${{PIPESTATUS[0]}}
'''


def load_bugsinpy_mf(limit: int = LIMIT) -> List[Dict]:
    """
    Load BugsInPy-MF dataset.

    The dataset contains multi-fault versions of Python projects from BugsInPy.
    """
    print(f"Loading BugsInPy-MF dataset...")

    # Try loading from HuggingFace first
    repo_names = [
        "bugsinpy/bugsinpy-mf",
        "DCallaz/bugsinpy-mf",
        "defects4ml/bugsinpy-mf",
    ]

    ds = None
    for repo_name in repo_names:
        try:
            ds = load_dataset(repo_name, split="test", streaming=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        # Fallback to base BugsInPy and note multi-fault
        try:
            ds = load_dataset("soarsmu/BugsInPy", split="train", streaming=True)
            print("Loaded from soarsmu/BugsInPy (base dataset)")
        except Exception as e:
            print(f"Could not load BugsInPy: {e}")
            print("Creating sample multi-fault data for demonstration...")
            return _create_sample_mf_data(limit)

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        bug_id = sample.get("bug_id", sample.get("id", ""))
        project = sample.get("project", sample.get("repo", ""))
        buggy_code = sample.get("buggy_code", sample.get("buggy", ""))
        fixed_code = sample.get("fixed_code", sample.get("fixed", ""))
        test_code = sample.get("test_code", sample.get("test", ""))

        if not bug_id or not project:
            continue

        # Multi-fault info
        fault_count = sample.get("fault_count", sample.get("num_faults", 1))
        fault_locations = sample.get("fault_locations", [])
        interacting_faults = sample.get("interacting_faults", [])

        samples.append({
            "bug_id": bug_id,
            "project": project,
            "buggy_code": buggy_code,
            "fixed_code": fixed_code,
            "test_code": test_code,
            "fault_count": fault_count,
            "fault_locations": fault_locations,
            "interacting_faults": interacting_faults,
            "version": sample.get("version", ""),
            "commit": sample.get("commit", sample.get("buggy_commit", "")),
            "fix_commit": sample.get("fix_commit", sample.get("fixed_commit", "")),
        })

    print(f"Loaded {len(samples)} multi-fault bug samples")
    return samples


def _create_sample_mf_data(limit: int) -> List[Dict]:
    """Create sample multi-fault data for demonstration."""
    samples = [
        {
            "bug_id": "mf-001",
            "project": "example-calculator",
            "buggy_code": '''def add(a, b):
    return a - b  # Bug 1: wrong operator

def multiply(a, b):
    return a + b  # Bug 2: wrong operator

def divide(a, b):
    return a * b  # Bug 3: wrong operator

def calculate(op, a, b):
    if op == "add":
        return add(a, b)
    elif op == "mul":
        return multiply(a, b)
    elif op == "div":
        if b == 0:
            return 0  # Bug 4: should raise exception
        return divide(a, b)
''',
            "fixed_code": '''def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b

def calculate(op, a, b):
    if op == "add":
        return add(a, b)
    elif op == "mul":
        return multiply(a, b)
    elif op == "div":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return divide(a, b)
''',
            "test_code": '''import pytest
from solution import add, multiply, divide, calculate

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6

def test_divide():
    assert divide(6, 2) == 3
    assert divide(10, 5) == 2

def test_calculate():
    assert calculate("add", 2, 3) == 5
    assert calculate("mul", 2, 3) == 6
    assert calculate("div", 6, 2) == 3

def test_divide_by_zero():
    with pytest.raises(ValueError):
        calculate("div", 5, 0)
''',
            "fault_count": 4,
            "fault_locations": ["add:line2", "multiply:line5", "divide:line8", "calculate:line15"],
            "interacting_faults": [("add", "calculate"), ("multiply", "calculate")],
        },
        {
            "bug_id": "mf-002",
            "project": "example-string-utils",
            "buggy_code": '''def reverse_string(s):
    return s  # Bug 1: not reversing

def is_palindrome(s):
    return s == s  # Bug 2: comparing with self, not reversed

def count_chars(s, char):
    count = 0
    for c in s:
        if c == char:
            count = 1  # Bug 3: should increment
    return count
''',
            "fixed_code": '''def reverse_string(s):
    return s[::-1]

def is_palindrome(s):
    return s == s[::-1]

def count_chars(s, char):
    count = 0
    for c in s:
        if c == char:
            count += 1
    return count
''',
            "test_code": '''import pytest
from solution import reverse_string, is_palindrome, count_chars

def test_reverse_string():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("") == ""

def test_is_palindrome():
    assert is_palindrome("radar") == True
    assert is_palindrome("hello") == False

def test_count_chars():
    assert count_chars("hello", "l") == 2
    assert count_chars("aaa", "a") == 3
''',
            "fault_count": 3,
            "fault_locations": ["reverse_string:line2", "is_palindrome:line5", "count_chars:line11"],
            "interacting_faults": [("reverse_string", "is_palindrome")],
        },
    ]

    return samples[:limit]


def create_instruction_from_mf_bug(sample: Dict) -> str:
    """Create task instruction from multi-fault bug sample."""
    project = sample.get("project", "Unknown project")
    bug_id = sample.get("bug_id", "Unknown")
    fault_count = sample.get("fault_count", 1)
    fault_locations = sample.get("fault_locations", [])
    buggy_code = sample.get("buggy_code", "")

    instruction = f"""# Multi-Fault Bug Fix Task

## Project
{project}

## Bug ID
{bug_id}

## Overview
This code contains **{fault_count} interacting bugs** that need to be fixed.
Some bugs may mask or interact with others, making diagnosis harder.

"""

    if fault_locations:
        instruction += f"""## Known Fault Locations
The following locations are known to contain bugs:
"""
        for loc in fault_locations[:10]:
            instruction += f"- {loc}\n"
        instruction += "\n"

    instruction += f"""## Buggy Code

```python
{buggy_code}
```

## Your Task

1. Identify all {fault_count} bugs in the code
2. Fix each bug while considering how they might interact
3. Ensure all tests pass after your fixes
4. Place your fixed code in `/app/solution.py`

## Important Notes

- Bugs may mask each other - fixing one bug might reveal another
- Consider the interaction between functions
- Run all tests to verify your fixes work together
"""

    return instruction


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a BugsInPy-MF sample."""
    instruction = create_instruction_from_mf_bug(sample)

    test_files = {
        "test_solution.py": sample.get("test_code", ""),
    }

    # Add fault info file
    fault_info = {
        "fault_count": sample.get("fault_count", 1),
        "fault_locations": sample.get("fault_locations", []),
        "interacting_faults": sample.get("interacting_faults", []),
    }
    test_files["fault_info.json"] = json.dumps(fault_info, indent=2)

    metadata = {
        "source": "bugsinpy-mf",
        "bug_id": sample.get("bug_id", ""),
        "project": sample.get("project", ""),
        "fault_count": sample.get("fault_count", 1),
        "version": sample.get("version", ""),
        "commit": sample.get("commit", ""),
    }

    # Store fixed code as solution
    solution_files = None
    if sample.get("fixed_code"):
        solution_files = {"solution.py": sample["fixed_code"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=BUGSINPY_MF_DOCKERFILE,
        test_sh=create_bugsinpy_mf_test_sh("pytest /tests/test_solution.py -v --tb=short"),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(samples: List[Dict], dataset_prefix: str = "bugsinpy-mf") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, sample in enumerate(tqdm(samples, desc="Creating tasks")):
        create_harbor_task(temp_dir, i, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(limit: int = LIMIT, upload: bool = True) -> None:
    """Main pipeline for generating BugsInPy-MF tasks."""

    print(f"Step 1: Loading BugsInPy-MF dataset...")
    samples = load_bugsinpy_mf(limit=limit)
    print(f"  -> {len(samples)} multi-fault versions loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating harbor task directories...")
    dataset_prefix = "bugsinpy-mf"
    task_dir = generate_tasks(samples, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 3: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} BugsInPy-MF tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from BugsInPy-MF dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")

    args = parser.parse_args()
    main(limit=args.limit, upload=not args.no_upload)
