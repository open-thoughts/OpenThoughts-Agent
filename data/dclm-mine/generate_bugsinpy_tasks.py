#!/usr/bin/env python3
"""
Generate tasks from BugsInPy dataset (493 real-world Python bugs).

BugsInPy provides:
- Real bugs from popular Python projects
- Test cases that expose the bugs
- Buggy and fixed versions
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

from data.completions import run_completions
from data.commons import (
    create_harbor_task_directory_generic,
    create_pytest_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 493  # Full dataset
MODEL = "gpt-4o-mini"

BUG_FIX_PROMPT = """You are an expert at creating bug-fixing tasks.

Given the following information about a bug in a Python project, create a clear task description for fixing it:

Project: {{project}}
Bug ID: {{bug_id}}
Buggy Code:
{{buggy_code}}

Test that fails:
{{failing_test}}

The task should:
1. Describe what functionality is broken
2. Explain what the fix should accomplish (without giving the answer)
3. Be self-contained and actionable
4. The fix should be applied in /app/

Create a bug-fixing task description:"""


def load_bugsinpy(limit: int = LIMIT) -> List[Dict]:
    """
    Load BugsInPy dataset from HuggingFace.
    """
    print("Loading BugsInPy dataset...")

    # Try multiple sources (updated Jan 2026)
    repo_names = [
        "Harryxun/BugsInPy_data",  # Has buggy/fixed pairs with test commands
        "xin1997/bugsinpy_all_only_input",
        "soarsmu/BugsInPy",
        "bugsinpy/bugsinpy",
    ]

    ds = None
    for repo_name in repo_names:
        try:
            ds = load_dataset(repo_name, split="train", streaming=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        raise ValueError("Could not load BugsInPy dataset from any source")

    samples = []
    print(f"Collecting {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        project = sample.get("project", sample.get("proj", ""))
        bug_id = sample.get("bug_id", sample.get("id", ""))
        buggy = sample.get("buggy", sample.get("buggy_code", sample.get("source", "")))
        fixed = sample.get("fixed", sample.get("fixed_code", sample.get("target", "")))
        test = sample.get("test_command", sample.get("failing_test", sample.get("test", "")))

        if not buggy:
            continue

        samples.append({
            "project": project,
            "bug_id": bug_id,
            "buggy_code": buggy,
            "fixed_code": fixed,
            "failing_test": test,
            "test_file": sample.get("test_file", ""),
        })

    print(f"Loaded {len(samples)} bugs")
    return samples


def create_test_file_content(sample: Dict) -> str:
    """Create pytest test file from sample."""
    test_code = sample.get("failing_test", "")

    # Add imports if needed
    imports = "import pytest\nimport sys\nsys.path.insert(0, '/app')\n\n"

    if "import pytest" not in test_code:
        test_code = imports + test_code

    return test_code


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a BugsInPy bug."""
    dockerfile = get_dockerfile("python")

    test_files = {
        "test_fix.py": create_test_file_content(sample),
    }

    metadata = {
        "source": "bugsinpy",
        "project": sample.get("project", ""),
        "bug_id": sample.get("bug_id", ""),
    }

    # Include fixed code as solution
    solution_files = None
    if sample.get("fixed_code"):
        solution_files = {"solution.py": sample["fixed_code"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=create_pytest_test_sh("/tests/test_fix.py"),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "bugsinpy",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(
        zip(samples, instructions),
        total=len(samples),
        desc="Creating tasks"
    )):
        create_harbor_task(temp_dir, i, instruction, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(limit: int = LIMIT) -> None:
    """Main pipeline for generating BugsInPy tasks."""

    print("Step 1: Loading BugsInPy dataset...")
    samples = load_bugsinpy(limit=limit)
    print(f"  -> {len(samples)} bugs loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating bug-fix task descriptions...")
    dataset = Dataset.from_list(samples)

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": BUG_FIX_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instructions = result.dataset["task_description"]
    print(f"  -> Generated {len(instructions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, "bugsinpy")
    print(f"  -> Task directory: {task_dir}")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_bugsinpy")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} BugsInPy tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from BugsInPy dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")

    args = parser.parse_args()
    main(limit=args.limit)
