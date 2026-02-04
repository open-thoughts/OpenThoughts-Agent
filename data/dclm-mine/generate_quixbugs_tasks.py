#!/usr/bin/env python3
"""
Generate tasks from QuixBugs dataset (40 classic algorithm bugs).

QuixBugs provides:
- 40 buggy implementations of classic algorithms
- Same bugs in Python and Java
- Test cases that expose the bugs
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
LIMIT = 40  # Full dataset
MODEL = "gpt-4o-mini"
DEFAULT_LANGUAGE = "python"

BUG_FIX_PROMPT = """You are an expert at creating bug-fixing tasks.

Given the following buggy implementation of a classic algorithm, create a clear task description for fixing it:

Algorithm: {{algorithm}}
Language: {{language}}

Buggy Code:
{{buggy_code}}

Test cases:
{{test_cases}}

The task should:
1. Describe what the algorithm should do
2. Explain what's wrong with the current implementation (without giving the answer)
3. Be self-contained and actionable
4. The fix should be applied in /app/

Create a bug-fixing task description:"""


def load_quixbugs(language: str = "python", limit: int = LIMIT) -> List[Dict]:
    """
    Load QuixBugs dataset from HuggingFace.
    """
    print(f"Loading QuixBugs dataset ({language})...")

    # Try multiple sources (updated Jan 2026)
    repo_names = [
        "Muennighoff/quixbugs",  # Current working source
        "jkoppel/QuixBugs",
        "quixbugs/quixbugs",
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
        raise ValueError("Could not load QuixBugs dataset from any source")

    samples = []
    print(f"Collecting {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit * 2), desc="Loading samples"):
        # Note: Muennighoff/quixbugs dataset doesn't have language field
        # All samples are Python
        algorithm = sample.get("algorithm", sample.get("name", sample.get("program", "")))
        buggy = sample.get("buggy_program", sample.get("buggy_code", sample.get("buggy", sample.get("source", ""))))
        fixed = sample.get("solution", sample.get("fixed_code", sample.get("fixed", sample.get("target", ""))))
        tests = sample.get("test_cases", sample.get("tests", []))

        if not buggy:
            continue

        samples.append({
            "algorithm": algorithm,
            "language": language,
            "buggy_code": buggy,
            "fixed_code": fixed,
            "test_cases": tests if isinstance(tests, str) else json.dumps(tests),
        })

        if len(samples) >= limit:
            break

    print(f"Loaded {len(samples)} bugs")
    return samples


def create_test_file_content(sample: Dict) -> str:
    """Create pytest test file from sample."""
    algorithm = sample.get("algorithm", "solution")
    test_cases = sample.get("test_cases", "[]")

    if isinstance(test_cases, str):
        try:
            test_cases = json.loads(test_cases)
        except:
            test_cases = []

    # Generate pytest test functions
    test_code = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import {algorithm}

"""

    for i, test in enumerate(test_cases[:10]):  # Limit to 10 test cases
        if isinstance(test, dict):
            inp = test.get("input", test.get("inputs", ""))
            expected = test.get("output", test.get("expected", ""))
        elif isinstance(test, (list, tuple)) and len(test) >= 2:
            inp = test[0]
            expected = test[1]
        else:
            continue

        test_code += f"""
def test_case_{i}():
    result = {algorithm}({repr(inp) if not isinstance(inp, str) else inp})
    assert result == {repr(expected)}, f"Expected {repr(expected)}, got {{result}}"
"""

    return test_code


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a QuixBugs bug."""
    language = sample.get("language", "python")

    if language == "python":
        dockerfile = get_dockerfile("python")
        test_sh = create_pytest_test_sh("/tests/test_solution.py")
        test_file = "test_solution.py"
    else:
        dockerfile = get_dockerfile("java")
        test_file = "TestSolution.java"
        test_sh = '''#!/bin/bash
set -e
mkdir -p /logs/verifier
cd /app
mvn test -q 2>&1 | tee /logs/verifier/test_output.txt
if [ $? -eq 0 ]; then echo "1" > /logs/verifier/reward.txt; else echo "0" > /logs/verifier/reward.txt; fi
'''

    test_files = {
        test_file: create_test_file_content(sample),
    }

    metadata = {
        "source": "quixbugs",
        "algorithm": sample.get("algorithm", ""),
        "language": language,
    }

    # Include fixed code as solution
    solution_files = None
    if sample.get("fixed_code"):
        ext = "py" if language == "python" else "java"
        solution_files = {f"solution.{ext}": sample["fixed_code"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=test_sh,
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "quixbugs",
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


def main(language: str = DEFAULT_LANGUAGE, limit: int = LIMIT) -> None:
    """Main pipeline for generating QuixBugs tasks."""

    print(f"Step 1: Loading QuixBugs dataset ({language})...")
    samples = load_quixbugs(language=language, limit=limit)
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
    dataset_prefix = f"quixbugs-{language}"
    task_dir = generate_tasks(samples, instructions, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} QuixBugs tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from QuixBugs dataset")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        choices=["python", "java"],
                        help="Programming language")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")

    args = parser.parse_args()
    main(language=args.language, limit=args.limit)
