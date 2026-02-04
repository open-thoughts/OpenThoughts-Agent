#!/usr/bin/env python3
"""
Generate tasks from Defects4J dataset (854 real-world Java bugs).

Defects4J provides:
- Real bugs from popular Java projects (JFreeChart, Closure, Math, etc.)
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
    create_generic_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 854  # Full dataset
MODEL = "gpt-4o-mini"

BUG_FIX_PROMPT = """You are an expert at creating bug-fixing tasks.

Given the following information about a bug in a Java project, create a clear task description for fixing it:

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


def load_defects4j(limit: int = LIMIT) -> List[Dict]:
    """
    Load Defects4J dataset from HuggingFace.

    Available at various HuggingFace repos.
    """
    print("Loading Defects4J dataset...")

    # Try multiple sources (updated Jan 2026)
    repo_names = [
        "rufimelo/defects4j",        # Current working source
        "CoQuIR/Defects4J",          # Alternative source
        "defects4j/defects4j",
        "ASSERT-KTH/defects4j",
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
        raise ValueError("Could not load Defects4J dataset from any source")

    samples = []
    print(f"Collecting {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        # Handle rufimelo/defects4j schema: bug_id, func_before, func_after
        bug_id = sample.get("bug_id", sample.get("id", ""))
        buggy = (
            sample.get("func_before") or      # rufimelo/defects4j
            sample.get("buggy_code") or
            sample.get("buggy") or
            sample.get("source", "")
        )
        fixed = (
            sample.get("func_after") or       # rufimelo/defects4j
            sample.get("fixed_code") or
            sample.get("fixed") or
            sample.get("target", "")
        )
        test = sample.get("failing_test", sample.get("test", ""))

        # Extract project from bug_id (e.g., "Compress-35" -> "Compress")
        project = sample.get("project", "")
        if not project and "-" in str(bug_id):
            project = str(bug_id).rsplit("-", 1)[0]

        if not buggy:
            continue

        samples.append({
            "project": project,
            "bug_id": bug_id,
            "buggy_code": buggy,
            "fixed_code": fixed,
            "failing_test": test,
            "test_class": sample.get("test_class", ""),
            "test_method": sample.get("test_method", ""),
        })

    print(f"Loaded {len(samples)} bugs")
    return samples


def create_java_bug_test_sh() -> str:
    """Create test.sh for Java bug fix testing."""
    return '''#!/bin/bash
set -e

mkdir -p /logs/verifier

cleanup() {
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}
trap cleanup EXIT

cd /app

# Compile and run tests
echo "Compiling..."
mvn compile -q 2>&1 || true

echo "Running tests..."
mvn test -q 2>&1 | tee /logs/verifier/test_output.txt

exit ${PIPESTATUS[0]}
'''


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a Defects4J bug."""
    dockerfile = get_dockerfile("java")

    # Create test file
    test_code = sample.get("failing_test", "")
    test_class = sample.get("test_class", "TestFix")

    if "class" not in test_code:
        test_code = f"""import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

public class {test_class} {{
    {test_code}
}}"""

    test_files = {
        f"{test_class}.java": test_code,
    }

    metadata = {
        "source": "defects4j",
        "project": sample.get("project", ""),
        "bug_id": sample.get("bug_id", ""),
    }

    # Include fixed code as solution
    solution_files = None
    if sample.get("fixed_code"):
        solution_files = {"Solution.java": sample["fixed_code"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=create_java_bug_test_sh(),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "defects4j",
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
    """Main pipeline for generating Defects4J tasks."""

    print("Step 1: Loading Defects4J dataset...")
    samples = load_defects4j(limit=limit)
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
    task_dir = generate_tasks(samples, instructions, "defects4j")
    print(f"  -> Task directory: {task_dir}")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_defects4j")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} Defects4J tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from Defects4J dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")

    args = parser.parse_args()
    main(limit=args.limit)
