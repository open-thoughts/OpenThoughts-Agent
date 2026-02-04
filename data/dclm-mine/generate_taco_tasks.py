#!/usr/bin/env python3
"""
Generate tasks from TACO dataset (26K competitive programming problems).

TACO (Testing AI Coding Operations) from BAAI contains:
- Problem descriptions
- Input/output test cases
- Difficulty levels
- Topics/tags
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
    PROBLEM_STATEMENT_CLEANUP_PROMPT,
    create_harbor_task_directory_generic,
    create_io_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"


def load_taco(limit: int = LIMIT) -> List[Dict]:
    """
    Load TACO dataset from HuggingFace.

    Available at: BAAI/TACO
    """
    print("Loading TACO dataset...")

    # Try multiple sources (updated Jan 2026)
    # BAAI/TACO deprecated custom scripts - use verified/converted versions
    repo_configs = [
        ("likaixin/TACO-verified", "train"),      # Verified solutions, parquet format
    ]

    ds = None
    for repo_name, split in repo_configs:
        try:
            ds = load_dataset(repo_name, split=split, streaming=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        raise ValueError("Could not load TACO dataset from any source")

    samples = []
    print(f"Collecting {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        # Extract problem description
        problem = (
            sample.get("question") or
            sample.get("problem") or
            sample.get("description", "")
        )

        if not problem:
            continue

        # Get test cases - TACO uses 'input_output' field
        input_output = sample.get("input_output", {})
        if isinstance(input_output, str):
            try:
                input_output = json.loads(input_output)
            except:
                input_output = {}

        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])

        # Also try alternate field names
        if not inputs:
            inputs = sample.get("inputs", sample.get("input", []))
        if not outputs:
            outputs = sample.get("outputs", sample.get("output", []))

        # Handle string format
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]

        samples.append({
            "problem_description": problem,
            "inputs": inputs,
            "outputs": outputs,
            "difficulty": sample.get("difficulty", ""),
            "tags": sample.get("tags", sample.get("topics", [])),
            "source": sample.get("source", sample.get("url", "")),
            "solutions": sample.get("solutions", []),
        })

    print(f"Loaded {len(samples)} problems")
    return samples


def create_test_files(sample: Dict) -> Dict[str, str]:
    """Create input/output test files from sample."""
    test_files = {}

    inputs = sample.get("inputs", [])
    outputs = sample.get("outputs", [])

    # Create input/output files
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        # Handle nested lists
        if isinstance(inp, list):
            inp = "\n".join(str(x) for x in inp)
        if isinstance(out, list):
            out = "\n".join(str(x) for x in out)

        test_files[f"inputs/input_{i}.txt"] = str(inp)
        test_files[f"outputs/output_{i}.txt"] = str(out)

    # If no test cases, create placeholder
    if not inputs:
        test_files["inputs/input_0.txt"] = ""
        test_files["outputs/output_0.txt"] = ""

    return test_files


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a TACO sample."""
    dockerfile = get_dockerfile("python")

    test_files = create_test_files(sample)

    metadata = {
        "source": "taco",
        "difficulty": sample.get("difficulty", ""),
        "tags": sample.get("tags", []),
        "original_source": sample.get("source", ""),
        "num_tests": len(sample.get("inputs", [])),
    }

    # Include solution if available
    solution_files = None
    solutions = sample.get("solutions", [])
    if solutions:
        # Get the first solution
        solution = solutions[0] if isinstance(solutions, list) else solutions
        if isinstance(solution, str):
            solution_files = {"solution.py": solution}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=create_io_test_sh(solution_cmd="python3 /app/solution.py"),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "taco",
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
    """Main pipeline for generating TACO tasks."""

    print("Step 1: Loading TACO dataset...")
    samples = load_taco(limit=limit)
    print(f"  -> {len(samples)} problems loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Cleaning up problem descriptions...")
    dataset = Dataset.from_list([{"description": s["problem_description"]} for s in samples])

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": PROBLEM_STATEMENT_CLEANUP_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instructions = result.dataset["task_description"]
    print(f"  -> Processed {len(instructions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, "taco")
    print(f"  -> Task directory: {task_dir}")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_taco")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} TACO tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from TACO dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")

    args = parser.parse_args()
    main(limit=args.limit)
