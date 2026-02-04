#!/usr/bin/env python3
"""
Generate tasks from CodeElo dataset (387 competition-level algorithm problems).

CodeElo contains 387 curated CodeForces problems (Div. 1-4) with Elo-rated
difficulty. Covers 35 algorithmic domains (DP, graphs, number theory, geometry).
Zero false positives - validated on actual CodeForces platform.

This is one of the hardest algorithmic benchmarks available.
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
    create_io_test_sh,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"

# Dockerfile for competitive programming (supports multiple languages)
CODEELO_DOCKERFILE = """FROM python:3.10-slim

WORKDIR /app

# Install multiple language runtimes for competitive programming
RUN apt-get update && apt-get install -y \\
    g++ \\
    openjdk-17-jdk \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for competitive programming
RUN pip install --no-cache-dir \\
    numpy \\
    scipy \\
    networkx \\
    sympy

RUN mkdir -p /logs/verifier /tests/inputs /tests/outputs
"""


def create_codeelo_test_sh(language: str = "python") -> str:
    """Create test.sh for CodeElo I/O-based tests."""
    # Language-specific run commands
    run_commands = {
        "python": "python3 /app/solution.py",
        "cpp": "g++ -O2 -std=c++17 /app/solution.cpp -o /app/solution && /app/solution",
        "java": "javac /app/Solution.java && java -cp /app Solution",
    }

    solution_cmd = run_commands.get(language.lower(), run_commands["python"])

    return f'''#!/bin/bash
set -e

mkdir -p /logs/verifier

PASSED=0
FAILED=0
TOTAL=0

cd /app

for input_file in /tests/inputs/input_*.txt; do
    if [ ! -f "$input_file" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))
    test_num=$(basename "$input_file" | sed 's/input_//;s/.txt//')
    expected_file="/tests/outputs/output_$test_num.txt"

    if [ ! -f "$expected_file" ]; then
        echo "Missing expected output for test $test_num"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "Running test $test_num..."

    # Run with timeout (60 seconds per test)
    actual_output=$(timeout 60 {solution_cmd} < "$input_file" 2>&1) || {{
        echo "Test $test_num: TIMEOUT or ERROR"
        FAILED=$((FAILED + 1))
        continue
    }}

    expected_output=$(cat "$expected_file")

    # Normalize whitespace for comparison
    actual_normalized=$(echo "$actual_output" | tr -s '[:space:]' ' ' | sed 's/^ //;s/ $//')
    expected_normalized=$(echo "$expected_output" | tr -s '[:space:]' ' ' | sed 's/^ //;s/ $//')

    if [ "$actual_normalized" = "$expected_normalized" ]; then
        echo "Test $test_num: PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "Test $test_num: FAILED"
        echo "Expected: $expected_output"
        echo "Actual: $actual_output"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Results: $PASSED/$TOTAL passed"

if [ $FAILED -eq 0 ] && [ $TOTAL -gt 0 ]; then
    echo "1" > /logs/verifier/reward.txt
    exit 0
else
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi
'''


def load_codeelo(limit: int = LIMIT) -> List[Dict]:
    """
    Load CodeElo dataset.

    CodeElo is available at codeelo-bench.github.io
    May need to download from the website or find HuggingFace mirror.
    """
    print(f"Loading CodeElo dataset...")

    # Try different possible sources
    repo_names = [
        "codeelo/codeelo-bench",
        "CodeElo/codeelo",
        "codeelo-bench/codeelo",
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
        # Fallback: try loading CodeForces problems from another source
        try:
            ds = load_dataset("deepmind/code_contests", split="test", streaming=True)
            print("Loaded from deepmind/code_contests as fallback")
        except Exception as e:
            print(f"Could not load fallback dataset: {e}")
            raise ValueError("Could not load CodeElo or similar dataset")

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        # Try to extract fields (format varies by dataset)
        problem_id = sample.get("problem_id", sample.get("name", sample.get("id", "")))
        description = sample.get("description", sample.get("problem_statement", sample.get("question", "")))

        # Test cases
        public_tests = sample.get("public_tests", {})
        private_tests = sample.get("private_tests", {})
        generated_tests = sample.get("generated_tests", {})

        # Combine all test cases
        inputs = (
            public_tests.get("input", []) +
            private_tests.get("input", []) +
            generated_tests.get("input", [])
        )
        outputs = (
            public_tests.get("output", []) +
            private_tests.get("output", []) +
            generated_tests.get("output", [])
        )

        if not problem_id or not description:
            continue

        # Get difficulty info
        difficulty = sample.get("difficulty", sample.get("rating", sample.get("elo", "")))
        tags = sample.get("tags", sample.get("cf_tags", []))
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        samples.append({
            "problem_id": problem_id,
            "description": description,
            "inputs": inputs[:20],  # Limit test cases
            "outputs": outputs[:20],
            "difficulty": difficulty,
            "tags": tags,
            "time_limit": sample.get("time_limit", sample.get("cf_time_limit", 2)),
            "memory_limit": sample.get("memory_limit", sample.get("cf_memory_limit", 256)),
            "solutions": sample.get("solutions", {}),
        })

    print(f"Loaded {len(samples)} CodeElo problems")
    return samples


def create_instruction_from_problem(sample: Dict) -> str:
    """Create task instruction from CodeElo problem."""
    problem_id = sample.get("problem_id", "Unknown")
    description = sample.get("description", "No description available.")
    difficulty = sample.get("difficulty", "Unknown")
    tags = sample.get("tags", [])
    time_limit = sample.get("time_limit", 2)
    memory_limit = sample.get("memory_limit", 256)

    instruction = f"""# Problem: {problem_id}

## Difficulty
Rating: {difficulty}

## Tags
{', '.join(tags) if tags else 'None specified'}

## Constraints
- Time Limit: {time_limit} seconds
- Memory Limit: {memory_limit} MB

## Problem Statement

{description}

## Your Task

Implement a solution that:
1. Reads input from stdin
2. Produces output to stdout
3. Passes all test cases within the time and memory limits

Place your solution in `/app/solution.py` (or solution.cpp/Solution.java for C++/Java).

## Tips for Competitive Programming
- Consider edge cases (empty input, maximum values, etc.)
- Optimize for the given time/memory constraints
- Test with the provided examples before submitting
"""

    # Add example test cases to instruction
    inputs = sample.get("inputs", [])
    outputs = sample.get("outputs", [])

    if inputs and outputs:
        instruction += "\n## Examples\n"
        for i, (inp, out) in enumerate(zip(inputs[:3], outputs[:3])):
            instruction += f"""
### Example {i + 1}
**Input:**
```
{inp}
```

**Output:**
```
{out}
```
"""

    return instruction


def create_test_files(sample: Dict) -> Dict[str, str]:
    """Create input/output test files from CodeElo sample."""
    test_files = {}
    inputs = sample.get("inputs", [])
    outputs = sample.get("outputs", [])

    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        test_files[f"inputs/input_{i}.txt"] = inp if isinstance(inp, str) else str(inp)
        test_files[f"outputs/output_{i}.txt"] = out if isinstance(out, str) else str(out)

    return test_files


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a CodeElo sample."""
    instruction = create_instruction_from_problem(sample)
    test_files = create_test_files(sample)

    metadata = {
        "source": "codeelo",
        "problem_id": sample.get("problem_id", ""),
        "difficulty": sample.get("difficulty", ""),
        "tags": sample.get("tags", []),
        "time_limit": sample.get("time_limit", 2),
        "memory_limit": sample.get("memory_limit", 256),
        "test_count": len(sample.get("inputs", [])),
    }

    # Store solutions if available
    solution_files = None
    solutions = sample.get("solutions", {})
    if solutions:
        solution_files = {}
        if "python" in solutions or "python3" in solutions:
            sol = solutions.get("python3", solutions.get("python", []))
            if sol:
                solution_files["solution.py"] = sol[0] if isinstance(sol, list) else sol
        if "cpp" in solutions or "c++" in solutions:
            sol = solutions.get("cpp", solutions.get("c++", []))
            if sol:
                solution_files["solution.cpp"] = sol[0] if isinstance(sol, list) else sol

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=CODEELO_DOCKERFILE,
        test_sh=create_codeelo_test_sh("python"),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(samples: List[Dict], dataset_prefix: str = "codeelo") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, sample in enumerate(tqdm(samples, desc="Creating tasks")):
        create_harbor_task(temp_dir, i, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(limit: int = LIMIT, upload: bool = True) -> None:
    """Main pipeline for generating CodeElo tasks."""

    print(f"Step 1: Loading CodeElo dataset...")
    samples = load_codeelo(limit=limit)
    print(f"  -> {len(samples)} problems loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating harbor task directories...")
    dataset_prefix = "codeelo"
    task_dir = generate_tasks(samples, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 3: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} CodeElo tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from CodeElo dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")

    args = parser.parse_args()
    main(limit=args.limit, upload=not args.no_upload)
