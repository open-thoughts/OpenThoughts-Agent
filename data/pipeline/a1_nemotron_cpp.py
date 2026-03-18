#!/usr/bin/env python3
"""
Generate C++ test tasks from Nemotron - filters for Google Test/Catch2 test files.
"""

import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from nemotron_loader import load_nemotron_stream, normalize_sample
from data.completions import run_completions
from data.commons import create_standard_task_toml, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 5_000
MODEL = "gpt-5-nano-2025-08-07"

CPP_TEST_TO_TASK_PROMPT = """You are an expert at creating coding tasks from C++ test code.

Given the following C++ test code (Google Test, Catch2, or similar), create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):"""


def is_cpp_test_file(content: str) -> bool:
    """Check if content is a C++ test file."""
    # Google Test patterns
    has_gtest = '#include <gtest' in content or '#include "gtest' in content
    has_test_macro = 'TEST(' in content or 'TEST_F(' in content
    has_expect = 'EXPECT_' in content or 'ASSERT_' in content

    # Catch2 patterns
    has_catch = '#include <catch' in content or '#include "catch' in content
    has_test_case = 'TEST_CASE(' in content
    has_require = 'REQUIRE(' in content or 'CHECK(' in content

    # Doctest patterns
    has_doctest = '#include <doctest' in content or '#include "doctest' in content

    is_gtest = has_gtest or (has_test_macro and has_expect)
    is_catch = has_catch or (has_test_case and has_require)
    is_doctest = has_doctest

    return is_gtest or is_catch or is_doctest


def count_test_cases(content: str) -> int:
    """Count number of test cases."""
    test_count = len(re.findall(r'TEST\s*\(', content))
    test_f_count = len(re.findall(r'TEST_F\s*\(', content))
    test_case_count = len(re.findall(r'TEST_CASE\s*\(', content))
    return test_count + test_f_count + test_case_count


def filter_cpp_tests_from_nemotron(limit: int, max_scan: int = 50_000_000) -> List[Dict]:
    """Filter Nemotron for C++ test content."""
    print("Loading Nemotron (C++ subset, streaming)...")
    ds = load_nemotron_stream("c++")

    cpp_samples = []
    scanned = 0

    print(f"Scanning for C++ test files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding C++ test files")

    for raw_sample in itertools.islice(ds, max_scan):
        scanned += 1
        sample = normalize_sample(raw_sample)
        content = sample['content']

        # Check content directly
        if is_cpp_test_file(content):
            test_count = count_test_cases(content)
            if test_count >= 2:
                cpp_samples.append({
                    'text': content,
                    'path': sample['path'],
                    'repo': sample['repo'],
                    'size': sample['size'],
                    'test_count': test_count,
                })
                pbar.update(1)

                if len(cpp_samples) >= limit:
                    break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(cpp_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(cpp_samples)} C++ test files")
    return cpp_samples


def create_cpp_dockerfile() -> str:
    """Create C++ Dockerfile for test environment."""
    return """FROM gcc:13

WORKDIR /app

# Install build tools including bash for Harbor agent
RUN apt-get update && apt-get install -y cmake libgtest-dev bash && rm -rf /var/lib/apt/lists/*

# Build and install gtest
RUN cd /usr/src/gtest && cmake . && make && cp lib/*.a /usr/lib/
"""


def create_test_sh() -> str:
    """Create test.sh that runs C++ tests and reports results."""
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

# Build and run tests
echo "Building and running C++ tests..."

if [ -f CMakeLists.txt ]; then
    mkdir -p build && cd build
    cmake .. && make
    timeout 300 ctest --output-on-failure 2>&1 | tee /logs/verifier/test_output.txt
else
    # Manual compilation with gtest - try to compile any .cpp files in /app
    CPP_FILES=$(find /app -maxdepth 1 -name "*.cpp" 2>/dev/null | tr '\n' ' ')
    if [ -n "$CPP_FILES" ]; then
        g++ -std=c++17 -o test_runner /tests/test_solution.cpp $CPP_FILES -lgtest -lgtest_main -pthread 2>&1 | tee /logs/verifier/compile_output.txt
    else
        # Just compile the test file if no implementation files exist yet
        g++ -std=c++17 -o test_runner /tests/test_solution.cpp -lgtest -lgtest_main -pthread 2>&1 | tee /logs/verifier/compile_output.txt
    fi
    timeout 300 ./test_runner 2>&1 | tee /logs/verifier/test_output.txt
fi

exit ${PIPESTATUS[0]}
'''


def create_harbor_task_directory(
    output_dir: Path,
    task_id: int,
    instruction_content: str,
    test_code: str,
    dataset_prefix: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """Create a harbor-format task directory."""
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(create_cpp_dockerfile(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / "test_solution.cpp").write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    if metadata is None:
        metadata = {}
    metadata["source_dataset"] = "nemotron"
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_cpp_tasks(samples: List[Dict], task_descriptions: List[str], dataset_prefix: str = "nemotron-cpp") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, description) in enumerate(tqdm(zip(samples, task_descriptions), total=len(samples), desc="Creating tasks")):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "test_count": sample.get("test_count", 0),
        }
        create_harbor_task_directory(temp_dir, i, description, sample["text"], dataset_prefix, metadata)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main() -> None:
    """Main pipeline."""
    print("Step 1: Filtering C++ test files from Nemotron...")
    cpp_samples = filter_cpp_tests_from_nemotron(LIMIT)
    print(f"  -> {len(cpp_samples)} C++ test files found")

    if not cpp_samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Synthesizing tasks...")
    dataset = Dataset.from_list(cpp_samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": CPP_TEST_TO_TASK_PROMPT, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    task_descriptions = result.dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_cpp_tasks(cpp_samples, task_descriptions, "nemotron-cpp")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_nemotron-cpp")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(cpp_samples)} C++ tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
