#!/usr/bin/env python3
"""
Generate C# test tasks from The Stack - filters for NUnit/xUnit/MSTest files.
"""

import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import create_standard_task_toml, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10_000
MODEL = "gpt-5-nano-2025-08-07"

CSHARP_TEST_TO_TASK_PROMPT = """You are an expert at creating coding tasks from C# test code.

Given the following C# test code (NUnit, xUnit, or MSTest), create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):"""


def is_csharp_test_file(content: str) -> bool:
    """Check if content is a C# test file."""
    # NUnit patterns
    has_nunit = 'using NUnit' in content
    has_test_attr = '[Test]' in content
    has_testcase = '[TestCase' in content

    # xUnit patterns
    has_xunit = 'using Xunit' in content
    has_fact = '[Fact]' in content
    has_theory = '[Theory]' in content

    # MSTest patterns
    has_mstest = 'using Microsoft.VisualStudio.TestTools' in content
    has_testmethod = '[TestMethod]' in content

    # Common assertions
    has_assert = 'Assert.' in content

    is_nunit = has_nunit or has_test_attr or has_testcase
    is_xunit = has_xunit or has_fact or has_theory
    is_mstest = has_mstest or has_testmethod

    return (is_nunit or is_xunit or is_mstest) and has_assert


def count_test_methods(content: str) -> int:
    """Count number of test methods."""
    test_count = len(re.findall(r'\[Test\]', content))
    fact_count = len(re.findall(r'\[Fact\]', content))
    theory_count = len(re.findall(r'\[Theory\]', content))
    testmethod_count = len(re.findall(r'\[TestMethod\]', content))
    return test_count + fact_count + theory_count + testmethod_count


def filter_csharp_tests_from_stack(limit: int, max_scan: int = 5_000_000) -> List[Dict]:
    """Filter The Stack for C# test content."""
    print("Loading The Stack (C# subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/c-sharp',
        split='train',
        streaming=True
    )

    csharp_samples = []
    scanned = 0

    print(f"Scanning for C# test files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding C# test files")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')
        path = sample.get('path', '')

        # Check content directly (path may be empty in The Stack)
        if is_csharp_test_file(content):
            test_count = count_test_methods(content)
            if test_count >= 2:
                csharp_samples.append({
                    'text': content,
                    'path': sample.get('path', ''),
                    'repo': sample.get('repository_name', ''),
                    'size': sample.get('size', 0),
                    'test_count': test_count,
                })
                pbar.update(1)

                if len(csharp_samples) >= limit:
                    break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(csharp_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(csharp_samples)} C# test files")
    return csharp_samples


def create_csharp_dockerfile() -> str:
    """Create C# Dockerfile for test environment."""
    return """FROM mcr.microsoft.com/dotnet/sdk:8.0

WORKDIR /app

# Install bash for Harbor agent
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*
"""


def create_test_sh() -> str:
    """Create test.sh that runs C# tests and reports results."""
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

# Create test project if no .csproj exists
if ! ls *.csproj 1> /dev/null 2>&1; then
    echo "Creating xunit test project..."
    dotnet new xunit -n TestProject --force
    cd TestProject
    # Copy test file
    cp /tests/TestSolution.cs .
    # Copy any implementation files from /app
    for f in /app/*.cs; do
        [ -f "$f" ] && [ "$(basename $f)" != "TestSolution.cs" ] && cp "$f" . 2>/dev/null || true
    done
fi

# Restore packages
dotnet restore 2>&1 || true

# Run tests
echo "Running C# tests..."
timeout 300 dotnet test --no-restore 2>&1 | tee /logs/verifier/test_output.txt

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
    (env_dir / "Dockerfile").write_text(create_csharp_dockerfile(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / "TestSolution.cs").write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    if metadata:
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_csharp_tasks(samples: List[Dict], task_descriptions: List[str], dataset_prefix: str = "stack-csharp") -> str:
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
    print("Step 1: Filtering C# test files from The Stack...")
    csharp_samples = filter_csharp_tests_from_stack(LIMIT)
    print(f"  -> {len(csharp_samples)} C# test files found")

    if not csharp_samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Synthesizing tasks...")
    dataset = Dataset.from_list(csharp_samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": CSHARP_TEST_TO_TASK_PROMPT, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    task_descriptions = result.dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_csharp_tasks(csharp_samples, task_descriptions, "stack-csharp")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-csharp-v2-test")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(csharp_samples)} C# tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
