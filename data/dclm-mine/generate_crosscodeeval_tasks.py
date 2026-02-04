#!/usr/bin/env python3
"""
Generate tasks from CrossCodeEval dataset (cross-file context tasks).

CrossCodeEval requires cross-file contextual understanding to solve.
Tasks cannot be solved with single-file context alone. The dataset uses
real open-source repositories with no overlap with The Stack to prevent
data leakage.

Languages: Python, Java, TypeScript, C#
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
sys.path.append(str(Path(__file__).parent))  # Add dclm-mine directory

from dataset_utils import (
    download_crosscodeeval,
    download_github_archive,
    get_cache_dir,
    load_crosscodeeval_local,
)
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

# Language-specific Dockerfiles
CROSSCODE_DOCKERFILES = {
    "python": """FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest pytest-timeout
RUN mkdir -p /logs/verifier /context
""",
    "java": """FROM openjdk:17-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git maven && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /logs/verifier /context
""",
    "typescript": """FROM node:18-slim
WORKDIR /app
RUN npm install -g typescript ts-node jest @types/jest
RUN mkdir -p /logs/verifier /context
""",
    "csharp": """FROM mcr.microsoft.com/dotnet/sdk:7.0
WORKDIR /app
RUN mkdir -p /logs/verifier /context
""",
}


def get_crosscode_dockerfile(language: str) -> str:
    """Get Dockerfile for CrossCodeEval language."""
    language = language.lower()
    if language in CROSSCODE_DOCKERFILES:
        return CROSSCODE_DOCKERFILES[language]
    return CROSSCODE_DOCKERFILES["python"]


def create_crosscode_test_sh(language: str) -> str:
    """Create test.sh for CrossCodeEval tasks."""
    test_commands = {
        "python": "pytest /tests/test_solution.py -v --tb=short --timeout=60",
        "java": "mvn test -Dtest=TestSolution",
        "typescript": "npx jest /tests/test_solution.ts --verbose",
        "csharp": "dotnet test",
    }

    test_cmd = test_commands.get(language.lower(), test_commands["python"])

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

cd /app

echo "Running CrossCodeEval tests ({language})..."
{test_cmd} 2>&1 | tee /logs/verifier/test_output.txt

exit ${{PIPESTATUS[0]}}
'''


def load_crosscodeeval(language: str = "python", limit: int = LIMIT, local_data_dir: Path = None) -> List[Dict]:
    """
    Load CrossCodeEval dataset from HuggingFace or local GitHub download.

    The dataset focuses on cross-file code completion tasks.

    Args:
        language: Programming language (python, java, typescript, csharp)
        limit: Maximum samples to load
        local_data_dir: Path to locally downloaded CrossCodeEval data.
                        Download from https://github.com/amazon-science/cceval
    """
    print(f"Loading CrossCodeEval dataset ({language})...")

    # Option 1: Load from local data directory if provided
    if local_data_dir and Path(local_data_dir).exists():
        print(f"Loading from local directory: {local_data_dir}")
        try:
            return load_crosscodeeval_local(Path(local_data_dir), language=language, limit=limit)
        except Exception as e:
            print(f"Error loading from local directory: {e}")

    # Option 2: Check cache for previously downloaded data
    cache_dir = get_cache_dir()
    cached_dir = cache_dir / "github" / "amazon-science_cceval"
    if cached_dir.exists():
        print(f"Loading from cached directory: {cached_dir}")
        try:
            return load_crosscodeeval_local(cached_dir, language=language, limit=limit)
        except Exception as e:
            print(f"Error loading from cache: {e}")

    # Option 3: Try HuggingFace sources (updated Jan 2026)
    repo_names = [
        ("NTU-NLP-sg/xCodeEval", language),  # Current working source
        ("microsoft/CrossCodeEval", language),
        ("CrossCodeEval/crosscodeeval", language),
    ]

    ds = None
    for repo_name, config in repo_names:
        try:
            ds = load_dataset(repo_name, config, split="test", streaming=True)
            print(f"Loaded from {repo_name} ({config})")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}/{config}: {e}")
            continue

    if ds is None:
        # Try without config
        for repo_name, _ in repo_names:
            try:
                ds = load_dataset(repo_name, split="test", streaming=True)
                print(f"Loaded from {repo_name} (no config)")
                break
            except Exception:
                continue

    if ds is None:
        # Option 4: Try to download from GitHub
        print("")
        print("HuggingFace sources unavailable. Attempting to download from GitHub...")
        print("Source: https://github.com/amazon-science/cceval")
        try:
            repo_dir = download_crosscodeeval()
            return load_crosscodeeval_local(repo_dir, language=language, limit=limit)
        except Exception as e:
            print(f"GitHub download failed: {e}")
            print("")
            print("To load CrossCodeEval manually:")
            print("  1. Clone https://github.com/amazon-science/cceval")
            print("  2. Extract crosscodeeval_data.tar.xz (if present)")
            print("  3. Run with --local-data /path/to/cceval/data")
            raise ValueError(f"Could not load CrossCodeEval dataset for {language}")

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        # CrossCodeEval structure varies, try to extract common fields
        task_id = sample.get("task_id", sample.get("id", ""))
        prompt = sample.get("prompt", sample.get("context", ""))
        groundtruth = sample.get("groundtruth", sample.get("canonical_solution", ""))
        cross_context = sample.get("cross_context", sample.get("cross_file_context", []))

        if not task_id or not prompt:
            continue

        # Ensure cross_context is a list
        if isinstance(cross_context, str):
            cross_context = [cross_context]
        elif cross_context is None:
            cross_context = []

        samples.append({
            "task_id": task_id,
            "prompt": prompt,
            "groundtruth": groundtruth,
            "cross_context": cross_context,
            "language": sample.get("language", language),
            "repo": sample.get("repo", sample.get("repository", "")),
            "file_path": sample.get("file_path", sample.get("path", "")),
            "test": sample.get("test", ""),
        })

    print(f"Loaded {len(samples)} CrossCodeEval tasks")
    return samples


def create_instruction_from_sample(sample: Dict) -> str:
    """Create task instruction from CrossCodeEval sample."""
    prompt = sample.get("prompt", "")
    cross_context = sample.get("cross_context", [])
    language = sample.get("language", "python")
    file_path = sample.get("file_path", "")

    instruction = f"""# Cross-File Code Completion Task

## Target File
{file_path if file_path else "solution file"}

## Context (Code to Complete)

```{language}
{prompt}
```

"""

    if cross_context:
        instruction += """## Cross-File Context

The following code from other files in the project provides necessary context:

"""
        for i, ctx in enumerate(cross_context[:5]):  # Limit to 5 context files
            instruction += f"""### Context File {i + 1}

```{language}
{ctx[:2000]}{"..." if len(ctx) > 2000 else ""}
```

"""

    instruction += f"""## Your Task

Complete the code above, taking into account the cross-file context provided.
Your solution must work correctly with the related files in the project.

Place your complete solution in `/app/solution.{_get_extension(language)}`
"""

    return instruction


def _get_extension(language: str) -> str:
    """Get file extension for language."""
    extensions = {
        "python": "py",
        "java": "java",
        "typescript": "ts",
        "csharp": "cs",
        "javascript": "js",
    }
    return extensions.get(language.lower(), "py")


def create_test_file_content(sample: Dict) -> str:
    """Create test file content from CrossCodeEval sample."""
    language = sample.get("language", "python")
    test_code = sample.get("test", "")
    groundtruth = sample.get("groundtruth", "")

    if language.lower() == "python":
        if test_code:
            return f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{test_code}
"""
        else:
            # Create basic import test if no test provided
            return f"""import pytest
import sys
sys.path.insert(0, '/app')

def test_solution_imports():
    '''Test that solution can be imported.'''
    import solution
    assert solution is not None

def test_solution_not_empty():
    '''Test that solution file has content.'''
    import solution
    import inspect
    members = inspect.getmembers(solution)
    assert len(members) > 0, "Solution should define functions or classes"
"""

    elif language.lower() == "java":
        return f"""import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

public class TestSolution {{
    {test_code if test_code else '''
    @Test
    void testSolutionExists() {
        // Basic test - solution should compile
        assertTrue(true);
    }
    '''}
}}
"""

    elif language.lower() in ("typescript", "javascript"):
        return f"""const solution = require('/app/solution');

{test_code if test_code else '''
describe('Solution', () => {
    it('should export something', () => {
        expect(solution).toBeDefined();
    });
});
'''}
"""

    return test_code


def create_context_files(sample: Dict) -> Dict[str, str]:
    """Create context files from cross_context."""
    context_files = {}
    cross_context = sample.get("cross_context", [])
    language = sample.get("language", "python")
    ext = _get_extension(language)

    for i, ctx in enumerate(cross_context[:5]):
        context_files[f"context_{i}.{ext}"] = ctx

    return context_files


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a CrossCodeEval sample."""
    language = sample.get("language", "python")
    instruction = create_instruction_from_sample(sample)

    # Test files
    test_files = {
        f"test_solution.{_get_extension(language)}": create_test_file_content(sample),
    }

    # Add context files
    context_files = create_context_files(sample)
    for filename, content in context_files.items():
        test_files[f"context/{filename}"] = content

    metadata = {
        "source": "crosscodeeval",
        "task_id": sample.get("task_id", ""),
        "language": language,
        "repo": sample.get("repo", ""),
        "file_path": sample.get("file_path", ""),
        "context_files_count": len(context_files),
    }

    # Store groundtruth as solution
    solution_files = None
    if sample.get("groundtruth"):
        ext = _get_extension(language)
        solution_files = {f"solution.{ext}": sample["groundtruth"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_crosscode_dockerfile(language),
        test_sh=create_crosscode_test_sh(language),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(samples: List[Dict], dataset_prefix: str = "crosscodeeval") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, sample in enumerate(tqdm(samples, desc="Creating tasks")):
        create_harbor_task(temp_dir, i, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(language: str = "python", limit: int = LIMIT, upload: bool = True, local_data_dir: Path = None) -> None:
    """Main pipeline for generating CrossCodeEval tasks."""

    print(f"Step 1: Loading CrossCodeEval dataset ({language})...")
    samples = load_crosscodeeval(language=language, limit=limit, local_data_dir=local_data_dir)
    print(f"  -> {len(samples)} tasks loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating harbor task directories...")
    dataset_prefix = f"crosscodeeval-{language}"
    task_dir = generate_tasks(samples, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 3: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} CrossCodeEval tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from CrossCodeEval dataset")
    parser.add_argument("--language", default="python",
                        choices=["python", "java", "typescript", "csharp"],
                        help="Programming language")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")
    parser.add_argument("--local-data", type=Path,
                        help="Path to locally downloaded CrossCodeEval data (from GitHub: amazon-science/cceval)")

    args = parser.parse_args()
    main(language=args.language, limit=args.limit, upload=not args.no_upload, local_data_dir=args.local_data)
