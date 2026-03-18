#!/usr/bin/env python3
"""
Generate tasks from Software Heritage archive (5B files, 80M projects).

Software Heritage provides:
- Largest public archive of source code
- Available on AWS S3 and Amazon Athena
- Test directories from millions of projects

This implementation uses The Stack (bigcode/the-stack) which contains code
archived from Software Heritage and GitHub.
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
    TEST_TO_INSTRUCTION_PROMPT,
    create_harbor_task_directory_generic,
    create_pytest_test_sh,
    create_generic_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 1000
MODEL = "gpt-4o-mini"


def is_python_test_file(content: str) -> bool:
    """Check if content is a Python test file (pytest or unittest)."""
    has_test_func = 'def test_' in content
    has_test_class = 'class Test' in content
    has_assert = 'assert ' in content
    has_unittest = 'import unittest' in content or 'from unittest' in content
    has_pytest = 'import pytest' in content or 'from pytest' in content

    # Must have test functions/classes
    if not (has_test_func or has_test_class):
        return False

    # Must have assertions or be unittest based
    if not (has_assert or has_unittest):
        return False

    # Reject files with relative imports - they depend on helper files we don't have
    has_relative_import = 'from .' in content or 'import .' in content
    if has_relative_import:
        return False

    # Reject files with complex external dependencies
    problematic_imports = [
        'ddtrace', 'celery', 'django', 'flask', 'sqlalchemy', 'boto3', 'aws',
        'google.cloud', 'azure', 'kubernetes', 'docker', 'ansible', 'terraform',
        'pyspark', 'tensorflow', 'torch', 'keras', 'pandas', 'numpy', 'scipy',
        'matplotlib', 'seaborn', 'sklearn', 'cv2', 'PIL', 'opencv',
    ]
    for pkg in problematic_imports:
        if f'import {pkg}' in content or f'from {pkg}' in content:
            return False

    return True


def load_softwareheritage(limit: int = LIMIT, max_scan: int = 500_000) -> List[Dict]:
    """
    Load test files from Software Heritage via The Stack.

    The Stack contains code archived from Software Heritage and GitHub.
    We filter for Python test files with proper test patterns.

    Args:
        limit: Maximum number of test samples to collect
        max_scan: Maximum number of samples to scan before stopping

    Returns:
        List of test file samples with content and metadata
    """
    print("Loading Software Heritage dataset via The Stack...")

    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/python',
        split='train',
        streaming=True
    )
    print("Loaded from bigcode/the-stack (Python subset)")

    samples = []
    scanned = 0

    print(f"Scanning for Python test files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding test files")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')
        path = sample.get('path', '').lower()

        # Skip very short files
        if len(content) < 200:
            continue

        # Check if it's a test file based on content
        if not is_python_test_file(content):
            continue

        samples.append({
            "text": content,
            "path": path,
            "language": "python",
            "repo": sample.get("repository_name", sample.get("repo", "")),
            "size": sample.get("size", len(content)),
        })
        pbar.update(1)

        if len(samples) >= limit:
            break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(samples)} Python test files")
    return samples


def create_test_file_content(sample: Dict) -> str:
    """Process test file content for harbor format."""
    content = sample.get("text", "")
    language = sample.get("language", "python")

    if language == "python":
        # Add sys.path for importing from /app
        if "sys.path" not in content:
            content = "import sys\nsys.path.insert(0, '/app')\n\n" + content
        return content

    elif language == "javascript":
        # Add require path for /app
        if "require('/app" not in content and "require(\"/app" not in content:
            content = "const solution = require('/app/solution');\n\n" + content
        return content

    elif language == "go":
        # Add package declaration if missing
        if "package" not in content:
            content = "package main\n\nimport \"testing\"\n\n" + content
        return content

    return content


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory from SWH test file."""
    language = sample.get("language", "python")

    lang_config = {
        "python": ("python", create_pytest_test_sh("/tests/test_solution.py"), "test_solution.py"),
        "javascript": ("node", create_generic_test_sh("npx jest /tests/test_solution.js"), "test_solution.js"),
        "go": ("go", create_generic_test_sh("go test -v /tests/..."), "solution_test.go"),
        "java": ("java", create_generic_test_sh("mvn test -q"), "TestSolution.java"),
        "rust": ("rust", create_generic_test_sh("cargo test"), "tests.rs"),
        "ruby": ("ruby", create_generic_test_sh("ruby /tests/test_solution.rb"), "test_solution.rb"),
    }

    dockerfile_lang, test_sh, test_filename = lang_config.get(
        language, lang_config["python"]
    )

    try:
        dockerfile = get_dockerfile(dockerfile_lang)
    except ValueError:
        dockerfile = get_dockerfile("python")

    test_files = {
        test_filename: create_test_file_content(sample),
    }

    metadata = {
        "source": "softwareheritage",
        "original_path": sample.get("path", ""),
        "repo": sample.get("repo", ""),
        "language": language,
        "size": sample.get("size", 0),
    }

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=test_sh,
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "swh",
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
    """Main pipeline for generating Software Heritage tasks."""

    # Scale max_scan based on limit (need ~50x scan to find test files)
    max_scan = max(500_000, limit * 50)

    print("Step 1: Loading Software Heritage test files...")
    samples = load_softwareheritage(limit=limit, max_scan=max_scan)
    print(f"  -> {len(samples)} test files loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating task descriptions...")
    dataset = Dataset.from_list(samples)

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": TEST_TO_INSTRUCTION_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    instructions = result.dataset["task_description"]
    print(f"  -> Generated {len(instructions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, "swh")
    print(f"  -> Task directory: {task_dir}")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_softwareheritage")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} Software Heritage tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from Software Heritage")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum files to process")

    args = parser.parse_args()
    main(limit=args.limit)
