#!/usr/bin/env python3
"""
Generate tasks from pyMethods2Test dataset (2M+ Python focal-test pairs).

pyMethods2Test contains Python method-test pairs mined from GitHub repositories.
Each entry has:
- focal_method: The Python method being tested
- test_method: The pytest/unittest test for that method
- context: Optional class/module context
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
    download_pymethods2test,
    download_from_zenodo,
    extract_archive,
    get_cache_dir,
    load_pymethods2test_from_zenodo,
)
from data.completions import run_completions
from data.commons import (
    FOCAL_TO_INSTRUCTION_PROMPT,
    create_harbor_task_directory_generic,
    create_pytest_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"


def _extract_test_code(test_data) -> str:
    """
    Extract valid Python test code from test data that may come in various formats.

    The pyMethods2Test / KAKA22 datasets store test code in different ways:
    - A plain Python code string (the happy path)
    - A JSON-serialized string containing a list of test objects,
      e.g. '[{"ut_id": 0, "code": "import unittest\\n...", "FAR": ..., "FRR": ...}]'
    - A Python list of dicts with a "code" key
    - A Python list of code strings

    This function normalises all of these to a single Python code string.
    """
    if not test_data:
        return ""

    # Case 1: Already a Python list
    if isinstance(test_data, list):
        if len(test_data) == 0:
            return ""
        first = test_data[0]
        if isinstance(first, dict):
            return first.get("code", "")
        if isinstance(first, str):
            return first
        return ""

    # Case 2: Not a string -- coerce and return
    if not isinstance(test_data, str):
        return str(test_data)

    # Case 3: String that is a JSON array of test objects
    stripped = test_data.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list) and len(parsed) > 0:
                first = parsed[0]
                if isinstance(first, dict) and "code" in first:
                    return first["code"]
                if isinstance(first, str):
                    return first
        except (json.JSONDecodeError, TypeError):
            pass  # Not valid JSON -- fall through and treat as raw code

    # Case 4: String that is a single JSON object with a "code" key
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "code" in parsed:
                return parsed["code"]
        except (json.JSONDecodeError, TypeError):
            pass

    # Case 5: Plain Python code string (the normal/expected case)
    return test_data


def load_pymethods2test(limit: int = LIMIT, local_data_dir: Path = None) -> List[Dict]:
    """
    Load pyMethods2Test dataset from HuggingFace or Zenodo.

    The dataset is available at:
    - HuggingFace: various community uploads
    - Zenodo: https://zenodo.org/records/14264519

    Args:
        limit: Maximum samples to load
        local_data_dir: Path to locally downloaded/extracted Zenodo data
    """
    print("Loading pyMethods2Test dataset...")

    # Option 1: Load from local data directory if provided
    if local_data_dir and Path(local_data_dir).exists():
        print(f"Loading from local directory: {local_data_dir}")
        try:
            return load_pymethods2test_from_zenodo(Path(local_data_dir), limit=limit)
        except Exception as e:
            print(f"Error loading from local directory: {e}")

    # Option 2: Check cache for previously downloaded data
    cache_dir = get_cache_dir()
    cached_dir = cache_dir / "zenodo_14264519" / "focal-data"
    if cached_dir.exists():
        print(f"Loading from cached directory: {cached_dir}")
        try:
            return load_pymethods2test_from_zenodo(cached_dir, limit=limit)
        except Exception as e:
            print(f"Error loading from cache: {e}")

    # Option 3: Try HuggingFace sources (updated Jan 2026)
    repo_names = [
        "Arain/UnitTest-Finetuning",              # Has Python method-test pairs
        "KAKA22/CodeRM-UnitTest",                  # Python unit tests
        "methods2test/pyMethods2Test",
        "pymethods2test/python-methods-tests",
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
        # Option 4: Try to download from Zenodo
        print("")
        print("HuggingFace sources unavailable. Attempting to download from Zenodo...")
        print("Source: https://zenodo.org/records/14264519")
        try:
            data_dir = download_pymethods2test()
            return load_pymethods2test_from_zenodo(data_dir, limit=limit)
        except Exception as e:
            print(f"Zenodo download failed: {e}")
            print("")
            print("To load pyMethods2Test manually:")
            print("  1. Download focal-data.zip from https://zenodo.org/records/14264519")
            print("  2. Extract the zip file")
            print("  3. Run with --local-data /path/to/focal-data")
            raise ValueError("Could not load pyMethods2Test dataset from any source")

    samples = []
    print(f"Collecting {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        # Extract focal and test methods - handle multiple schema formats
        focal = (
            sample.get("focal_method") or
            sample.get("focal_function") or
            sample.get("method") or
            sample.get("focal") or
            sample.get("code_ground_truth") or  # KAKA22/CodeRM-UnitTest format
            ""
        )

        # Handle test code which might be a list or JSON
        test = sample.get("test_method") or sample.get("test_function") or sample.get("test") or sample.get("test_code") or ""

        # Handle KAKA22 format where unit_tests is a list of dicts
        if not test and "unit_tests" in sample:
            test = sample["unit_tests"]

        # Extract actual Python code from test data that may be:
        # - A JSON-serialized list of test objects: '[{"ut_id": 0, "code": "..."}]'
        # - A Python list of test dicts: [{"ut_id": 0, "code": "..."}]
        # - A Python list of code strings: ["import unittest..."]
        # - Already a plain Python code string (no extraction needed)
        test = _extract_test_code(test)

        if not focal or not test:
            continue

        samples.append({
            "focal_function": focal,
            "test_function": test,
            "focal_class": sample.get("focal_class", sample.get("class_name", "")),
            "test_class": sample.get("test_class", ""),
            "repo": sample.get("repo_name", sample.get("repository", "")),
            "path": sample.get("file_path", sample.get("path", "")),
            "imports": sample.get("imports", ""),
            "docstring": sample.get("docstring", sample.get("question", "")),
        })

    print(f"Loaded {len(samples)} focal-test pairs")
    return samples


def create_test_file_content(sample: Dict) -> str:
    """Create pytest test file content from sample."""
    test_code = sample.get("test_function", "")
    test_class = sample.get("test_class", "")
    imports = sample.get("imports", "")

    # Safety net: if test_code or test_class still contains embedded JSON
    # (e.g. a list of test objects), extract the actual Python code now.
    test_code = _extract_test_code(test_code)
    test_class = _extract_test_code(test_class)

    # Pick whichever is non-empty (prefer test_class since it may include the class wrapper)
    body = test_class if test_class else test_code
    if not body:
        # Last resort: produce a placeholder that pytest will collect (and skip)
        body = (
            "import pytest\n\n"
            "@pytest.mark.skip(reason='no test code extracted')\n"
            "def test_placeholder():\n"
            "    pass\n"
        )

    # Build the test file
    content_parts = []

    # Check if the test body already contains its own imports (common in
    # pyMethods2Test data where each test is a self-contained file).
    body_has_imports = any(
        body.lstrip().startswith(prefix)
        for prefix in ("import ", "from ")
    )

    if not body_has_imports:
        # Add standard imports only when the body doesn't bring its own
        content_parts.append("import pytest")
        content_parts.append("import sys")
        content_parts.append("sys.path.insert(0, '/app')")
        content_parts.append("from solution import *")

        # Add any additional imports from the sample
        if imports:
            content_parts.append(imports)

        content_parts.append("")
    elif "sys.path" not in body:
        # Body has its own imports but no sys.path -- prepend it so /app is importable
        content_parts.append("import sys")
        content_parts.append("sys.path.insert(0, '/app')")
        content_parts.append("from solution import *")
        content_parts.append("")
    else:
        # Body has its own imports and sys.path -- ensure it also imports from solution
        if "from solution import" not in body:
            body = body.replace(
                "sys.path.insert(0, '/app')",
                "sys.path.insert(0, '/app')\nfrom solution import *",
            )

    content_parts.append(body)

    return "\n".join(content_parts)


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a pyMethods2Test sample."""
    dockerfile = get_dockerfile("python")

    test_files = {
        "test_solution.py": create_test_file_content(sample),
    }

    metadata = {
        "source": "pymethods2test",
        "repo": sample.get("repo", ""),
        "path": sample.get("path", ""),
        "has_docstring": bool(sample.get("docstring")),
    }

    # Include focal function as solution hint
    solution_files = None
    if sample.get("focal_function"):
        solution_files = {"solution.py": sample["focal_function"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=create_pytest_test_sh(),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "pymethods2test",
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


def main(limit: int = LIMIT, local_data_dir: Path = None, upload: bool = True) -> None:
    """Main pipeline for generating pyMethods2Test tasks."""

    print("Step 1: Loading pyMethods2Test dataset...")
    samples = load_pymethods2test(limit=limit, local_data_dir=local_data_dir)
    print(f"  -> {len(samples)} focal-test pairs loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating task descriptions from focal functions...")
    dataset = Dataset.from_list(samples)

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": FOCAL_TO_INSTRUCTION_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instructions = result.dataset["task_description"]
    print(f"  -> Generated {len(instructions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, "pymethods2test")
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 4: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_pymethods2test")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} pyMethods2Test tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from pyMethods2Test dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--local-data", type=Path,
                        help="Path to locally downloaded Zenodo data (from zenodo.org/records/14264519)")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")

    args = parser.parse_args()
    main(limit=args.limit, local_data_dir=args.local_data, upload=not args.no_upload)
