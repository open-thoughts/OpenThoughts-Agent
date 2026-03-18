#!/usr/bin/env python3
"""
Generate tasks from UniTSyn dataset (2.7M focal-test pairs).

UniTSyn contains focal function + test function pairs from:
- Python (1.2M): pytest, unittest
- Java (1.1M): JUnit
- Go (361K): testing
- C++ (25K): GoogleTest
- JavaScript (13K): MochaJS
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
sys.path.append(str(Path(__file__).parent))  # Add dclm-mine directory

from dataset_utils import (
    download_github_archive,
    get_cache_dir,
    load_jsonl_file,
)
from data.completions import run_completions
from data.commons import (
    FOCAL_TO_INSTRUCTION_PROMPT,
    create_harbor_task_directory_generic,
    create_pytest_test_sh,
    create_generic_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"
DEFAULT_LANGUAGE = "python"

# Language-specific test command templates
TEST_COMMANDS = {
    "python": "pytest /tests/test_solution.py -v --tb=short",
    "java": "mvn test -Dtest=TestSolution",
    "go": "go test -v /tests/...",
    "cpp": "cd /tests && cmake . && make && ./run_tests",
    "javascript": "npx jest /tests/test_solution.js --verbose",
}


def get_test_sh_for_language(language: str) -> str:
    """Get test.sh content for a specific language."""
    language = language.lower()
    test_cmd = TEST_COMMANDS.get(language, TEST_COMMANDS["python"])

    setup_commands = ""
    if language == "python":
        setup_commands = """pip3 install --quiet pytest 2>/dev/null || true
# If agent didn't create solution.py, find the first .py file in /app and copy it
if [ ! -f /app/solution.py ]; then
    for f in /app/*.py; do
        [ -f "$f" ] && cp "$f" /app/solution.py && break
    done
fi"""
    elif language == "javascript":
        setup_commands = "npm install --silent jest 2>/dev/null || true"

    return create_generic_test_sh(test_command=test_cmd, setup_commands=setup_commands)


def _extract_test_code(test_data) -> str:
    """
    Extract valid test code from test data that may come in various formats.

    UniTSyn / KAKA22 datasets store test code in different ways:
    - A plain code string (the happy path)
    - A JSON-serialized string containing a list of test objects,
      e.g. '[{"ut_id": 0, "code": "import unittest\\n...", "FAR": ..., "FRR": ...}]'
    - A Python list of dicts with a "code" key
    - A Python list of code strings

    This function normalises all of these to a single code string.
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

    # Case 5: Plain code string (the normal/expected case)
    return test_data


def load_unitsyn_local(data_dir: Path, language: str = "python", limit: int = LIMIT) -> List[Dict]:
    """
    Load UniTSyn dataset from locally generated data.

    To generate UniTSyn data:
    1. git clone https://github.com/SecurityLab-UCD/UniTSyn
    2. cd UniTSyn
    3. python download_repos.py --lang python
    4. python mining.py --lang python

    Args:
        data_dir: Path to UniTSyn output directory containing JSONL files
        language: Programming language to load
        limit: Maximum samples to load
    """
    data_dir = Path(data_dir)

    # Look for language-specific JSONL files
    possible_files = [
        data_dir / f"{language}.jsonl",
        data_dir / f"{language}_tests.jsonl",
        data_dir / "output" / f"{language}.jsonl",
        data_dir / language / "tests.jsonl",
    ]

    for jsonl_file in possible_files:
        if jsonl_file.exists():
            return load_jsonl_file(jsonl_file, limit=limit)

    # Search for any JSONL files
    for jsonl_file in data_dir.rglob("*.jsonl"):
        if language in jsonl_file.name.lower():
            return load_jsonl_file(jsonl_file, limit=limit)

    raise FileNotFoundError(f"No UniTSyn data found for {language} in {data_dir}")


def load_unitsyn(language: str = "python", limit: int = LIMIT, local_data_dir: Path = None) -> List[Dict]:
    """
    Load UniTSyn dataset from HuggingFace or local sources.

    UniTSyn is a tool that generates focal-test pairs by mining GitHub repos.
    If not available on HuggingFace, it must be generated locally.

    Dataset structure:
    - focal_function: The function being tested
    - test_function: The test code
    - focal_class: Optional class containing the focal function
    - test_class: Optional test class

    Args:
        language: Programming language (python, java, go, cpp, javascript)
        limit: Maximum samples to load
        local_data_dir: Path to locally generated UniTSyn data
    """
    print(f"Loading UniTSyn dataset for {language}...")

    # Option 1: Load from local data directory if provided
    if local_data_dir and Path(local_data_dir).exists():
        print(f"Loading from local directory: {local_data_dir}")
        try:
            return load_unitsyn_local(Path(local_data_dir), language=language, limit=limit)
        except Exception as e:
            print(f"Error loading from local directory: {e}")

    # Option 2: Check cache for previously generated data
    cache_dir = get_cache_dir()
    cached_dir = cache_dir / "unitsyn" / language
    if cached_dir.exists():
        print(f"Loading from cached directory: {cached_dir}")
        try:
            return load_unitsyn_local(cached_dir, language=language, limit=limit)
        except Exception as e:
            print(f"Error loading from cache: {e}")

    # Option 3: Try HuggingFace sources (updated Jan 2026)
    # These are alternative datasets with similar focal-test pair structure
    repo_names = [
        "Arain/UnitTest-Finetuning",     # Alternative with Python/Java unit tests (1.48M samples)
        "KAKA22/CodeRM-UnitTest",         # Python unit tests (60k samples)
        "MERA-evaluation/UnitTests",      # Multi-language unit tests
        f"UNITSYN/unitsyn-{language}",
        f"SecurityLab-UCD/UniTSyn-{language}",
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
        # Fallback: try loading with language as config
        try:
            ds = load_dataset("SecurityLab-UCD/UniTSyn", language, split="train", streaming=True)
            print(f"Loaded from SecurityLab-UCD/UniTSyn ({language})")
        except Exception as e:
            print("")
            print("UniTSyn dataset not available on HuggingFace.")
            print("")
            print("UniTSyn must be generated locally from source:")
            print("  1. git clone https://github.com/SecurityLab-UCD/UniTSyn")
            print("  2. cd UniTSyn")
            print(f"  3. python download_repos.py --lang {language}")
            print(f"  4. python mining.py --lang {language}")
            print("  5. Run with --local-data /path/to/UniTSyn/output")
            print("")
            print("Alternatively, use one of these similar HuggingFace datasets:")
            print("  - Arain/UnitTest-Finetuning (1.48M Python/Java samples)")
            print("  - KAKA22/CodeRM-UnitTest (60k Python samples)")
            raise ValueError(f"Could not load UniTSyn dataset for {language}: {e}")

    samples = []
    print(f"Collecting {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        # Extract focal and test functions - handle multiple schema formats
        focal = (
            sample.get("focal_function") or
            sample.get("focal_method") or
            sample.get("focal") or
            sample.get("code_ground_truth") or  # KAKA22/CodeRM-UnitTest format
            ""
        )

        # Handle test code which might be a list or JSON
        test = sample.get("test_function") or sample.get("test_method") or sample.get("test") or ""

        # Handle KAKA22 format where unit_tests is a list of dicts / JSON string
        if not test and "unit_tests" in sample:
            test = sample["unit_tests"]

        # Extract actual code from test data that may be:
        # - A JSON-serialized list of test objects: '[{"ut_id": 0, "code": "..."}]'
        # - A Python list of test dicts: [{"ut_id": 0, "code": "..."}]
        # - A Python list of code strings: ["import unittest..."]
        # - Already a plain code string (no extraction needed)
        test = _extract_test_code(test)

        if not focal or not test:
            continue

        samples.append({
            "focal_function": focal,
            "test_function": test,
            "focal_class": sample.get("focal_class", ""),
            "test_class": sample.get("test_class", ""),
            "repo": sample.get("repo_name", sample.get("repository", "")),
            "path": sample.get("file_path", sample.get("path", "")),
            "language": language,
        })

    print(f"Loaded {len(samples)} focal-test pairs")
    return samples


def create_test_file_content(sample: Dict) -> str:
    """Create test file content from sample."""
    language = sample.get("language", "python")
    test_code = sample.get("test_function", "")
    test_class = sample.get("test_class", "")

    # Safety net: if test_code or test_class still contains embedded JSON
    # (e.g. a list of test objects), extract the actual code now.
    test_code = _extract_test_code(test_code)
    test_class = _extract_test_code(test_class)

    if language == "python":
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

        content_parts = []

        # Check if the test body already contains its own imports
        body_has_imports = any(
            body.lstrip().startswith(prefix)
            for prefix in ("import ", "from ")
        )

        if not body_has_imports:
            content_parts.append("import pytest")
            content_parts.append("import sys")
            content_parts.append("sys.path.insert(0, '/app')")
            content_parts.append("from solution import *")
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

    elif language == "java":
        # Wrap in JUnit test class if needed
        if "class" not in test_code:
            return f"""import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

public class TestSolution {{
    {test_code}
}}"""
        return test_code

    elif language == "go":
        if "package" not in test_code:
            return f"package main\n\nimport \"testing\"\n\n{test_code}"
        return test_code

    elif language == "javascript":
        return f"""const solution = require('/app/solution');
const {{ describe, it, expect }} = require('@jest/globals');

{test_code}"""

    return test_code


def get_test_filename(language: str) -> str:
    """Get test filename for language."""
    extensions = {
        "python": "test_solution.py",
        "java": "TestSolution.java",
        "go": "solution_test.go",
        "cpp": "test_solution.cpp",
        "javascript": "test_solution.js",
    }
    return extensions.get(language, "test_solution.py")


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a UniTSyn sample."""
    language = sample.get("language", "python")
    test_filename = get_test_filename(language)

    # Map language to dockerfile language
    dockerfile_lang = language
    if language == "javascript":
        dockerfile_lang = "node"

    try:
        dockerfile = get_dockerfile(dockerfile_lang)
    except ValueError:
        dockerfile = get_dockerfile("python")

    test_files = {
        test_filename: create_test_file_content(sample),
    }

    metadata = {
        "source": "unitsyn",
        "language": language,
        "repo": sample.get("repo", ""),
        "path": sample.get("path", ""),
    }

    # Include focal function as solution hint
    solution_files = None
    if sample.get("focal_function"):
        ext = {"python": "py", "java": "java", "go": "go", "cpp": "cpp", "javascript": "js"}.get(language, "py")
        solution_files = {f"solution.{ext}": sample["focal_function"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=get_test_sh_for_language(language),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "unitsyn",
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


def main(language: str = DEFAULT_LANGUAGE, limit: int = LIMIT, local_data_dir: Path = None, upload: bool = True) -> None:
    """Main pipeline for generating UniTSyn tasks."""

    print(f"Step 1: Loading UniTSyn dataset ({language})...")
    samples = load_unitsyn(language=language, limit=limit, local_data_dir=local_data_dir)
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
    dataset_prefix = f"unitsyn-{language}"
    task_dir = generate_tasks(samples, instructions, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 4: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} UniTSyn tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from UniTSyn dataset")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        choices=["python", "java", "go", "cpp", "javascript"],
                        help="Programming language")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--local-data", type=Path,
                        help="Path to locally generated UniTSyn data (from GitHub: SecurityLab-UCD/UniTSyn)")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")

    args = parser.parse_args()
    main(language=args.language, limit=args.limit, local_data_dir=args.local_data, upload=not args.no_upload)
