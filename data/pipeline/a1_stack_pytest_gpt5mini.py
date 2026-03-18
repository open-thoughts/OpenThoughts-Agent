#!/usr/bin/env python3
"""
Generate tasks from pytest test files in The Stack using GPT-5-nano.
Finds pytest files and SYNTHETICALLY generates task descriptions from the test code.
This is for RL training - the model learns to write code that passes the tests.

Unlike generate_self_documented_tasks which EXTRACTS from existing docs,
this script GENERATES instructions purely from the test code.
"""

import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import create_standard_task_toml, create_standard_dockerfile, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10_000
MODEL = "gpt-5-mini-2025-08-07"

# =============================================================================
# LLM Prompt - Generate instructions from test code (NOT extraction)
# =============================================================================

PYTEST_TO_TASK_PROMPT = """You are an expert at creating coding tasks from test files.

Given the following pytest test file, create a task description for what needs to be implemented to make these tests pass. The task should:
1. Describe what functionality needs to be built
2. Specify expected inputs, outputs, and behavior based on what the tests check
3. Be clear about function/class names and signatures that need to be implemented
4. NOT reveal the exact test assertions, but give enough info to implement correctly
5. The solution should be placed in /app/

The pytest test file:
{{text}}

Create a task description (just the task, no preamble). The solution should be placed in /app/:"""


# =============================================================================
# Detection Functions
# =============================================================================


def is_pytest_file(content: str) -> Tuple[bool, Dict]:
    """
    Check if a file is a valid pytest test file for task generation.

    Returns (is_valid, metadata) where metadata contains detection info.
    """
    # Must have pytest imports or test functions
    has_pytest_import = 'import pytest' in content or 'from pytest' in content
    has_test_functions = bool(re.search(r'def test_\w+', content))

    if not has_test_functions:
        return False, {}

    # Count test functions
    test_functions = re.findall(r'def (test_\w+)', content)
    num_tests = len(test_functions)

    # Must have at least 2 test functions for meaningful task
    if num_tests < 2:
        return False, {}

    # Check for assertions (the actual test logic)
    has_assertions = bool(re.search(r'assert\s+', content))
    if not has_assertions:
        return False, {}

    # Count lines (not too short, not too long)
    lines = content.split('\n')
    code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith('#'))

    if code_lines < 10 or code_lines > 500:
        return False, {}

    # Check for fixtures (indicates more sophisticated tests)
    has_fixtures = bool(re.search(r'@pytest\.fixture', content))

    # Check for parametrize (indicates good test coverage)
    has_parametrize = bool(re.search(r'@pytest\.mark\.parametrize', content))

    # Extract what's being tested (imports from the module under test)
    tested_imports = re.findall(r'from\s+(\w+)\s+import|import\s+(\w+)', content)

    metadata = {
        'has_pytest_import': has_pytest_import,
        'num_tests': num_tests,
        'test_functions': test_functions[:10],  # First 10 test names
        'has_fixtures': has_fixtures,
        'has_parametrize': has_parametrize,
        'code_lines': code_lines,
    }

    return True, metadata


def filter_pytest_files_from_stack(limit: int, max_scan: int = 5_000_000) -> List[Dict]:
    """Filter The Stack for pytest test files."""
    print("Loading The Stack (Python subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/python',
        split='train',
        streaming=True
    )

    samples = []
    scanned = 0

    print(f"Scanning for pytest files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding pytest files")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')
        path = sample.get('path', '').lower()

        # Check content - don't require path to have 'test' since The Stack paths vary
        is_valid, metadata = is_pytest_file(content)

        if is_valid:
            samples.append({
                'text': content,
                'path': sample.get('path', ''),
                'repo': sample.get('repository_name', ''),
                'size': sample.get('size', 0),
                **metadata,
            })
            pbar.update(1)

            if len(samples) >= limit:
                break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(samples)} pytest files")
    return samples


# =============================================================================
# Task Generation
# =============================================================================


def create_test_sh(test_filename: str = "test_solution.py") -> str:
    """Create test.sh that runs pytest with venv for PEP 668 compatibility."""
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

# Create virtual environment if needed (fixes PEP 668 on Python 3.12+)
if [ ! -d /app/.venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv /app/.venv
fi

# Activate virtual environment
source /app/.venv/bin/activate

# Install pytest if not available
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install --quiet pytest
fi

# Auto-detect and install missing imports from test file
echo "Checking for missing dependencies..."
MISSING=$(python3 -c "
import re, sys
with open('/tests/{test_filename}') as f:
    for line in f:
        m = re.match(r'^(?:from|import)\\s+(\\w+)', line.strip())
        if m:
            mod = m.group(1)
            if mod in ('pytest', 'sys', 'os', 're', 'json', 'math', 'collections', 'itertools', 'functools', 'typing', 'unittest', 'pathlib', 'datetime', 'time', 'copy', 'io', 'string', 'random', 'abc', 'contextlib', 'warnings', 'operator', 'dataclasses', 'enum', 'textwrap', 'inspect', 'types', 'numbers', 'decimal', 'fractions', 'statistics', 'heapq', 'bisect', 'array', 'weakref', 'pprint', 'reprlib', 'struct', 'codecs', 'locale', 'gettext', 'argparse', 'optparse', 'logging', 'errno', 'signal', 'subprocess', 'socket', 'select', 'selectors', 'asyncio', 'threading', 'multiprocessing', 'concurrent', 'queue', 'sched', 'contextvars', 'pickle', 'shelve', 'dbm', 'sqlite3', 'csv', 'configparser', 'tomllib', 'netrc', 'plistlib', 'hashlib', 'hmac', 'secrets', 'html', 'xml', 'base64', 'binascii', 'quopri', 'uu', 'difflib', 'cgi', 'cgitb', 'wsgiref', 'urllib', 'http', 'ftplib', 'poplib', 'imaplib', 'smtplib', 'uuid', 'telnetlib', 'mailbox', 'mimetypes', 'email', 'mailcap', 'audioop', 'wave', 'colorsys', 'imghdr', 'sndhdr', 'getpass', 'curses', 'platform', 'ctypes', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'shutil', 'glob', 'fnmatch', 'linecache', 'fileinput', 'tempfile', 'filecmp', 'stat', 'test'):
                continue
            try:
                __import__(mod)
            except ImportError:
                print(mod)
" 2>/dev/null)
[ -n "$MISSING" ] && pip install --quiet $MISSING 2>/dev/null || true

cd /app

echo "Running tests..."
pytest /tests/{test_filename} -v --tb=short 2>&1 | tee /logs/verifier/test_output.txt

PYTEST_EXIT=${{PIPESTATUS[0]}}

if [ $PYTEST_EXIT -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed."
    exit 1
fi
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

    # Create environment directory
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(create_standard_dockerfile(), encoding="utf-8")

    # Create tests directory
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    # Write test.sh
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)

    # Write the actual pytest file
    (tests_dir / "test_solution.py").write_text(test_code, encoding="utf-8")

    # Write instruction.md
    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")

    # Write task.toml
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    # Write metadata
    if metadata is None:
        metadata = {}
    metadata["generation_model"] = MODEL
    metadata["generation_type"] = "synthetic"  # NOT extraction
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_tasks(samples: List[Dict], instructions: List[str], dataset_prefix: str = "stack-pytest-synthetic") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(zip(samples, instructions), total=len(samples), desc="Creating tasks")):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "num_tests": sample.get("num_tests", 0),
            "test_functions": sample.get("test_functions", []),
            "has_fixtures": sample.get("has_fixtures", False),
            "has_parametrize": sample.get("has_parametrize", False),
            "code_lines": sample.get("code_lines", 0),
        }
        create_harbor_task_directory(temp_dir, i, instruction, sample["text"], dataset_prefix, metadata)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


# =============================================================================
# Main Pipeline
# =============================================================================


def main() -> None:
    """Main pipeline - find pytest files and SYNTHETICALLY generate instructions."""

    print("=" * 60)
    print("Pytest Tasks Generator (SYNTHETIC - not extraction)")
    print(f"Model: {MODEL}")
    print("=" * 60)

    # Step 1: Find pytest files
    print("\nStep 1: Finding pytest files from The Stack...")
    samples = filter_pytest_files_from_stack(LIMIT)
    print(f"  -> {len(samples)} pytest files found")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    # Step 2: SYNTHETICALLY generate task descriptions from test code
    print(f"\nStep 2: Generating task descriptions SYNTHETICALLY with {MODEL}...")
    print("  (NOT extracting from docs - generating from test code)")
    dataset = Dataset.from_list(samples)

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": PYTEST_TO_TASK_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
        require_all_responses=False,
    )
    instructions = result.dataset["task_description"]
    print(f"  -> Generated {len(instructions)} task descriptions")

    # Step 3: Create harbor task directories
    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, "stack-pytest-synthetic")
    print(f"  -> Task directory: {task_dir}")

    # Step 4: Upload to HuggingFace
    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-pytest-gpt5mini")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} pytest tasks (SYNTHETIC)!")
    print(f"Model used: {MODEL}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
