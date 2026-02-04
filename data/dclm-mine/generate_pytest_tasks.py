#!/usr/bin/env python3
"""
Generate pytest tasks from The Stack - filters for pytest content and synthesizes tasks.
Creates harbor-format tasks with test.sh that runs the original pytest tests.
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
from data.commons import create_standard_dockerfile, create_standard_task_toml, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10_000
MODEL = "gpt-5-nano-2025-08-07"

PYTEST_TO_TASK_PROMPT = """You are an expert at creating coding tasks from pytest test code.

Given the following pytest code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable

Pytest code:
{{text}}

Create a task description (just the task, no preamble):"""


def is_pytest_file(content: str) -> bool:
    """Check if content is a pytest test file."""
    import re
    has_import_pytest = 'import pytest' in content or 'from pytest' in content
    has_test_func = 'def test_' in content
    has_assert = 'assert ' in content

    if not (has_import_pytest and has_test_func and has_assert):
        return False

    # Reject files with relative imports - they depend on helper files we don't have
    has_relative_import = 'from .' in content or 'import .' in content
    if has_relative_import:
        return False

    # Reject files that import local modules (non-pip-installable)
    # Look for imports that seem like local project imports
    imports = re.findall(r'^(?:from|import)\s+(\w+)', content, re.MULTILINE)

    # Known pip-installable packages (stdlib + common test/utility packages)
    pip_installable = {
        # Python stdlib
        'pytest', 'os', 'sys', 're', 'json', 'math', 'collections', 'itertools',
        'functools', 'typing', 'unittest', 'pathlib', 'datetime', 'time', 'copy',
        'io', 'string', 'random', 'abc', 'contextlib', 'warnings', 'operator',
        'dataclasses', 'enum', 'textwrap', 'inspect', 'types', 'numbers', 'decimal',
        'fractions', 'statistics', 'heapq', 'bisect', 'array', 'weakref', 'pprint',
        'struct', 'codecs', 'argparse', 'logging', 'subprocess', 'socket', 'asyncio',
        'threading', 'queue', 'pickle', 'sqlite3', 'csv', 'configparser', 'hashlib',
        'hmac', 'secrets', 'html', 'xml', 'base64', 'difflib', 'urllib', 'http',
        'uuid', 'tempfile', 'shutil', 'glob', 'fnmatch', 'zipfile', 'tarfile', 'gzip',
        'multiprocessing', 'concurrent', 'contextvars', 'traceback', 'linecache',
        'reprlib', 'cProfile', 'profile', 'timeit', 'trace', 'gc', 'dis', 'pickletools',
        # Common pip packages
        'mock', 'pytest_mock', 'freezegun', 'responses', 'requests_mock',
        'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'sklearn', 'torch',
        'tensorflow', 'keras', 'requests', 'httpx', 'aiohttp', 'flask', 'django',
        'fastapi', 'sqlalchemy', 'alembic', 'celery', 'redis', 'pymongo',
        'boto3', 'botocore', 'click', 'typer', 'pydantic', 'attrs', 'marshmallow',
        'yaml', 'toml', 'dotenv', 'jinja2', 'lxml', 'bs4', 'beautifulsoup4',
        'PIL', 'pillow', 'cv2', 'opencv', 'networkx', 'sympy', 'nltk', 'spacy',
        'transformers', 'datasets', 'tokenizers', 'accelerate', 'tqdm', 'rich',
        'loguru', 'structlog', 'pytest_asyncio', 'pytest_cov', 'coverage', 'hypothesis',
        'faker', 'factory_boy', 'parameterized', 'nose', 'nose2',
    }

    # Count how many imports look like local project imports
    local_import_count = 0
    for imp in imports:
        if imp not in pip_installable:
            # Heuristic: local imports often have underscores or are short lowercase names
            # Skip if it looks like a common package pattern
            local_import_count += 1

    # Allow up to 2 non-recognized imports (might be pip packages we don't know)
    # But reject if too many (likely local project imports)
    if local_import_count > 2:
        return False

    return True


def filter_pytest_from_stack(limit: int, max_scan: int = 5_000_000) -> List[Dict]:
    """
    Filter The Stack for pytest content.

    Args:
        limit: Maximum number of pytest samples to collect
        max_scan: Maximum number of samples to scan before stopping

    Returns:
        List of pytest samples with content and metadata
    """
    print(f"Loading The Stack (Python subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/python',
        split='train',
        streaming=True
    )

    pytest_samples = []
    scanned = 0

    print(f"Scanning for pytest files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding pytest files")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')

        if is_pytest_file(content):
            pytest_samples.append({
                'text': content,
                'path': sample.get('path', ''),
                'repo': sample.get('repository_name', ''),
                'size': sample.get('size', 0),
            })
            pbar.update(1)

            if len(pytest_samples) >= limit:
                break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(pytest_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(pytest_samples)} pytest files")
    return pytest_samples


def create_test_sh(test_filename: str = "test_solution.py") -> str:
    """Create test.sh that runs pytest and reports results."""
    return f'''#!/bin/bash
set -e

# Ensure logs directory exists
mkdir -p /logs/verifier

# Cleanup handler - always write reward file on exit
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

# Change to app directory
cd /app

# Run pytest and capture output
echo "Running pytest..."
pytest /tests/{test_filename} -v --tb=short 2>&1 | tee /logs/verifier/pytest_output.txt

# Check pytest exit code
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
    pytest_code: str,
    dataset_prefix: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Create a harbor-format task directory with pytest tests.

    Args:
        output_dir: Parent directory for tasks
        task_id: Unique task identifier
        instruction_content: Task description for instruction.md
        pytest_code: Original pytest test code
        dataset_prefix: Prefix for task directory name
        metadata: Optional metadata dict

    Returns:
        Path to created task directory
    """
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create environment directory with Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(create_standard_dockerfile(), encoding="utf-8")

    # Create tests directory with test.sh and the pytest file
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh("test_solution.py"), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)

    # Write the original pytest code as the test file
    test_py_path = tests_dir / "test_solution.py"
    test_py_path.write_text(pytest_code, encoding="utf-8")

    # Create instruction.md
    instruction_path = task_dir / "instruction.md"
    instruction_path.write_text(instruction_content, encoding="utf-8")

    # Create task.toml
    task_toml_path = task_dir / "task.toml"
    task_toml_path.write_text(create_standard_task_toml(), encoding="utf-8")

    # Create metadata.json if provided
    if metadata is not None:
        metadata_path = task_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_pytest_tasks(
    pytest_samples: List[Dict],
    task_descriptions: List[str],
    dataset_prefix: str = "stack-pytest",
) -> str:
    """
    Generate harbor-format task directories from pytest samples and descriptions.

    Args:
        pytest_samples: List of dicts with 'text', 'path', 'repo' keys
        task_descriptions: List of generated task descriptions
        dataset_prefix: Prefix for task directory names

    Returns:
        Path to the output directory containing all tasks
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, description) in enumerate(tqdm(
        zip(pytest_samples, task_descriptions),
        total=len(pytest_samples),
        desc="Creating task directories"
    )):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "source_size": sample.get("size", 0),
        }

        create_harbor_task_directory(
            output_dir=temp_dir,
            task_id=i,
            instruction_content=description,
            pytest_code=sample["text"],
            dataset_prefix=dataset_prefix,
            metadata=metadata,
        )

    print(f"Generated {len(pytest_samples)} harbor tasks successfully!")
    return str(temp_dir)


def main() -> None:
    """Main pipeline for generating pytest tasks from The Stack."""

    # Step 1: Filter for pytest content from The Stack
    print("Step 1: Filtering pytest files from The Stack...")
    pytest_samples = filter_pytest_from_stack(LIMIT)
    print(f"  -> {len(pytest_samples)} pytest files found")

    if not pytest_samples:
        print("\nNo pytest samples found. Exiting.")
        return

    # Step 2: Synthesize tasks from pytest using run_completions
    print("\nStep 2: Synthesizing tasks from pytest code...")
    dataset = Dataset.from_list(pytest_samples)

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
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    # Access the dataset from CuratorResponse
    tasks_dataset = result.dataset
    task_descriptions = tasks_dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    # Step 3: Generate harbor-format task directories
    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_pytest_tasks(pytest_samples, task_descriptions, "stack-pytest")
    print(f"  -> Task directory: {task_dir}")

    # Step 4: Upload to HuggingFace
    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-pytest")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated and uploaded {len(pytest_samples)} pytest tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
