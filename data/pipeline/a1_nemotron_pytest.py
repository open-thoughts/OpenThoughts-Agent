#!/usr/bin/env python3
"""
Generate tasks from pytest test files in Nemotron using GPT-5-mini.
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

from datasets import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from nemotron_loader import load_nemotron_stream, normalize_sample
from data.completions import run_completions
from data.commons import create_standard_task_toml, create_standard_dockerfile, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 5_000
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

IMPORTANT constraints for the task description:
- Keep it simple: prefer a SINGLE Python file implementation (e.g., /app/solution.py or /app/<module_name>.py)
- If the tests import from a package like `from foo.bar import X`, instruct the user to create a minimal package structure with __init__.py files, but keep the actual logic in one file
- Only use Python standard library — no external dependencies
- Be very explicit about exact function/class names, parameter names, and return types
- Include concrete examples of expected behavior (input → output)
- Keep the description concise — under 60 lines

The pytest test file:
{{text}}

Create a task description (just the task, no preamble). The solution should be placed in /app/:"""


# =============================================================================
# Detection Functions
# =============================================================================


def is_pytest_file(content: str) -> Tuple[bool, Dict]:
    """
    Check if a file is a valid pytest test file for task generation.

    Applies strict filtering to reject overly complex tasks that an agent
    can't reasonably solve in 15 minutes. Nemotron's synthetic code tends
    to produce very complex test files (Django, rasterio, Selenium, etc.)
    so we need aggressive complexity checks.

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

    # Count lines (not too short, not too long - tighter upper bound)
    lines = content.split('\n')
    code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith('#'))

    if code_lines < 10 or code_lines > 200:
        return False, {}

    # --- COMPLEXITY FILTERS (key for Nemotron synthetic code) ---

    # 1. Validate Python syntax with AST parsing
    import ast
    try:
        ast.parse(content)
    except SyntaxError:
        return False, {}

    # 2. Reject files importing heavy/complex frameworks
    HEAVY_FRAMEWORKS = {
        'django', 'flask', 'fastapi', 'tornado', 'aiohttp', 'starlette',
        'selenium', 'playwright', 'cypress',
        'rasterio', 'gdal', 'osgeo', 'fiona', 'shapely', 'geopandas',
        'tensorflow', 'torch', 'keras', 'sklearn', 'scipy',
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'sqlalchemy', 'peewee', 'mongoengine',
        'celery', 'redis', 'kafka', 'rabbitmq',
        'boto3', 'azure', 'google.cloud',
        'docker', 'kubernetes', 'ansible',
        'slack_sdk', 'slack', 'discord', 'telegram',
        'requests', 'httpx', 'aiohttp', 'urllib3',
        'pydantic', 'marshmallow', 'attrs',
        'cryptography', 'jwt', 'oauth',
        'grpc', 'protobuf', 'thrift',
        'paramiko', 'fabric',
        'cv2', 'PIL', 'pillow', 'imageio',
        'lxml', 'beautifulsoup4', 'bs4', 'scrapy',
    }
    content_lower = content.lower()
    all_imports = re.findall(r'(?:from|import)\s+([\w.]+)', content)
    top_level_imports = {imp.split('.')[0].lower() for imp in all_imports}

    # Allow pytest and stdlib, reject heavy frameworks
    if top_level_imports & HEAVY_FRAMEWORKS:
        return False, {}

    # 3. Reject files with too many non-stdlib imports (complex dependencies)
    STDLIB_AND_TEST = {
        'pytest', 'sys', 'os', 're', 'json', 'math', 'collections',
        'itertools', 'functools', 'typing', 'unittest', 'pathlib',
        'datetime', 'time', 'copy', 'io', 'string', 'random', 'abc',
        'contextlib', 'warnings', 'operator', 'dataclasses', 'enum',
        'textwrap', 'inspect', 'types', 'numbers', 'decimal', 'fractions',
        'statistics', 'heapq', 'bisect', 'array', 'hashlib', 'hmac',
        'base64', 'struct', 'pickle', 'csv', 'tempfile', 'shutil',
        'glob', 'fnmatch', 'uuid', 'logging', 'argparse', 'configparser',
        'subprocess', 'threading', 'multiprocessing', 'concurrent',
        'asyncio', 'socket', 'queue', 'secrets',
    }
    non_stdlib_imports = top_level_imports - STDLIB_AND_TEST
    # Allow at most 1 non-stdlib import (the module being tested)
    if len(non_stdlib_imports) > 2:
        return False, {}

    # 4. Reject tests that need complex file I/O or external services
    COMPLEX_PATTERNS = [
        r'\.read_csv\(', r'\.to_csv\(',        # data files
        r'open\(.+\.tif', r'open\(.+\.geotiff', # geospatial files
        r'\.connect\(', r'\.cursor\(',           # database connections
        r'requests\.get\(', r'requests\.post\(', # HTTP calls
        r'mock\.patch',                          # heavy mocking
        r'subprocess\.run\(', r'subprocess\.Popen',  # shell calls
        r'@app\.route', r'@api\.',               # web framework decorators
        r'\.env\b',                              # environment files
    ]
    for pat in COMPLEX_PATTERNS:
        if re.search(pat, content):
            return False, {}

    # 5. Cap number of test functions (too many = too complex task)
    if num_tests > 15:
        return False, {}

    # 6. Reject files with class-based tests that indicate complex OOP
    test_classes = re.findall(r'class\s+Test\w+', content)
    if len(test_classes) > 1:
        return False, {}

    # Check for fixtures (indicates more sophisticated tests)
    has_fixtures = bool(re.search(r'@pytest\.fixture', content))

    # Check for parametrize (indicates good test coverage)
    has_parametrize = bool(re.search(r'@pytest\.mark\.parametrize', content))

    metadata = {
        'has_pytest_import': has_pytest_import,
        'num_tests': num_tests,
        'test_functions': test_functions[:10],  # First 10 test names
        'has_fixtures': has_fixtures,
        'has_parametrize': has_parametrize,
        'code_lines': code_lines,
        'non_stdlib_imports': list(non_stdlib_imports),
    }

    return True, metadata


def filter_pytest_files_from_nemotron(limit: int, max_scan: int = 50_000_000) -> List[Dict]:
    """Filter Nemotron for pytest test files."""
    print("Loading Nemotron (Python subset, streaming)...")
    ds = load_nemotron_stream("python")

    samples = []
    scanned = 0

    print(f"Scanning for pytest files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding pytest files")

    for raw_sample in itertools.islice(ds, max_scan):
        scanned += 1
        sample = normalize_sample(raw_sample)
        content = sample['content']

        # Check content - don't require path to have 'test' since paths vary
        is_valid, metadata = is_pytest_file(content)

        if is_valid:
            samples.append({
                'text': content,
                'path': sample['path'],
                'repo': sample['repo'],
                'size': sample['size'],
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
export PYTHONPATH=/app:$PYTHONPATH

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


def _build_import_hint(test_code: str) -> str:
    """Parse test imports and build a hint block showing exactly what files to create.

    Analyses ``from pkg.mod import ...`` and ``import mod`` lines in the test
    file and turns them into concrete file-path instructions the agent can
    follow.  Stdlib / pytest / typing imports are ignored.
    """
    SKIP_MODULES = {
        'pytest', 'sys', 'os', 're', 'json', 'math', 'collections',
        'itertools', 'functools', 'typing', 'unittest', 'pathlib',
        'datetime', 'time', 'copy', 'io', 'string', 'random', 'abc',
        'contextlib', 'warnings', 'operator', 'dataclasses', 'enum',
        'textwrap', 'inspect', 'types', 'numbers', 'decimal', 'fractions',
        'statistics', 'heapq', 'bisect', 'array', 'hashlib', 'hmac',
        'secrets', 'struct', 'codecs', 'locale', 'logging', 'pprint',
        'subprocess', 'threading', 'multiprocessing', 'concurrent',
        'asyncio', 'socket', 'uuid', 'tempfile', 'shutil', 'glob',
        'csv', 'configparser', 'pickle', 'sqlite3', 'xml', 'html',
        'http', 'urllib', 'email', 'base64', 'binascii', 'difflib',
        'argparse', 'signal', 'queue', 'weakref', 'platform', 'ctypes',
        'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'fnmatch',
    }

    files_needed: dict[str, list[str]] = {}  # path -> list of names imported

    # Join multi-line imports: "from x import (\n  A,\n  B\n)" -> single line
    joined = re.sub(r'\(\s*\n', '(', test_code)
    joined = re.sub(r',\s*\n\s*', ', ', joined)
    joined = re.sub(r'\s*\)', ')', joined)

    for line in joined.splitlines():
        line = line.strip()

        # from pkg.sub.mod import X, Y  (or with parens)
        m = re.match(r'^from\s+([\w.]+)\s+import\s+\(?\s*(.+?)\s*\)?\s*$', line)
        if m:
            module_path = m.group(1)
            raw_names = m.group(2)
            names = [n.strip().split(' as ')[0].strip() for n in raw_names.split(',') if n.strip() and re.match(r'^[A-Za-z_]\w*', n.strip())]
            top = module_path.split('.')[0]
            if top.lower() in SKIP_MODULES:
                continue
            # Convert dotted module to file path: foo.bar.baz -> foo/bar/baz.py
            parts = module_path.split('.')
            file_path = '/app/' + '/'.join(parts[:-1]) + '/' + parts[-1] + '.py' if len(parts) > 1 else '/app/' + parts[0] + '.py'
            files_needed[file_path] = names
            # Add __init__.py for packages
            for i in range(1, len(parts)):
                init_path = '/app/' + '/'.join(parts[:i]) + '/__init__.py'
                if init_path not in files_needed:
                    files_needed[init_path] = []
            continue

        # import mod (or import mod as alias)
        m = re.match(r'^import\s+([\w.]+)', line)
        if m:
            module_path = m.group(1)
            top = module_path.split('.')[0]
            if top.lower() in SKIP_MODULES:
                continue
            parts = module_path.split('.')
            file_path = '/app/' + '/'.join(parts) + '.py' if len(parts) == 1 else '/app/' + '/'.join(parts[:-1]) + '/' + parts[-1] + '.py'
            files_needed.setdefault(file_path, [])
            for i in range(1, len(parts)):
                init_path = '/app/' + '/'.join(parts[:i]) + '/__init__.py'
                if init_path not in files_needed:
                    files_needed[init_path] = []
            continue

    if not files_needed:
        return ""

    lines = ["\n\n---\n**Important: File structure required by the tests**\n"]
    lines.append("The test file will run from `/app/` and imports your code. You must create these exact files:\n")
    for path in sorted(files_needed.keys()):
        names = files_needed[path]
        if path.endswith('__init__.py'):
            lines.append(f"- `{path}` (can be empty)")
        elif names:
            lines.append(f"- `{path}` (must export: {', '.join(f'`{n}`' for n in names)})")
        else:
            lines.append(f"- `{path}`")
    lines.append("\nThe working directory is `/app/`. Tests run via `pytest /tests/test_solution.py` from `/app/`.")
    return '\n'.join(lines)


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

    # Append import hints to instruction so the agent knows exact file paths
    hint = _build_import_hint(test_code)
    final_instruction = instruction_content + hint

    # Write instruction.md
    (task_dir / "instruction.md").write_text(final_instruction, encoding="utf-8")

    # Write task.toml
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    # Write metadata
    if metadata is None:
        metadata = {}
    metadata["generation_model"] = MODEL
    metadata["generation_type"] = "synthetic"  # NOT extraction
    metadata["source_dataset"] = "nemotron"
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_tasks(samples: List[Dict], instructions: List[str], dataset_prefix: str = "nemotron-pytest-synthetic") -> str:
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
    print("\nStep 1: Finding pytest files from Nemotron...")
    samples = filter_pytest_files_from_nemotron(LIMIT)
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

    # Filter out tasks with empty/short instructions (LLM sometimes returns empty)
    valid_pairs = [
        (s, instr) for s, instr in zip(samples, instructions)
        if instr and len(instr.strip()) >= 50
    ]
    skipped = len(samples) - len(valid_pairs)
    if skipped:
        print(f"  -> Skipped {skipped} tasks with empty/short instructions")
    samples = [p[0] for p in valid_pairs]
    instructions = [p[1] for p in valid_pairs]
    print(f"  -> {len(samples)} valid tasks remaining")

    # Step 3: Create harbor task directories
    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, "nemotron-pytest-synthetic")
    print(f"  -> Task directory: {task_dir}")

    # Step 4: Upload to HuggingFace
    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_nemotron-pytest-gpt5mini")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} pytest tasks (SYNTHETIC)!")
    print(f"Model used: {MODEL}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
