#!/usr/bin/env python3
"""
Find and generate tasks from files that already contain BOTH:
1. Documentation/instructions (docstrings, comments, README-style text)
2. Test code (pytest, unittest, etc.)

These are "self-documented" test files where the task description is already present.
"""

import argparse
import ast
import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import create_standard_task_toml, create_standard_dockerfile, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10_000
MODEL = "gpt-5-nano-2025-08-07"

# Prompt to clean up/extract the instruction from the file
EXTRACT_INSTRUCTION_PROMPT = """You are an expert at extracting task descriptions from documented test files.

The following file contains BOTH documentation/instructions AND test code. Extract or synthesize a clear task description from the documentation. The task should:
1. Describe what functionality needs to be implemented
2. Be based on the docstrings, comments, or README-style text in the file
3. NOT include the test code itself in the description
4. Be self-contained and actionable
5. The solution should be placed in /app/

File content:
{{text}}

Extract the task description (just the task, no preamble):"""


# =============================================================================
# Standard library modules (should NOT be pip-installed)
# =============================================================================
STDLIB_MODULES = {
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
    "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
    "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
    "colorsys", "compileall", "concurrent", "configparser", "contextlib",
    "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
    "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal",
    "difflib", "dis", "distutils", "doctest", "email", "encodings",
    "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "formatter", "fractions", "ftplib", "functools", "gc",
    "getopt", "getpass", "gettext", "glob", "grp", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
    "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
    "json", "keyword", "lib2to3", "linecache", "locale", "logging",
    "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes",
    "mmap", "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
    "numbers", "operator", "optparse", "os", "ossaudiodev", "parser",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
    "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
    "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib",
    "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site",
    "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd",
    "sqlite3", "sre_compile", "sre_constants", "sre_parse", "ssl",
    "stat", "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo",
    "types", "typing", "unicodedata", "unittest", "urllib", "uu",
    "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib",
    # Also include common sub-modules
    "__future__", "_pytest", "pytest", "unittest.mock",
    "collections.abc", "os.path", "typing_extensions",
    "io", "abc", "contextlib",
}

# Known pip-installable packages (module name -> pip package name)
KNOWN_PIP_PACKAGES = {
    "numpy": "numpy",
    "np": "numpy",
    "pandas": "pandas",
    "pd": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "keras": "keras",
    "flask": "flask",
    "django": "django",
    "fastapi": "fastapi",
    "requests": "requests",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "pydantic": "pydantic",
    "yaml": "pyyaml",
    "pyyaml": "pyyaml",
    "PIL": "pillow",
    "pillow": "pillow",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sqlalchemy": "sqlalchemy",
    "celery": "celery",
    "redis": "redis",
    "pymongo": "pymongo",
    "boto3": "boto3",
    "botocore": "botocore",
    "click": "click",
    "rich": "rich",
    "tqdm": "tqdm",
    "pytest": "pytest",
    "mock": "mock",
    "six": "six",
    "attr": "attrs",
    "attrs": "attrs",
    "marshmallow": "marshmallow",
    "jinja2": "jinja2",
    "Jinja2": "jinja2",
    "werkzeug": "werkzeug",
    "lxml": "lxml",
    "bs4": "beautifulsoup4",
    "beautifulsoup4": "beautifulsoup4",
    "cryptography": "cryptography",
    "jwt": "pyjwt",
    "pytz": "pytz",
    "dateutil": "python-dateutil",
    "paramiko": "paramiko",
    "fabric": "fabric",
    "toml": "toml",
    "tomli": "tomli",
    "tomllib": "tomli",
    "msgpack": "msgpack",
    "protobuf": "protobuf",
    "grpc": "grpcio",
    "graphene": "graphene",
    "hypothesis": "hypothesis",
    "faker": "faker",
    "docutils": "docutils",
    "sphinx": "sphinx",
    "coverage": "coverage",
    "mypy": "mypy",
    "black": "black",
    "isort": "isort",
    "aioresponses": "aioresponses",
    "responses": "responses",
    "freezegun": "freezegun",
    "arrow": "arrow",
    "pendulum": "pendulum",
    "ujson": "ujson",
    "orjson": "orjson",
    "jsonschema": "jsonschema",
    "voluptuous": "voluptuous",
    "cerberus": "cerberus",
    "dynaconf": "dynaconf",
    "dotenv": "python-dotenv",
    "decouple": "python-decouple",
    "networkx": "networkx",
    "sympy": "sympy",
    "dask": "dask",
    "joblib": "joblib",
    "psutil": "psutil",
    "watchdog": "watchdog",
    "apscheduler": "apscheduler",
    "tenacity": "tenacity",
    "retry": "retry",
    "wrapt": "wrapt",
    "decorator": "decorator",
    "more_itertools": "more-itertools",
    "toolz": "toolz",
    "cytoolz": "cytoolz",
    "sortedcontainers": "sortedcontainers",
    "bitarray": "bitarray",
    "cachetools": "cachetools",
    "diskcache": "diskcache",
    "tinydb": "tinydb",
    "peewee": "peewee",
    "alembic": "alembic",
    "pytest_mock": "pytest-mock",
    "unittest2": "unittest2",
    "nose": "nose",
    "numpyro": "numpyro",
    "jax": "jax",
    "funsor": "funsor",
}


# =============================================================================
# Detection Functions
# =============================================================================


def has_pytest(content: str) -> bool:
    """Check if content has pytest tests."""
    return ('import pytest' in content or 'from pytest' in content) and 'def test_' in content


def has_unittest(content: str) -> bool:
    """Check if content has unittest tests."""
    return 'import unittest' in content and 'class' in content and 'def test_' in content


def has_doctest(content: str) -> bool:
    """Check if content has doctest examples."""
    return '>>>' in content and 'doctest' in content.lower()


def count_docstring_lines(content: str) -> int:
    """Count lines in docstrings (triple-quoted strings at start of functions/classes)."""
    # Find all docstrings
    docstrings = re.findall(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', content)
    total_lines = 0
    for ds in docstrings:
        total_lines += ds.count('\n')
    return total_lines


def count_comment_lines(content: str) -> int:
    """Count lines that are comments."""
    lines = content.split('\n')
    return sum(1 for l in lines if l.strip().startswith('#') and len(l.strip()) > 5)


def has_task_keywords(content: str) -> bool:
    """Check if content has task-like keywords in comments/docstrings."""
    content_lower = content.lower()
    keywords = [
        'implement', 'create', 'write', 'build', 'design',
        'should return', 'should accept', 'must return', 'must accept',
        'given', 'when', 'then',  # BDD style
        'requirement', 'specification', 'example usage',
        'input:', 'output:', 'args:', 'returns:',
        'todo:', 'task:', 'exercise:', 'problem:',
    ]
    return any(kw in content_lower for kw in keywords)


def extract_docstring(content: str) -> str:
    """Extract the main module/class docstring."""
    # Try to find module-level docstring
    match = re.match(r'^[\s]*(?:"""([\s\S]*?)"""|\'\'\'([\s\S]*?)\'\'\')', content)
    if match:
        return match.group(1) or match.group(2) or ""
    return ""


def is_valid_python3(content: str) -> bool:
    """Check if content is valid Python 3 syntax."""
    try:
        ast.parse(content)
        return True
    except SyntaxError:
        return False


def has_relative_imports(content: str) -> bool:
    """Check if content uses relative imports (from . import X, from ..module import Y)."""
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                return True
    except SyntaxError:
        pass
    return False


def extract_third_party_imports(content: str) -> Set[str]:
    """
    Extract third-party (non-stdlib, non-project) import module names from test code.

    Returns a set of pip package names that should be installed.
    """
    pip_packages = set()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return pip_packages

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split('.')[0]
                if top_module in KNOWN_PIP_PACKAGES:
                    pip_packages.add(KNOWN_PIP_PACKAGES[top_module])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                top_module = node.module.split('.')[0]
                if top_module in KNOWN_PIP_PACKAGES:
                    pip_packages.add(KNOWN_PIP_PACKAGES[top_module])

    return pip_packages


def is_self_documented_test(content: str) -> Tuple[bool, Dict]:
    """
    Check if a file is a self-documented test file.

    Returns (is_valid, metadata) where metadata contains detection info.
    """
    # Must have tests
    has_tests = has_pytest(content) or has_unittest(content) or has_doctest(content)
    if not has_tests:
        return False, {}

    # Must be valid Python 3 syntax
    if not is_valid_python3(content):
        return False, {}

    # Must not have relative imports (they break when running from /tests/)
    if has_relative_imports(content):
        return False, {}

    # Count documentation
    docstring_lines = count_docstring_lines(content)
    comment_lines = count_comment_lines(content)
    total_doc_lines = docstring_lines + comment_lines

    # Must have substantial documentation (at least 10 lines)
    if total_doc_lines < 10:
        return False, {}

    # Should have task-like keywords
    if not has_task_keywords(content):
        return False, {}

    # Check ratio of documentation to code
    total_lines = content.count('\n')
    doc_ratio = total_doc_lines / max(total_lines, 1)

    # At least 15% should be documentation
    if doc_ratio < 0.15:
        return False, {}

    metadata = {
        'has_pytest': has_pytest(content),
        'has_unittest': has_unittest(content),
        'has_doctest': has_doctest(content),
        'docstring_lines': docstring_lines,
        'comment_lines': comment_lines,
        'doc_ratio': doc_ratio,
        'main_docstring': extract_docstring(content)[:500],
        'third_party_packages': sorted(extract_third_party_imports(content)),
    }

    return True, metadata


def filter_self_documented_from_stack(limit: int, max_scan: int = 5_000_000) -> List[Dict]:
    """Filter The Stack for self-documented test files."""
    print("Loading The Stack (Python subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/python',
        split='train',
        streaming=True
    )

    samples = []
    scanned = 0
    skipped_syntax = 0
    skipped_relative = 0

    print(f"Scanning for self-documented test files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding self-documented tests")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')

        # Check content directly (path may be empty in The Stack)
        is_valid, metadata = is_self_documented_test(content)
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
    print(f"Scanned {scanned:,} files, found {len(samples)} self-documented test files")
    return samples


# =============================================================================
# Task Generation
# =============================================================================


def create_test_sh(third_party_packages: Optional[List[str]] = None) -> str:
    """Create test.sh that runs pytest with venv, PYTHONPATH, and dependency installation.

    Args:
        third_party_packages: List of pip package names to install before running tests.
            These are known third-party packages detected from the test file's imports.
    """
    # Build the pip install line for detected third-party packages
    if third_party_packages:
        # Deduplicate and sort for deterministic output
        pkgs = sorted(set(third_party_packages))
        install_block = f'''
# Install detected third-party test dependencies
echo "Installing test dependencies..."
pip install --quiet {" ".join(pkgs)} 2>&1 | tail -1 || true
'''
    else:
        install_block = ""

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
{install_block}
cd /app
export PYTHONPATH="/app:${{PYTHONPATH:-}}"

echo "Running tests..."
pytest /tests/test_solution.py -v --tb=short 2>&1 | tee /logs/verifier/test_output.txt

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

    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(create_standard_dockerfile(), encoding="utf-8")

    # Extract third-party packages from the test code
    third_party_pkgs = sorted(extract_third_party_imports(test_code))

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(third_party_pkgs), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / "test_solution.py").write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    if metadata:
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_tasks(samples: List[Dict], instructions: List[str], dataset_prefix: str = "stack-selfdoc") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(zip(samples, instructions), total=len(samples), desc="Creating tasks")):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "doc_ratio": sample.get("doc_ratio", 0),
            "docstring_lines": sample.get("docstring_lines", 0),
            "has_pytest": sample.get("has_pytest", False),
            "has_unittest": sample.get("has_unittest", False),
            "third_party_packages": sample.get("third_party_packages", []),
        }
        create_harbor_task_directory(temp_dir, i, instruction, sample["text"], dataset_prefix, metadata)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main() -> None:
    """Main pipeline."""
    parser = argparse.ArgumentParser(description="Generate self-documented tasks from The Stack")
    parser.add_argument("--limit", type=int, default=LIMIT, help=f"Number of tasks to generate (default: {LIMIT})")
    args = parser.parse_args()
    limit = args.limit

    print(f"Step 1: Filtering self-documented test files from The Stack (limit={limit})...")
    samples = filter_self_documented_from_stack(limit)
    print(f"  -> {len(samples)} self-documented test files found")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Extracting/cleaning task descriptions...")
    dataset = Dataset.from_list(samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": EXTRACT_INSTRUCTION_PROMPT, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    instructions = result.dataset["task_description"]
    print(f"  -> Extracted {len(instructions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, "stack-selfdoc")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-selfdoc")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} self-documented tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
