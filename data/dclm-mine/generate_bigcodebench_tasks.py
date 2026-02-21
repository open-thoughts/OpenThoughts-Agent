#!/usr/bin/env python3
"""
Generate tasks from BigCodeBench dataset (1,140 multi-library tasks).

BigCodeBench tests practical coding with 139 different libraries across
7 domains (data analysis, ML, web dev, etc.). Average 5.6 test cases per
task with 99% branch coverage. Best models (GPT-4o) only solve 60%.

Dataset structure:
- task_id: Unique identifier (e.g., "BigCodeBench/1")
- complete_prompt: Function signature + docstring
- instruct_prompt: Natural language instruction
- canonical_solution: Reference implementation
- test: Test cases
- libs: Required libraries for the task
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

from data.commons import (
    create_harbor_task_directory_generic,
    create_generic_test_sh,
    create_pytest_test_sh,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"

# Base Dockerfile - will be customized per task with required libs
BIGCODEBENCH_DOCKERFILE_TEMPLATE = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for common libraries
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    libffi-dev \\
    libxml2-dev \\
    libxslt1-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install base Python packages
RUN pip install --no-cache-dir \\
    pytest \\
    pytest-timeout \\
    numpy \\
    pandas \\
    scipy \\
    matplotlib \\
    seaborn \\
    scikit-learn \\
    requests \\
    beautifulsoup4 \\
    lxml \\
    pillow \\
    opencv-python-headless

{additional_installs}

RUN mkdir -p /logs/verifier
"""


def get_dockerfile_for_libs(libs: List[str]) -> str:
    """Generate Dockerfile with required libraries installed."""
    # Common library mappings (some need special handling)
    lib_install_map = {
        "torch": "torch --index-url https://download.pytorch.org/whl/cpu",
        "tensorflow": "tensorflow-cpu",
        "cv2": "opencv-python-headless",
        "PIL": "pillow",
        "bs4": "beautifulsoup4",
        "sklearn": "scikit-learn",
    }

    # Python standard library modules - never pip install these
    stdlib_modules = {
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
        "sqlite3", "ssl", "stat", "statistics", "string", "stringprep",
        "struct", "subprocess", "sunau", "symtable", "sys", "sysconfig",
        "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile", "termios",
        "test", "textwrap", "threading", "time", "timeit", "tkinter",
        "token", "tokenize", "trace", "traceback", "tracemalloc", "tty",
        "turtle", "turtledemo", "types", "typing", "unicodedata", "unittest",
        "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
        "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml",
        "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
    }

    # Filter out already installed base packages
    base_packages = {
        "numpy", "pandas", "scipy", "matplotlib", "seaborn",
        "scikit-learn", "sklearn", "requests", "beautifulsoup4",
        "bs4", "lxml", "pillow", "PIL", "cv2", "opencv-python-headless"
    }

    additional_libs = []
    for lib in libs:
        lib_lower = lib.lower().strip()
        # Skip stdlib and base packages
        if lib_lower in stdlib_modules or lib_lower in base_packages:
            continue
        install_name = lib_install_map.get(lib, lib)
        additional_libs.append(install_name)

    if additional_libs:
        additional_installs = "RUN pip install --no-cache-dir \\\n    " + " \\\n    ".join(additional_libs)
    else:
        additional_installs = "# No additional libraries required"

    return BIGCODEBENCH_DOCKERFILE_TEMPLATE.format(additional_installs=additional_installs)


def create_bigcodebench_test_sh() -> str:
    """Create test.sh for BigCodeBench tasks."""
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

# Install any missing dependencies quietly
pip install --quiet pytest pytest-timeout 2>/dev/null || true

echo "Running BigCodeBench tests..."
pytest /tests/test_solution.py -v --tb=short --timeout=60 2>&1 | tee /logs/verifier/test_output.txt

exit ${PIPESTATUS[0]}
'''


def load_bigcodebench(split: str = "v0.1.2", limit: int = LIMIT) -> List[Dict]:
    """
    Load BigCodeBench dataset from HuggingFace.

    The dataset has versions like v0.1.0, v0.1.1, v0.1.2
    """
    print(f"Loading BigCodeBench dataset ({split})...")

    # BigCodeBench uses version as split name (not "train")
    # Available splits: v0.1.0_hf, v0.1.1, v0.1.2, v0.1.3, v0.1.4
    versions_to_try = ["v0.1.4", "v0.1.3", "v0.1.2", "v0.1.1", "v0.1.0_hf"]

    ds = None
    for version in versions_to_try:
        try:
            ds = load_dataset("bigcode/bigcodebench", split=version, streaming=True)
            print(f"Loaded from bigcode/bigcodebench (split={version})")
            break
        except Exception as e:
            print(f"Could not load bigcode/bigcodebench split={version}: {e}")
            continue

    if ds is None:
        raise ValueError(f"Could not load BigCodeBench dataset from any version")

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        task_id = sample.get("task_id", "")
        instruct_prompt = sample.get("instruct_prompt", "")
        complete_prompt = sample.get("complete_prompt", "")
        test = sample.get("test", "")

        if not task_id or (not instruct_prompt and not complete_prompt):
            continue

        # Parse libraries from libs field or extract from imports
        libs = sample.get("libs", [])
        if isinstance(libs, str):
            # libs field can be a string like "['random', 'itertools']"
            import ast
            try:
                libs = ast.literal_eval(libs)
            except (ValueError, SyntaxError):
                libs = [l.strip().strip("'\"") for l in libs.strip("[]").split(",") if l.strip()]
        if isinstance(libs, list):
            libs = [l.strip().strip("'\"") for l in libs if isinstance(l, str)]

        samples.append({
            "task_id": task_id,
            "instruct_prompt": instruct_prompt,
            "complete_prompt": complete_prompt,
            "canonical_solution": sample.get("canonical_solution", ""),
            "test": test,
            "libs": libs,
            "entry_point": sample.get("entry_point", "solution"),
        })

    print(f"Loaded {len(samples)} BigCodeBench tasks")
    return samples


def create_instruction_from_sample(sample: Dict) -> str:
    """Create task instruction from BigCodeBench sample."""
    instruct = sample.get("instruct_prompt", "")
    complete = sample.get("complete_prompt", "")
    libs = sample.get("libs", [])

    # Prefer instruct_prompt if available
    if instruct:
        instruction = f"""# Task

{instruct}

"""
    elif complete:
        instruction = f"""# Task

Implement the following function:

```python
{complete}
```

"""
    else:
        instruction = "# Task\n\nImplement the required functionality.\n"

    if libs:
        instruction += f"""## Required Libraries

This task requires the following libraries: {', '.join(libs)}

"""

    instruction += """## Instructions

1. Place your solution in `/app/solution.py`
2. Export the required function(s)
3. Your solution should pass all test cases
"""

    return instruction


def create_test_file_content(sample: Dict) -> str:
    """Create test file content from BigCodeBench sample."""
    test_code = sample.get("test", "")
    entry_point = sample.get("entry_point", "solution")

    # Wrap test code with imports
    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

# Import the solution
try:
    from solution import *
except ImportError as e:
    print(f"Could not import solution: {{e}}")
    raise

{test_code}
"""
    return test_content


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a BigCodeBench sample."""
    instruction = create_instruction_from_sample(sample)
    test_content = create_test_file_content(sample)

    test_files = {
        "test_solution.py": test_content,
    }

    # Add requirements info
    libs = sample.get("libs", [])
    if libs:
        test_files["requirements.txt"] = "\n".join(libs)

    metadata = {
        "source": "bigcodebench",
        "task_id": sample.get("task_id", ""),
        "entry_point": sample.get("entry_point", ""),
        "libs": libs,
        "libs_count": len(libs),
    }

    # Store canonical solution
    solution_files = None
    if sample.get("canonical_solution"):
        solution_files = {"solution.py": sample["canonical_solution"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile_for_libs(libs),
        test_sh=create_bigcodebench_test_sh(),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(samples: List[Dict], dataset_prefix: str = "bigcodebench") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, sample in enumerate(tqdm(samples, desc="Creating tasks")):
        create_harbor_task(temp_dir, i, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(split: str = "v0.1.2", limit: int = LIMIT, upload: bool = True) -> None:
    """Main pipeline for generating BigCodeBench tasks."""

    print(f"Step 1: Loading BigCodeBench dataset ({split})...")
    samples = load_bigcodebench(split=split, limit=limit)
    print(f"  -> {len(samples)} tasks loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating harbor task directories...")
    dataset_prefix = "bigcodebench"
    task_dir = generate_tasks(samples, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 3: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} BigCodeBench tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from BigCodeBench dataset")
    parser.add_argument("--split", default="v0.1.2", help="Dataset version/split")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")

    args = parser.parse_args()
    main(split=args.split, limit=args.limit, upload=not args.no_upload)
