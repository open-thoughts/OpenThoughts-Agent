#!/usr/bin/env python3
"""
Generate tasks from BugsInPy dataset (1,215 real-world Python bugs across 17 projects).

BugsInPy provides:
- Real bugs from popular Python projects (pandas, keras, scrapy, fastapi, etc.)
- Buggy and fixed code versions (as patches/diffs)
- Test commands that expose the bugs

Data sources (tried in order):
1. Local clone of GitHub soarsmu/BugsInPy (fastest - cloned once, reused)
2. GitHub raw CDN (fallback if clone fails)
3. HuggingFace xin1997/bugsinpy_all_only_input (last resort - 501 rows, different schema)

Task structure:
- Dockerfile: python:3.10-slim + pytest (standard)
- test.sh: runs pytest /tests/test_solution.py (standard create_pytest_test_sh)
- instruction.md: describes the bug, provides buggy code, asks agent to fix in /app/solution.py
- tests/test_solution.py: LLM-generated self-contained pytest tests (from solution import *)
"""

import itertools
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import requests
from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import (
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

# GitHub URLs for BugsInPy
BUGSINPY_GITHUB_REPO = "https://github.com/soarsmu/BugsInPy.git"
BUGSINPY_GITHUB_API = "https://api.github.com/repos/soarsmu/BugsInPy/contents"
BUGSINPY_RAW_BASE = "https://raw.githubusercontent.com/soarsmu/BugsInPy/master"

# Local cache directory for the cloned repo
BUGSINPY_CACHE_DIR = Path(tempfile.gettempdir()) / "bugsinpy_repo_cache"

# All 17 BugsInPy projects
BUGSINPY_PROJECTS = [
    "PySnooper", "ansible", "black", "cookiecutter", "fastapi",
    "httpie", "keras", "luigi", "matplotlib", "pandas",
    "sanic", "scrapy", "spacy", "thefuck", "tornado",
    "tqdm", "youtube-dl",
]

# =============================================================================
# LLM Prompts
# =============================================================================

GENERATE_TEST_PROMPT = """You are an expert Python test engineer. Given a bug fix (shown as a diff/patch) and the ACTUAL buggy solution.py file, generate a self-contained pytest test file.

Project: {{project}}
Bug ID: {{bug_id}}

Bug patch (diff showing buggy -> fixed):
```
{{bug_patch}}
```

ACTUAL solution.py file (this is what `from solution import *` will import):
```python
{{buggy_code}}
```

Test command from original project:
```
{{test_command}}
```

Requirements for the test file:
1. Import from solution: `from solution import *`
2. CRITICAL: You can ONLY import names that are defined in the solution.py shown above. Check the __all__ list or the top-level definitions. Do NOT import names that don't exist in solution.py.
3. The test must be self-contained (no external dependencies beyond pytest and standard library)
4. Write 2-5 test functions that would FAIL with the buggy code and PASS with the fixed code
5. Focus on testing the specific behavior that the patch fixes
6. Use descriptive test function names like `test_<description>()`
7. Include assertions with clear error messages
8. Do NOT import from the original project (pandas, keras, etc.) - only from `solution`
9. If the solution.py uses `import inspect` or similar stdlib modules internally, do NOT assume those are exported - import them yourself if needed in tests

Generate ONLY the Python test file content, nothing else:"""

GENERATE_BUGGY_CODE_PROMPT = """You are an expert Python developer. Given a bug patch (diff from buggy→fixed), generate the BUGGY version of the code as a complete, self-contained Python file.

Project: {{project}}
Bug ID: {{bug_id}}

Bug patch (diff showing buggy -> fixed):
```
{{bug_patch}}
```

Requirements:
1. Generate the BUGGY version of the code (BEFORE the fix was applied)
2. The file must be self-contained and importable as a Python module
3. Include ALL functions and classes that the patch touches - implement them fully
4. CRITICAL: Do NOT import from {{project}} or any third-party package. Only use Python standard library imports (os, sys, re, json, pathlib, collections, etc.). If the original code imports from {{project}}, define those classes/functions locally instead. For example, if the code does `from ansible.errors import AnsibleError`, define `class AnsibleError(Exception): pass` locally.
5. The code should be the BUGGY version that has the bug described in the patch
6. If the patch touches multiple files, combine the relevant functions/classes into one file
7. Do NOT include the fix - this is the BROKEN version the agent needs to fix
8. Export all public names so `from solution import *` works
9. Define __all__ to list all exported names

Generate ONLY the Python file content, nothing else:"""

GENERATE_INSTRUCTION_PROMPT = """You are an expert at creating bug-fixing coding tasks for AI agents.

Given the following information about a real bug in a Python project, create a clear task description.

Project: {{project}}
Bug ID: {{bug_id}}

Bug patch (what was changed to fix the bug):
```
{{bug_patch}}
```

The buggy code is already provided in `/app/solution.py`. The agent must FIX the bug in that file.

Create a task description that:
1. Describes what the code is supposed to do
2. Explains what behavior is currently broken (without giving the exact fix)
3. Tells the agent that the buggy code is in `/app/solution.py` and they need to fix it
4. Mentions that tests will verify the fix via `from solution import *`

IMPORTANT: Do NOT include the full source code in the task description (it's already in solution.py).
Do NOT reveal the exact fix. Give enough context for the agent to understand what's wrong.

Task description:"""


# =============================================================================
# Local Repo Clone Loading (fastest approach)
# =============================================================================

def _clone_or_update_repo() -> Optional[Path]:
    """Clone or update the BugsInPy repo to a local cache directory."""
    repo_dir = BUGSINPY_CACHE_DIR
    if repo_dir.exists() and (repo_dir / ".git").exists():
        # Repo already cloned, do a pull to update
        print(f"  Using cached repo at {repo_dir}")
        try:
            subprocess.run(
                ["git", "pull", "--quiet"],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass  # Stale cache is fine
        return repo_dir

    # Clone fresh
    print(f"  Cloning BugsInPy repo to {repo_dir}...")
    try:
        repo_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", BUGSINPY_GITHUB_REPO, str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"  Clone successful")
            return repo_dir
        else:
            print(f"  Clone failed: {result.stderr[:200]}")
            return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
        print(f"  Clone failed: {e}")
        return None


def _read_file_safe(path: Path) -> Optional[str]:
    """Read a file, returning None if it doesn't exist or can't be read."""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError):
        pass
    return None


def _parse_bug_info(content: str) -> Dict[str, str]:
    """Parse a bug.info file into a dict.

    Format is key="value" pairs, one per line.
    Example:
        python_version="3.6.9"
        buggy_commit_id="8cc777fe..."
        fixed_commit_id="c0dcf39b..."
        test_file="tqdm/tests/tests_contrib.py"
    """
    info = {}
    if not content:
        return info
    for line in content.strip().split("\n"):
        line = line.strip()
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip().strip('"')
            value = value.strip().strip('"')
            info[key] = value
    return info


def load_bugsinpy_from_local(limit: int = LIMIT) -> List[Dict]:
    """
    Load BugsInPy by cloning the GitHub repo locally and parsing files.

    This is the fastest approach since it avoids per-file HTTP requests.
    The repo is cloned with --depth 1 (~5MB) and cached for reuse.

    BugsInPy repo structure:
        projects/<project>/bugs/<bug_id>/
            bug.info         - commit IDs, test file, python version
            bug_patch.txt    - diff between buggy and fixed code
            run_test.sh      - shell command to run the test
            requirements.txt - dependencies
            setup.sh         - setup commands
    """
    print("Loading BugsInPy from local clone...")
    repo_dir = _clone_or_update_repo()
    if repo_dir is None:
        return []

    projects_dir = repo_dir / "projects"
    if not projects_dir.exists():
        print(f"  Error: {projects_dir} does not exist")
        return []

    samples = []

    for project in tqdm(BUGSINPY_PROJECTS, desc="Loading projects"):
        bugs_dir = projects_dir / project / "bugs"
        if not bugs_dir.exists():
            continue

        # Get list of bug directories (numbered 1, 2, 3, ...)
        bug_dirs = sorted(
            [d for d in bugs_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )

        for bug_dir in bug_dirs:
            if len(samples) >= limit:
                break

            bug_id = bug_dir.name

            # Read bug.info
            bug_info_text = _read_file_safe(bug_dir / "bug.info")
            if bug_info_text is None:
                continue
            bug_info = _parse_bug_info(bug_info_text)

            # Read bug patch (the key data)
            bug_patch = _read_file_safe(bug_dir / "bug_patch.txt")
            if not bug_patch:
                continue

            # Read test command
            run_test = _read_file_safe(bug_dir / "run_test.sh")
            test_command = run_test.strip() if run_test else ""

            # Read requirements
            requirements = _read_file_safe(bug_dir / "requirements.txt")

            samples.append({
                "project": project,
                "bug_id": bug_id,
                "bug_patch": bug_patch,
                "test_command": test_command,
                "buggy_commit": bug_info.get("buggy_commit_id", ""),
                "fixed_commit": bug_info.get("fixed_commit_id", ""),
                "test_file": bug_info.get("test_file", ""),
                "python_version": bug_info.get("python_version", "3.10"),
                "requirements": requirements or "",
            })

        if len(samples) >= limit:
            break

    if samples:
        projects = set(s["project"] for s in samples)
        print(f"Loaded {len(samples)} bugs from {len(projects)} projects: "
              f"{', '.join(sorted(projects))}")
    return samples


# =============================================================================
# GitHub Raw CDN Loading (fallback if clone fails)
# =============================================================================

def _github_get(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    """Make a GitHub API/raw request with retries and rate limit handling."""
    for attempt in range(max_retries):
        try:
            headers = {}
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 403:
                # Rate limited - wait and retry
                wait_time = min(60, 2 ** attempt * 10)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif resp.status_code == 404:
                return None
            else:
                return None
        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def _fetch_raw_file(path: str) -> Optional[str]:
    """Fetch a raw file from the BugsInPy GitHub repo via CDN."""
    url = f"{BUGSINPY_RAW_BASE}/{path}"
    resp = _github_get(url)
    if resp and resp.status_code == 200:
        return resp.text
    return None


def load_bugsinpy_from_github_cdn(limit: int = LIMIT) -> List[Dict]:
    """
    Load BugsInPy from GitHub raw CDN (slower, per-file HTTP requests).

    Used as fallback when git clone is not available.
    Note: This makes ~3-4K HTTP requests and takes 10-30 minutes.
    """
    print("Loading BugsInPy from GitHub CDN (this may take a while)...")
    samples = []

    for project in tqdm(BUGSINPY_PROJECTS, desc="Loading projects"):
        bugs_found = 0
        consecutive_misses = 0

        # Try sequential bug IDs (BugsInPy uses 1-indexed numbering)
        for bug_id in range(1, 200):  # No project has > 170 bugs
            if len(samples) >= limit:
                break

            base_path = f"projects/{quote(project)}/bugs/{bug_id}"

            # Fetch bug.info (existence check)
            bug_info_text = _fetch_raw_file(f"{base_path}/bug.info")
            if bug_info_text is None:
                consecutive_misses += 1
                if consecutive_misses >= 3 and bugs_found > 0:
                    break  # No more bugs for this project
                continue

            consecutive_misses = 0
            bugs_found += 1
            bug_info = _parse_bug_info(bug_info_text)

            # Fetch bug patch
            bug_patch = _fetch_raw_file(f"{base_path}/bug_patch.txt")
            if not bug_patch:
                continue

            # Fetch test command
            run_test = _fetch_raw_file(f"{base_path}/run_test.sh")
            test_command = run_test.strip() if run_test else ""

            # Fetch requirements
            requirements = _fetch_raw_file(f"{base_path}/requirements.txt")

            samples.append({
                "project": project,
                "bug_id": str(bug_id),
                "bug_patch": bug_patch,
                "test_command": test_command,
                "buggy_commit": bug_info.get("buggy_commit_id", ""),
                "fixed_commit": bug_info.get("fixed_commit_id", ""),
                "test_file": bug_info.get("test_file", ""),
                "python_version": bug_info.get("python_version", "3.10"),
                "requirements": requirements or "",
            })

        if len(samples) >= limit:
            break

    if samples:
        projects = set(s["project"] for s in samples)
        print(f"Loaded {len(samples)} bugs from {len(projects)} projects: "
              f"{', '.join(sorted(projects))}")
    return samples


# =============================================================================
# HuggingFace Fallback Loader
# =============================================================================

def _parse_xin1997_sample(sample: Dict) -> Optional[Dict]:
    """
    Parse a sample from xin1997/bugsinpy_all_only_input.

    Schema: id (e.g. "sanic-bug-5"), content (concatenated buggy+fixed code),
            max_stars_repo_path (comma-separated paths showing buggy and fixed files)
    """
    sample_id = sample.get("id", "")
    content = sample.get("content", "")
    repo_path = sample.get("max_stars_repo_path", "")

    if not sample_id or not content:
        return None

    # Parse project and bug_id from id field (e.g., "sanic-bug-5")
    match = re.match(r"^(\w[\w-]*?)-bug-(\d+)$", sample_id)
    if not match:
        return None

    project = match.group(1)
    bug_id = match.group(2)

    # The content field contains both buggy and fixed versions concatenated.
    # Create a pseudo-patch from it.
    bug_patch = (
        f"# Content from {sample_id}\n"
        f"# File paths: {repo_path}\n\n"
        f"{content[:5000]}"
    )

    return {
        "project": project,
        "bug_id": bug_id,
        "bug_patch": bug_patch,
        "test_command": "",
        "buggy_commit": "",
        "fixed_commit": "",
        "test_file": "",
        "python_version": "3.10",
        "requirements": "",
    }


def load_bugsinpy_from_hf(limit: int = LIMIT) -> List[Dict]:
    """
    Load BugsInPy from HuggingFace (last resort fallback).

    Tries multiple HF repo names and adapts to their different schemas.
    """
    print("Loading BugsInPy from HuggingFace (fallback)...")

    repo_names = [
        "Harryxun/BugsInPy_data",
        "xin1997/bugsinpy_all_only_input",
    ]

    for repo_name in repo_names:
        try:
            ds = load_dataset(repo_name, split="train", streaming=True)
            print(f"  Loaded from {repo_name}")

            samples = []
            for raw_sample in tqdm(
                itertools.islice(ds, limit), total=limit, desc="Loading"
            ):
                # Try Harryxun schema (project, bug_id, buggy, fixed, test_command)
                if "project" in raw_sample and "buggy" in raw_sample:
                    project = raw_sample.get("project", "")
                    buggy = raw_sample.get("buggy", "")
                    fixed = raw_sample.get("fixed", "")
                    if not project or not buggy:
                        continue
                    # Create a pseudo-patch from buggy vs fixed
                    bug_patch = "--- buggy\n+++ fixed\n"
                    for bline, fline in itertools.zip_longest(
                        buggy.splitlines(), fixed.splitlines(), fillvalue=""
                    ):
                        if bline != fline:
                            bug_patch += f"-{bline}\n+{fline}\n"
                        else:
                            bug_patch += f" {bline}\n"

                    samples.append({
                        "project": project,
                        "bug_id": str(raw_sample.get("bug_id", len(samples))),
                        "bug_patch": bug_patch,
                        "test_command": raw_sample.get("test_command", ""),
                        "buggy_commit": "",
                        "fixed_commit": "",
                        "test_file": "",
                        "python_version": "3.10",
                        "requirements": "",
                    })
                # Try xin1997 schema (id, content, max_stars_repo_path)
                elif "id" in raw_sample and "content" in raw_sample:
                    parsed = _parse_xin1997_sample(raw_sample)
                    if parsed:
                        samples.append(parsed)

            if samples:
                projects = set(s["project"] for s in samples)
                print(f"  Loaded {len(samples)} bugs from {len(projects)} projects")
                return samples

        except Exception as e:
            print(f"  Could not load from {repo_name}: {e}")
            continue

    return []


# =============================================================================
# Main Loader
# =============================================================================

def load_bugsinpy(limit: int = LIMIT, source: str = "auto") -> List[Dict]:
    """
    Load BugsInPy dataset, trying sources in order of speed/reliability.

    Args:
        limit: Maximum number of samples to load.
        source: Data source - "local" (git clone), "cdn" (GitHub raw),
                "hf" (HuggingFace), or "auto" (try all in order).
    """
    loaders = {
        "local": ("Local clone", load_bugsinpy_from_local),
        "cdn": ("GitHub CDN", load_bugsinpy_from_github_cdn),
        "hf": ("HuggingFace", load_bugsinpy_from_hf),
    }

    if source != "auto":
        name, loader = loaders[source]
        samples = loader(limit=limit)
        if samples:
            return samples
        raise ValueError(f"Could not load BugsInPy from {name}.")

    # Auto mode: try all sources in order
    for key in ["local", "cdn", "hf"]:
        name, loader = loaders[key]
        try:
            samples = loader(limit=limit)
            if samples:
                return samples
        except Exception as e:
            print(f"{name} loading failed: {e}")

    raise ValueError(
        "Could not load BugsInPy dataset from any source.\n"
        "Tried: local git clone, GitHub CDN, HuggingFace.\n"
        "Ensure you have internet access or set GITHUB_TOKEN for higher rate limits."
    )


# =============================================================================
# Task Generation
# =============================================================================

def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    test_code: str,
    buggy_code: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a BugsInPy bug."""
    # Python Dockerfile with buggy code COPY'd into /app
    base_dockerfile = get_dockerfile("python")
    # Add COPY to put buggy solution.py into /app (must be after WORKDIR /app)
    if buggy_code:
        dockerfile = base_dockerfile.rstrip() + "\n\n# Copy buggy starter code into /app\nCOPY solution.py /app/solution.py\n"
    else:
        dockerfile = base_dockerfile

    # Standard pytest test.sh (runs /tests/test_solution.py, with solution.py fallback)
    test_sh = create_pytest_test_sh("/tests/test_solution.py")

    # Test file: LLM-generated self-contained pytest tests
    test_files = {
        "test_solution.py": test_code,
    }

    metadata = {
        "source": "bugsinpy",
        "project": sample.get("project", ""),
        "bug_id": sample.get("bug_id", ""),
        "test_command": sample.get("test_command", ""),
        "buggy_commit": sample.get("buggy_commit", ""),
        "fixed_commit": sample.get("fixed_commit", ""),
    }

    task_dir = create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=test_sh,
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
    )

    # Write buggy code into environment/ dir so Dockerfile can COPY it
    if buggy_code:
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "solution.py").write_text(buggy_code)

    return task_dir


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    test_codes: List[str],
    buggy_codes: List[str],
    dataset_prefix: str = "bugsinpy",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction, test_code, buggy_code) in enumerate(tqdm(
        zip(samples, instructions, test_codes, buggy_codes),
        total=len(samples),
        desc="Creating tasks",
    )):
        create_harbor_task(temp_dir, i, instruction, test_code, buggy_code, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def _clean_llm_code_output(text: str) -> str:
    """Extract Python code from LLM output that might include markdown fences."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _validate_buggy_code(code: str, project: str) -> str:
    """Validate and fix buggy code - remove project-internal imports that would fail."""
    if not code:
        return code

    # Standard library modules that are safe to import
    STDLIB_MODULES = {
        "abc", "argparse", "ast", "asyncio", "base64", "bisect", "calendar",
        "collections", "contextlib", "copy", "csv", "dataclasses", "datetime",
        "decimal", "difflib", "enum", "errno", "fnmatch", "fractions",
        "functools", "glob", "hashlib", "heapq", "hmac", "html", "http",
        "importlib", "inspect", "io", "itertools", "json", "logging", "math",
        "multiprocessing", "operator", "os", "pathlib", "pickle", "pprint",
        "queue", "random", "re", "shlex", "shutil", "signal", "socket",
        "sqlite3", "string", "struct", "subprocess", "sys", "tempfile",
        "textwrap", "threading", "time", "traceback", "types", "typing",
        "unicodedata", "unittest", "urllib", "uuid", "warnings", "weakref",
        "xml", "zipfile",
    }

    project_lower = project.lower()
    lines = code.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        # Check for project-internal imports
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Extract module name
            if stripped.startswith("from "):
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[1].split(".")[0]
                else:
                    module = ""
            else:
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[1].split(".")[0]
                else:
                    module = ""

            # Skip imports from the project itself or known third-party packages
            if module.lower() == project_lower or module in BUGSINPY_PROJECTS_LOWER:
                # Comment out the bad import
                cleaned_lines.append(f"# REMOVED: {stripped}  # project-internal import")
                continue

            # Allow stdlib
            if module in STDLIB_MODULES or module.startswith("_"):
                cleaned_lines.append(line)
                continue

            # Allow pytest for test files
            if module == "pytest":
                cleaned_lines.append(line)
                continue

            # Unknown module - keep it but warn (might be stdlib we missed)
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# Lowercase project names for import validation
BUGSINPY_PROJECTS_LOWER = {p.lower() for p in BUGSINPY_PROJECTS}


def main(limit: int = LIMIT, upload: bool = True, source: str = "auto") -> Optional[str]:
    """Main pipeline for generating BugsInPy tasks."""

    print("=" * 60)
    print("BugsInPy Task Generator")
    print("=" * 60)

    # Step 1: Load data
    print("\nStep 1: Loading BugsInPy dataset...")
    samples = load_bugsinpy(limit=limit, source=source)
    print(f"  -> {len(samples)} bugs loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return None

    # Step 2: Generate buggy code files with LLM
    print("\nStep 2: Generating buggy code files with LLM...")
    dataset = Dataset.from_list(samples)

    buggy_result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": GENERATE_BUGGY_CODE_PROMPT,
            "output_column": "buggy_code",
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    buggy_ds = buggy_result.dataset if hasattr(buggy_result, "dataset") else buggy_result
    buggy_codes = [_clean_llm_code_output(t) for t in buggy_ds["buggy_code"]]
    # Validate buggy codes - strip project-internal imports
    buggy_codes = [
        _validate_buggy_code(code, sample["project"])
        for code, sample in zip(buggy_codes, samples)
    ]
    print(f"  -> Generated {len(buggy_codes)} buggy code files")

    # Step 3: Generate self-contained pytest tests with LLM
    # IMPORTANT: Include buggy_code in the prompt so tests only import what actually exists
    print("\nStep 3: Generating pytest test files with LLM...")
    # Create a fresh dataset with buggy_code included for test generation
    test_samples = [{**s, "_step": "test", "buggy_code": bc} for s, bc in zip(samples, buggy_codes)]
    test_dataset = Dataset.from_list(test_samples)

    test_result = run_completions(
        test_dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": GENERATE_TEST_PROMPT,
            "output_column": "generated_test",
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    test_ds = test_result.dataset if hasattr(test_result, "dataset") else test_result
    raw_tests = test_ds["generated_test"]
    test_codes = [_clean_llm_code_output(t) for t in raw_tests]
    print(f"  -> Generated {len(test_codes)} test files")

    # Step 4: Generate task instructions with LLM
    print("\nStep 4: Generating task instructions with LLM...")
    # Create a fresh dataset to avoid curator cache collision with steps 2 and 3
    instr_samples = [{**s, "_step": "instruction"} for s in samples]
    instr_dataset = Dataset.from_list(instr_samples)
    instr_result = run_completions(
        instr_dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": GENERATE_INSTRUCTION_PROMPT,
            "output_column": "task_description",
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instr_ds = instr_result.dataset if hasattr(instr_result, "dataset") else instr_result
    instructions = instr_ds["task_description"]
    print(f"  -> Generated {len(instructions)} task descriptions")

    # Step 5: Create harbor tasks
    print("\nStep 5: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, test_codes, buggy_codes, "bugsinpy")
    print(f"  -> Task directory: {task_dir}")

    # Step 5: Upload
    if upload:
        print("\nStep 5: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_bugsinpy")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'=' * 60}")
    print(f"Successfully generated {len(samples)} BugsInPy tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'=' * 60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from BugsInPy dataset")
    parser.add_argument("--limit", type=int, default=LIMIT,
                        help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HuggingFace upload")
    parser.add_argument("--source", choices=["local", "cdn", "hf", "auto"],
                        default="auto",
                        help="Data source: local (git clone), cdn (GitHub raw), "
                             "hf (HuggingFace), or auto (try all)")

    args = parser.parse_args()
    main(limit=args.limit, upload=not args.no_upload, source=args.source)
