#!/usr/bin/env python3
"""
Generate MULTI-FAULT tasks from BugsInPy dataset (493 real Python bugs across 17 projects).

This is the multi-fault variant: it combines 2-3 bugs from the SAME project into one
task, making it more challenging than the single-fault version. The agent must find
and fix multiple independent bugs in a single file.

Architecture (v4 - self-contained):
- Each task has a single solution.py with buggy code from 2-3 combined bugs
- Tests are LLM-generated, self-contained pytest (from solution import *)
- Standard Python Dockerfile (no project pip install needed)
- Agent must fix ALL bugs in solution.py for the task to pass

Previous approach (v3 - broken):
- Installed project via pip and ran original test commands
- Failed because pip doesn't include test files from source repos
"""

import os
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from itertools import combinations as iter_combos
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset
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
BUGS_PER_TASK = (2, 3)  # Combine 2-3 bugs per multi-fault task
BUGSINPY_REPO_URL = "https://github.com/soarsmu/BugsInPy.git"
RANDOM_SEED = 42

# =============================================================================
# LLM Prompts
# =============================================================================

GENERATE_COMBINED_BUGGY_CODE_PROMPT = """You are an expert Python developer. Given patches for MULTIPLE bugs from the same project, generate a single BUGGY Python file that contains ALL the buggy code combined.

Project: {{project}}
Number of bugs: {{num_bugs}}

{{all_patches}}

Requirements:
1. Generate ONE self-contained Python file that includes ALL buggy functions/classes from ALL patches
2. Each function/class should contain its respective bug (the version BEFORE the fix)
3. The file must be importable as a Python module (`from solution import *`)
4. CRITICAL: Do NOT import from {{project}} or any third-party package. Only use Python standard library imports (os, sys, re, json, pathlib, collections, etc.). If the code originally imports from {{project}}, define those classes/functions locally. For example: `class AnsibleError(Exception): pass`
5. Include ALL functions/classes that the patches touch - implement them fully
6. If different patches touch different files, combine all relevant code into one file
7. Use an `__all__` list to export all public names
8. Do NOT include any fixes - every function should have its original bug
9. Make sure function signatures are complete and the code can actually run (even if buggy)

Generate ONLY the Python file content, nothing else:"""

GENERATE_COMBINED_TEST_PROMPT = """You are an expert Python test engineer. Given MULTIPLE bug patches and the ACTUAL buggy solution.py file, generate a self-contained pytest test file that tests ALL the bugs.

Project: {{project}}
Number of bugs: {{num_bugs}}

{{all_patches}}

ACTUAL solution.py file (this is what `from solution import *` will import):
```python
{{buggy_code}}
```

Requirements for the test file:
1. Import from solution: `from solution import *`
2. CRITICAL: You can ONLY import names that are defined in the solution.py shown above. Check the __all__ list or top-level definitions carefully. Do NOT import names that don't exist.
3. Write separate test functions for EACH bug (at least 2 tests per bug, clearly labeled)
4. Tests should FAIL with the buggy code and PASS with the fixed code
5. Use descriptive names: `test_bug1_<description>()`, `test_bug2_<description>()`, etc.
6. The test must be self-contained (no external deps beyond pytest and standard library)
7. Do NOT import from the original project - only from `solution`
8. If the solution.py uses stdlib modules internally (like `inspect`), import them yourself if needed in tests

Generate ONLY the Python test file content, nothing else:"""

GENERATE_INSTRUCTION_PROMPT = """You are an expert at creating challenging bug-fixing tasks for software engineering agents.

This is a MULTI-FAULT task: the agent must find and fix MULTIPLE bugs in the same file.

Project: {{project}}
Number of bugs: {{num_bugs}}

{{all_patches}}

The buggy code is already provided in `/app/solution.py`. The agent must FIX ALL bugs in that file.

Create a task description that:
1. Explains this is a multi-fault debugging challenge with {{num_bugs}} bugs to fix
2. For each bug, describes what functionality is broken WITHOUT revealing the exact fix
3. Tells the agent the buggy code is in `/app/solution.py` and ALL bugs must be fixed
4. Mentions that tests will verify all fixes via `from solution import *`
5. Emphasizes ALL bugs must be fixed for the task to pass

IMPORTANT: Do NOT include full source code. Do NOT reveal exact fixes.

Task description:"""

# =============================================================================
# Data Loading
# =============================================================================

def ensure_bugsinpy_repo(cache_dir: Path) -> Path:
    """Clone BugsInPy repo if not already cached."""
    repo_dir = cache_dir / "BugsInPy"
    if repo_dir.exists() and (repo_dir / "projects").exists():
        print(f"  Using cached BugsInPy repo at {repo_dir}")
        return repo_dir

    print(f"  Cloning BugsInPy repo to {repo_dir}...")
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", BUGSINPY_REPO_URL, str(repo_dir)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    return repo_dir


def parse_bug_info(bug_info_path: Path) -> Dict:
    """Parse a bug.info file into a dict."""
    info = {}
    if not bug_info_path.exists():
        return info
    text = bug_info_path.read_text(errors="ignore").strip()
    for line in text.split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            info[key.strip()] = value.strip().strip('"').strip("'")
    return info


def load_bugsinpy_from_repo(limit: int = LIMIT) -> List[Dict]:
    """Load BugsInPy bugs directly from the GitHub repository."""
    cache_dir = Path(os.environ.get("DATA_CACHE_DIR", "/scratch/10000/eguha3/data_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = ensure_bugsinpy_repo(cache_dir)
    projects_dir = repo_dir / "projects"

    if not projects_dir.exists():
        raise ValueError(f"BugsInPy projects directory not found at {projects_dir}")

    samples = []
    project_names = sorted(
        d.name for d in projects_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    print(f"  Found {len(project_names)} projects: {', '.join(project_names)}")

    for project_name in project_names:
        bugs_dir = projects_dir / project_name / "bugs"
        if not bugs_dir.exists():
            continue

        bug_dirs = sorted(
            (d for d in bugs_dir.iterdir() if d.is_dir()),
            key=lambda d: int(d.name) if d.name.isdigit() else 0,
        )

        for bug_dir in bug_dirs:
            bug_id = bug_dir.name

            bug_info = parse_bug_info(bug_dir / "bug.info")
            if not bug_info:
                continue

            patch_path = bug_dir / "bug_patch.txt"
            patch_text = ""
            if patch_path.exists():
                patch_text = patch_path.read_text(errors="ignore").strip()
            if not patch_text:
                continue

            # Read test command(s) for metadata
            run_test_path = bug_dir / "run_test.sh"
            test_commands = []
            if run_test_path.exists():
                for line in run_test_path.read_text(errors="ignore").strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        test_commands.append(line)

            # Extract changed files from patch
            changed_files = []
            for match in re.finditer(r'^diff --git a/(.+?) b/', patch_text, re.MULTILINE):
                changed_files.append(match.group(1))

            samples.append({
                "project": project_name,
                "bug_id": bug_id,
                "patch": patch_text,
                "test_commands": test_commands,
                "changed_files": changed_files,
                "buggy_commit": bug_info.get("buggy_commit_id", ""),
                "fixed_commit": bug_info.get("fixed_commit_id", ""),
                "test_file": bug_info.get("test_file", ""),
            })

            if len(samples) >= limit:
                break
        if len(samples) >= limit:
            break

    projects_found = set(s["project"] for s in samples)
    print(f"  Loaded {len(samples)} bugs from {len(projects_found)} projects")
    return samples


# =============================================================================
# Multi-fault combination logic
# =============================================================================

def create_multi_fault_combinations(
    samples: List[Dict],
    bugs_per_task: Tuple[int, int] = BUGS_PER_TASK,
    limit: int = LIMIT,
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """Combine 2-3 bugs from the SAME project into multi-fault task samples."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["project"]].append(sample)

    combined = []
    min_bugs, max_bugs = bugs_per_task

    for project_name in sorted(groups.keys()):
        project_bugs = groups[project_name]
        if len(project_bugs) < min_bugs:
            continue

        combos = []
        for k in range(min_bugs, max_bugs + 1):
            if k > len(project_bugs):
                continue
            if len(project_bugs) > 10:
                seen = set()
                for _ in range(min(200, len(project_bugs) * 10)):
                    chosen = tuple(sorted(rng.sample(range(len(project_bugs)), k)))
                    if chosen not in seen:
                        seen.add(chosen)
                        combos.append([project_bugs[i] for i in chosen])
                    if len(seen) >= 50:
                        break
            else:
                for indices in iter_combos(range(len(project_bugs)), k):
                    combos.append([project_bugs[i] for i in indices])

        rng.shuffle(combos)

        for bug_group in combos:
            combined.append(_merge_bugs_into_task(project_name, bug_group))
            if len(combined) >= limit:
                break
        if len(combined) >= limit:
            break

    rng.shuffle(combined)
    if len(combined) > limit:
        combined = combined[:limit]

    print(f"  Created {len(combined)} multi-fault combinations")
    return combined


def _merge_bugs_into_task(project_name: str, bugs: List[Dict]) -> Dict:
    """Merge multiple bugs from the same project into a single task sample."""
    # Build patch descriptions for LLM
    all_patches_parts = []
    for i, bug in enumerate(bugs, 1):
        all_patches_parts.append(
            f"### Bug {i} (ID: {bug['bug_id']}):\n"
            f"Patch (diff showing buggy -> fixed):\n```\n{bug['patch']}\n```"
        )

    return {
        "project": project_name,
        "bug_ids": [b["bug_id"] for b in bugs],
        "num_bugs": len(bugs),
        "bugs": bugs,
        "all_patches": "\n\n".join(all_patches_parts),
        "all_changed_files": list({f for b in bugs for f in b["changed_files"]}),
    }


# =============================================================================
# Task Generation
# =============================================================================

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


# Known BugsInPy project names (for import validation)
_BUGSINPY_PROJECTS = {
    "pysnooper", "ansible", "black", "cookiecutter", "fastapi",
    "httpie", "keras", "luigi", "matplotlib", "pandas",
    "sanic", "scrapy", "spacy", "thefuck", "tornado",
    "tqdm", "youtube-dl", "youtube_dl",
}

# Python stdlib modules
_STDLIB_MODULES = {
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


def _validate_buggy_code(code: str, project: str) -> str:
    """Remove project-internal imports that would fail in isolated container."""
    if not code:
        return code

    project_lower = project.lower().replace("-", "_")
    lines = code.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if stripped.startswith("from "):
                parts = stripped.split()
                module = parts[1].split(".")[0] if len(parts) >= 2 else ""
            else:
                parts = stripped.split()
                module = parts[1].split(".")[0] if len(parts) >= 2 else ""

            if module.lower() in _BUGSINPY_PROJECTS or module.lower() == project_lower:
                cleaned.append(f"# REMOVED: {stripped}  # project-internal import")
                continue

        cleaned.append(line)

    return "\n".join(cleaned)


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    test_code: str,
    buggy_code: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a multi-fault BugsInPy task."""
    # Python Dockerfile with buggy code COPY'd into /app
    base_dockerfile = get_dockerfile("python")
    if buggy_code:
        dockerfile = base_dockerfile.rstrip() + "\n\n# Copy buggy starter code into /app\nCOPY solution.py /app/solution.py\n"
    else:
        dockerfile = base_dockerfile

    test_sh = create_pytest_test_sh("/tests/test_solution.py")

    test_files = {"test_solution.py": test_code}

    metadata = {
        "source": "bugsinpy_mf",
        "variant": "multi-fault",
        "project": sample["project"],
        "bug_ids": sample["bug_ids"],
        "num_bugs": sample["num_bugs"],
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
    combined_samples: List[Dict],
    instructions: List[str],
    test_codes: List[str],
    buggy_codes: List[str],
    dataset_prefix: str = "bugsinpy_mf",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction, test_code, buggy_code) in enumerate(tqdm(
        zip(combined_samples, instructions, test_codes, buggy_codes),
        total=len(combined_samples),
        desc="Creating tasks",
    )):
        create_harbor_task(temp_dir, i, instruction, test_code, buggy_code, sample, dataset_prefix)

    print(f"Generated {len(combined_samples)} multi-fault harbor tasks!")
    return str(temp_dir)


# =============================================================================
# Main pipeline
# =============================================================================

def main(limit: int = LIMIT, upload: bool = True) -> Optional[str]:
    """Main pipeline for generating BugsInPy multi-fault tasks."""

    print("=" * 60)
    print("BugsInPy Multi-Fault Task Generator (v4 - self-contained)")
    print("=" * 60)

    # Step 1: Load bugs
    print("\nStep 1: Loading BugsInPy bugs from GitHub repo...")
    individual_bugs = load_bugsinpy_from_repo(limit=limit * 5)
    print(f"  -> {len(individual_bugs)} individual bugs loaded")

    if not individual_bugs:
        print("\nNo bugs found. Exiting.")
        return None

    # Step 2: Combine into multi-fault tasks
    print("\nStep 2: Creating multi-fault combinations (2-3 bugs per task)...")
    combined_samples = create_multi_fault_combinations(
        individual_bugs, bugs_per_task=BUGS_PER_TASK, limit=limit, seed=RANDOM_SEED,
    )
    print(f"  -> {len(combined_samples)} multi-fault tasks created")

    if not combined_samples:
        print("\nNo valid combinations. Need projects with >= 2 bugs.")
        return None

    # Step 3: Generate combined buggy code files
    print(f"\nStep 3: Generating combined buggy code files via {MODEL}...")
    llm_records = [
        {
            "project": s["project"],
            "num_bugs": str(s["num_bugs"]),
            "all_patches": s["all_patches"],
        }
        for s in combined_samples
    ]
    dataset = Dataset.from_list(llm_records)

    buggy_result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": GENERATE_COMBINED_BUGGY_CODE_PROMPT,
            "output_column": "buggy_code",
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    buggy_ds = buggy_result.dataset if hasattr(buggy_result, "dataset") else buggy_result
    buggy_codes = [_clean_llm_code_output(t) for t in buggy_ds["buggy_code"]]
    # Validate buggy codes - strip project-internal imports
    buggy_codes = [
        _validate_buggy_code(code, s["project"])
        for code, s in zip(buggy_codes, combined_samples)
    ]
    print(f"  -> Generated {len(buggy_codes)} buggy code files")

    # Step 4: Generate self-contained pytest tests (with buggy code context)
    print(f"\nStep 4: Generating pytest test files via {MODEL}...")
    test_records = [
        {
            "project": s["project"],
            "num_bugs": str(s["num_bugs"]),
            "all_patches": s["all_patches"],
            "buggy_code": bc,
            "_step": "test",
        }
        for s, bc in zip(combined_samples, buggy_codes)
    ]
    test_dataset = Dataset.from_list(test_records)

    test_result = run_completions(
        test_dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": GENERATE_COMBINED_TEST_PROMPT,
            "output_column": "generated_test",
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    test_ds = test_result.dataset if hasattr(test_result, "dataset") else test_result
    test_codes = [_clean_llm_code_output(t) for t in test_ds["generated_test"]]
    print(f"  -> Generated {len(test_codes)} test files")

    # Step 5: Generate task instructions
    print(f"\nStep 5: Generating task instructions via {MODEL}...")
    instr_records = [
        {
            "project": s["project"],
            "num_bugs": str(s["num_bugs"]),
            "all_patches": s["all_patches"],
            "_step": "instruction",
        }
        for s in combined_samples
    ]
    instr_dataset = Dataset.from_list(instr_records)

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
    instructions = list(instr_ds["task_description"])
    print(f"  -> Generated {len(instructions)} task descriptions")

    # Step 6: Create harbor tasks
    print("\nStep 6: Generating harbor task directories...")
    task_dir = generate_tasks(combined_samples, instructions, test_codes, buggy_codes)
    print(f"  -> Task directory: {task_dir}")

    # Step 7: Upload
    if upload:
        print("\nStep 7: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_bugsinpy-mf")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(combined_samples)} BugsInPy multi-fault tasks!")
    print(f"  Bugs per task: {BUGS_PER_TASK[0]}-{BUGS_PER_TASK[1]}")
    print(f"  Output directory: {task_dir}")
    print(f"  Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate multi-fault tasks from BugsInPy dataset"
    )
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--min-bugs", type=int, default=BUGS_PER_TASK[0])
    parser.add_argument("--max-bugs", type=int, default=BUGS_PER_TASK[1])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    args = parser.parse_args()
    BUGS_PER_TASK = (args.min_bugs, args.max_bugs)
    RANDOM_SEED = args.seed
    main(limit=args.limit, upload=not args.no_upload)
