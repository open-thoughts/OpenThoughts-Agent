#!/usr/bin/env python3
"""
Generate repo scaffolding tasks from bigcode/the-stack.

Goal: Give the agent a codebase with function signatures but empty bodies
(raise NotImplementedError). Agent fills in implementations to pass tests.

Data source: bigcode/the-stack (Python, streaming). Find repos with co-located
test files and implementation files. Require 2-4 implementation files per task,
30-200 lines each.

LLM calls (2x):
  1. Generate scaffolded versions of implementation files (keep signatures +
     docstrings, remove bodies)
  2. Generate task instruction from scaffold + tests describing what to implement

Test strategy: Original test files from the repo (unmodified — highest quality
tests).

Expected pass rate: 20-40%
"""

import ast
import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from data.completions import run_completions
from data.commons import (
    create_harbor_task_directory_generic,
    create_generic_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)
from data.commons import load_harbor_tasks_as_dicts
from data.exp_8_utils import _clean_llm_test_code

# =============================================================================
# Configuration
# =============================================================================
LIMIT = int(os.environ.get("LIMIT", "500"))
MODEL = os.environ.get("MODEL", "gpt-5-nano-2025-08-07")
PREFIX = "scaffold"
HF_REPO = "DCAgent/exp_rpt_scaffold"
MAX_SCAN = int(os.environ.get("MAX_SCAN", "500000"))

SOURCE_REPOS = [
    "DCAgent/selfinstruct-naive-sandboxes-1",
    "DCAgent/selfinstruct-naive-sandboxes-2",
    "DCAgent/bash_textbook_tasks",
    "DCAgent/nl2bash",
    "DCAgent/exp_rpt_stack-pytest-v2",
]

DOCKERFILE = get_dockerfile("python")

# =============================================================================
# LLM Prompts
# =============================================================================

SCAFFOLD_PROMPT = """You are creating a scaffolded version of a Python file for a coding exercise.

ORIGINAL IMPLEMENTATION:
{{implementation_code}}

Create a scaffolded version that:
1. Keeps ALL function/method signatures exactly as they are
2. Keeps ALL class definitions and their __init__ signatures
3. Keeps ALL docstrings and type hints
4. Replaces ALL function/method bodies with: raise NotImplementedError("TODO: implement this")
5. Keeps all import statements at the top
6. Keeps module-level constants and type aliases

Output ONLY the scaffolded Python code (no markdown fences, no explanation).
"""

INSTRUCTION_PROMPT = """You are creating a coding task instruction based on scaffolded code and its test suite.

SCAFFOLDED CODE (the agent will receive this as starter code):
{{scaffold_code}}

TEST CODE (the agent must make these tests pass):
{{test_code}}

Write a clear task instruction that:
1. Explains what needs to be implemented (fill in the NotImplementedError stubs)
2. Describes the purpose of each function/class based on the tests and docstrings
3. Lists the files the agent needs to modify (in /app/)
4. Mentions that tests will run via pytest
5. Does NOT reveal the exact implementation — describe behavior, not code

Output ONLY the task instruction in markdown format. The agent's code goes in /app/.
"""

GENERATE_TEST_PROMPT = """You are generating pytest tests for a scaffold-based coding task.

TASK INSTRUCTION:
{{task_instruction}}

SCAFFOLDED CODE (function signatures the agent must implement):
{{scaffold_code}}

Generate a comprehensive pytest test file that:
1. Tests each function/method in the scaffold
2. Imports from the solution module (e.g., from solution import func_name, or based on filenames in the instruction)
3. Has 5-10 test functions
4. Tests both normal cases and edge cases
5. Uses pytest assertions

Output ONLY valid Python code (no markdown fences).
"""


# =============================================================================
# Data Loading & Filtering
# =============================================================================


def _is_test_file(path: str, content: str) -> bool:
    """Check if a file is a Python test file."""
    basename = path.split("/")[-1] if "/" in path else path
    if not basename.startswith("test_") and not basename.endswith("_test.py"):
        return False
    if not basename.endswith(".py"):
        return False
    return bool(re.search(r"def test_\w+", content))


def _is_impl_file(path: str, content: str) -> bool:
    """Check if a file is a Python implementation file (not a test)."""
    basename = path.split("/")[-1] if "/" in path else path
    if not basename.endswith(".py"):
        return False
    if basename.startswith("test_") or basename.endswith("_test.py"):
        return False
    if basename in ("setup.py", "conftest.py", "__init__.py", "setup.cfg"):
        return False
    # Must have function or class definitions
    return bool(re.search(r"(?:def |class )\w+", content))


def _count_code_lines(content: str) -> int:
    """Count non-empty, non-comment lines."""
    return sum(
        1
        for line in content.split("\n")
        if line.strip() and not line.strip().startswith("#")
    )


def _validate_python(content: str) -> bool:
    """Check if content is valid Python."""
    try:
        ast.parse(content)
        return True
    except SyntaxError:
        return False


def _has_no_heavy_deps(content: str) -> bool:
    """Reject files importing heavy frameworks."""
    HEAVY = {
        "django", "flask", "fastapi", "tornado", "tensorflow", "torch",
        "keras", "sklearn", "scipy", "pandas", "numpy", "matplotlib",
        "sqlalchemy", "celery", "redis", "boto3", "requests", "httpx",
        "selenium", "playwright", "cv2", "PIL",
    }
    imports = re.findall(r"(?:from|import)\s+([\w.]+)", content)
    top_levels = {imp.split(".")[0].lower() for imp in imports}
    return not (top_levels & HEAVY)


def load_stack_repo_groups(limit: int, max_scan: int) -> List[Dict]:
    """
    Stream bigcode/the-stack (Python) looking for repos with test + impl files.

    Groups files by repository and selects repos that have:
    - At least 1 test file
    - 2-4 implementation files with 30-200 code lines each
    """
    print("Loading bigcode/the-stack (Python, streaming)...")
    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    # Group by repo
    repo_files: Dict[str, Dict[str, str]] = {}  # repo -> {path: content}
    scanned = 0
    found = 0

    samples = []
    pbar = tqdm(total=limit, desc="Finding scaffold candidates")

    for raw in itertools.islice(ds, max_scan):
        scanned += 1
        repo = raw.get("repository_name", "") or raw.get("repo_name", "")
        path = raw.get("path", "")
        content = raw.get("content", "")

        if not repo or not path or not content:
            continue

        if repo not in repo_files:
            repo_files[repo] = {}
        repo_files[repo][path] = content

        # Periodically check if any repos are ready
        if scanned % 5000 == 0:
            ready_repos = []
            for r, files in list(repo_files.items()):
                test_files = {
                    p: c for p, c in files.items()
                    if _is_test_file(p, c) and _validate_python(c)
                }
                impl_files = {
                    p: c for p, c in files.items()
                    if _is_impl_file(p, c)
                    and _validate_python(c)
                    and 30 <= _count_code_lines(c) <= 200
                    and _has_no_heavy_deps(c)
                }

                if test_files and 2 <= len(impl_files) <= 4:
                    ready_repos.append(r)
                    all_test_code = "\n\n".join(
                        f"# {p}\n{c}" for p, c in test_files.items()
                    )
                    samples.append({
                        "repo": r,
                        "test_code": all_test_code,
                        "test_files": test_files,
                        "impl_files": impl_files,
                        "implementation_code": "\n\n".join(
                            f"# {p}\n{c}" for p, c in impl_files.items()
                        ),
                    })
                    pbar.update(1)
                    found += 1

            # Remove processed repos
            for r in ready_repos:
                del repo_files[r]

            if found >= limit:
                break

            pbar.set_postfix({"scanned": f"{scanned:,}", "repos": len(repo_files)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(samples)} scaffold candidates")
    return samples[:limit]


def _load_harbor_scaffold_candidates(limit: int) -> List[Dict]:
    """Load scaffold candidates from large harbor task repos."""
    samples = []
    for repo in SOURCE_REPOS:
        try:
            tasks = load_harbor_tasks_as_dicts(repo, limit=limit)
            print(f"  Loaded {len(tasks)} from {repo}")
            for t in tasks:
                instruction = t.get("instruction", "")
                test_files_raw = json.loads(t.get("test_files_json", "{}"))
                test_code = "\n\n".join(
                    f"# {fn}\n{content}"
                    for fn, content in test_files_raw.items()
                    if fn.endswith(".py")
                )
                if not test_code or len(test_code) < 50:
                    # No Python test files — use instruction as implementation source
                    test_code = ""
                samples.append({
                    "repo": repo,
                    "test_code": test_code or "# No test code available",
                    "test_files": test_files_raw,
                    "impl_files": {},
                    "implementation_code": instruction,
                })
                if len(samples) >= limit:
                    break
        except Exception as e:
            print(f"  Warning: Could not load {repo}: {e}")
        if len(samples) >= limit:
            break
    print(f"  Total: {len(samples)} scaffold candidates")
    return samples[:limit]


def _load_local_scaffold_candidates(limit: int) -> List[Dict]:
    """Load scaffold candidates from local benchmark tasks."""
    from local_task_loader import load_local_tasks_as_dicts

    print("Step 1: Loading scaffold candidates from local tasks...")
    samples = []
    for dataset_name in ["exercism", "bigcodebench", "codenet", "quixbugs",
                         "pymethods2test", "softwareheritage", "unitsyn"]:
        try:
            tasks = load_local_tasks_as_dicts(dataset_name=dataset_name, limit=limit)
            for t in tasks:
                test_files_raw = json.loads(t.get("test_files_json", "{}"))
                test_code = "\n\n".join(
                    f"# {fn}\n{content}"
                    for fn, content in test_files_raw.items()
                    if fn.endswith(".py")
                )
                if not test_code or len(test_code) < 50:
                    continue
                # Use instruction + test code as the "implementation" to scaffold
                samples.append({
                    "repo": dataset_name,
                    "test_code": test_code,
                    "test_files": test_files_raw,
                    "impl_files": {},
                    "implementation_code": t["instruction"],
                })
                if len(samples) >= limit:
                    break
        except Exception as e:
            print(f"  Warning: Could not load {dataset_name}: {e}")
        if len(samples) >= limit:
            break
    print(f"  Loaded {len(samples)} scaffold candidates from local tasks")
    return samples[:limit]


# =============================================================================
# Pipeline
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate repo scaffolding tasks")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Max tasks")
    parser.add_argument("--model", default=MODEL, help="LLM model")
    parser.add_argument("--max-scan", type=int, default=MAX_SCAN, help="Max files to scan")
    parser.add_argument("--no-upload", action="store_true", help="Skip HF upload")
    parser.add_argument("--local", action="store_true", help="Use local task dirs instead of the-stack")
    args = parser.parse_args()

    # Step 1: Load data
    if args.local:
        samples = _load_local_scaffold_candidates(args.limit)
    else:
        print("Step 1: Loading scaffold candidates from harbor repos...")
        samples = _load_harbor_scaffold_candidates(args.limit)
    if not samples:
        print("No scaffold candidates found. Exiting.")
        return

    dataset = Dataset.from_list(samples)

    # Step 2: Generate scaffolds
    print(f"\nStep 2: Generating scaffolds with {args.model}...")
    result = run_completions(
        dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": SCAFFOLD_PROMPT,
            "output_column": "scaffold_code",
        },
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"  Generated {len(result_dataset)} scaffolds")

    # Step 3: Generate instructions
    print("Step 3: Generating task instructions...")
    result2 = run_completions(
        result_dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": INSTRUCTION_PROMPT,
            "output_column": "task_instruction",
        },
        require_all_responses=False,
    )
    result_dataset = result2.dataset
    print(f"  Generated {len(result_dataset)} instructions")

    # Step 4: Generate tests
    print("Step 4: Generating tests...")
    result3 = run_completions(
        result_dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": GENERATE_TEST_PROMPT,
            "output_column": "generated_tests",
        },
        require_all_responses=False,
    )
    result_dataset = result3.dataset
    print(f"  Generated {len(result_dataset)} test suites")

    # Step 5: Create harbor task directories
    print("Step 5: Creating harbor task directories...")
    out = Path(tempfile.mkdtemp(prefix=f"{PREFIX}_"))
    created = 0

    test_filename = "test_scaffold.py"
    test_sh = create_generic_test_sh(
        test_command=f"pytest /tests/{test_filename} -v --tb=short",
        setup_commands=(
            "# Copy scaffold files as starting point\n"
            "cp /tests/starter/*.py /app/ 2>/dev/null || true\n"
            "export PYTHONPATH=/app:${PYTHONPATH:-}"
        ),
    )

    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        instruction = row.get("task_instruction", "")
        scaffold = row.get("scaffold_code", "")

        if not instruction or len(instruction.strip()) < 100:
            continue
        if not scaffold or len(scaffold.strip()) < 50:
            continue

        # Use generated tests
        generated_tests = row.get("generated_tests", "")
        if not generated_tests or len(generated_tests.strip()) < 30:
            continue
        cleaned_tests = _clean_llm_test_code(generated_tests)
        try:
            ast.parse(cleaned_tests)
        except SyntaxError:
            continue

        test_files = {test_filename: cleaned_tests}
        # Add scaffold files under starter/ subdirectory
        test_files["starter/scaffold.py"] = scaffold

        create_harbor_task_directory_generic(
            output_dir=out,
            task_id=created,
            instruction=instruction,
            dockerfile=DOCKERFILE,
            test_sh=test_sh,
            test_files=test_files,
            dataset_prefix=PREFIX,
            metadata={
                "repo": row.get("repo", ""),
                "task_type": "scaffold",
                "num_impl_files": len(row.get("impl_files", {})),
            },
        )
        created += 1

    print(f"  Created {created} valid tasks in {out}")

    if not args.no_upload and created > 0:
        print(f"\nStep 6: Uploading to {HF_REPO}...")
        repo_url = upload_tasks_to_hf(str(out), HF_REPO)
        print(f"  Done: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Scaffold generation complete!")
    print(f"  Tasks created: {created}")
    print(f"  Output: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
