#!/usr/bin/env python3
"""
Generate self-contained coding tasks from real GitHub issues (SWE-bench Lite).

Goal: Real GitHub issues → self-contained coding tasks with simplified repo
context. Unlike full SWE-bench (requires cloning entire repos), we extract only
the relevant 2-5 files and provide them as context in the instruction.

Data source: princeton-nlp/SWE-bench_Lite on HuggingFace (300 instances with
issue descriptions, patches, test patches).

LLM calls (2x):
  1. Simplify the issue + relevant code context into a self-contained task
     instruction (extract only the files that matter, provide them inline)
  2. Generate pytest tests from the test_patch field that verify the fix/feature

Expected pass rate: 10-30%
"""

import ast
import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from data.completions import run_completions
from data.commons import (
    create_harbor_task_directory_generic,
    create_generic_test_sh,
    upload_tasks_to_hf,
)
from data.commons import load_harbor_tasks_as_dicts
from data.exp_8_utils import _clean_llm_test_code

# =============================================================================
# Configuration
# =============================================================================
LIMIT = int(os.environ.get("LIMIT", "5000"))
MODEL = os.environ.get("MODEL", "gpt-5-nano-2025-08-07")
PREFIX = "issue"
HF_REPO = "DCAgent/exp_rpt_issue"

SOURCE_REPOS = [
    "DCAgent/selfinstruct-naive-sandboxes-1",
    "DCAgent/selfinstruct-naive-sandboxes-2",
    "DCAgent/bash_textbook_tasks",
    "DCAgent/nl2bash",
]

# Shared Python fat image
DOCKERFILE = """FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir pytest numpy pandas requests flask pyyaml
"""

# =============================================================================
# LLM Prompts
# =============================================================================

INSTRUCTION_PROMPT = """You are converting a real GitHub issue into a self-contained coding task.

REPOSITORY: {{repo}}

GITHUB ISSUE:
{{problem_statement}}

PATCH (the actual fix that was applied — do NOT reveal this to the agent):
{{patch}}

{{hints_section}}

Create a SELF-CONTAINED task instruction that:
1. Describes the bug or feature request clearly
2. Provides the relevant source code files INLINE as code blocks (extract only the 2-5 files that matter from the patch)
3. Shows the "before" state of the code (pre-fix version)
4. Tells the agent exactly which files to create/modify in /app/
5. Does NOT reveal the exact fix — describe the expected behavior instead
6. Is completable without cloning the full repository

Format the instruction as markdown. Include the relevant source files as ```python code blocks.
The agent should work in /app/ and create/modify files there.
"""

TEST_PROMPT = """You are generating pytest tests for a coding task based on a real GitHub issue fix.

TASK INSTRUCTION:
{{new_instruction}}

ORIGINAL TEST PATCH (real tests from the project — use as inspiration):
{{test_patch}}

Generate a self-contained pytest test file that:
1. Tests the fix/feature described in the instruction
2. Can run standalone (no dependency on the full repo)
3. Imports from files in /app/ (e.g., from solution import ..., or from the module names mentioned in the instruction)
4. Has 3-8 test functions
5. Tests both the happy path and edge cases

Output ONLY valid Python code (no markdown fences).
"""


# =============================================================================
# Data Loading
# =============================================================================


def _extract_files_from_patch(patch: str) -> Dict[str, List[str]]:
    """Extract file paths and changed lines from a unified diff patch."""
    files = {}
    current_file = None
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3].lstrip("b/")
                files[current_file] = []
        elif current_file and (line.startswith("+") and not line.startswith("+++")):
            files[current_file].append(line[1:])
    return files


def _load_harbor_issue_candidates(limit: int) -> List[Dict]:
    """Create issue-style candidates from large harbor task repos."""
    samples = []
    for repo in SOURCE_REPOS:
        try:
            tasks = load_harbor_tasks_as_dicts(repo, limit=limit)
            print(f"  Loaded {len(tasks)} from {repo}")
            for t in tasks:
                samples.append({
                    "instance_id": t["task_dir_name"],
                    "repo": repo,
                    "problem_statement": t["instruction"],
                    "patch": "",
                    "test_patch": "",
                    "hints_section": "",
                    "files_changed": 1,
                })
                if len(samples) >= limit:
                    break
        except Exception as e:
            print(f"  Warning: Could not load {repo}: {e}")
        if len(samples) >= limit:
            break
    print(f"  Created {len(samples)} issue candidates from harbor repos")
    return samples[:limit]


def load_swebench_all(limit: int) -> List[Dict]:
    """Load full SWE-bench + SWE-bench_Verified datasets."""
    samples = []
    for repo_name in ["princeton-nlp/SWE-bench", "princeton-nlp/SWE-bench_Verified"]:
        print(f"Loading {repo_name}...")
        try:
            ds = load_dataset(repo_name, split="test", streaming=True)
            for raw in tqdm(itertools.islice(ds, limit), desc=f"Loading {repo_name}"):
                instance_id = raw.get("instance_id", "")
                repo = raw.get("repo", "")
                problem_statement = raw.get("problem_statement", "")
                patch = raw.get("patch", "")
                test_patch = raw.get("test_patch", "")
                if not problem_statement or not patch:
                    continue
                hints = raw.get("hints_text", "")
                hints_section = f"HINTS:\n{hints}" if hints else ""
                files_changed = _extract_files_from_patch(patch)
                if len(files_changed) < 1 or len(files_changed) > 8:
                    continue
                samples.append({
                    "instance_id": instance_id,
                    "repo": repo,
                    "problem_statement": problem_statement,
                    "patch": patch,
                    "test_patch": test_patch,
                    "hints_section": hints_section,
                    "files_changed": len(files_changed),
                })
                if len(samples) >= limit:
                    break
        except Exception as e:
            print(f"  Warning: Could not load {repo_name}: {e}")
        if len(samples) >= limit:
            break
    print(f"Loaded {len(samples)} SWE-bench instances total")
    return samples[:limit]


def load_swebench_lite(limit: int) -> List[Dict]:
    """Load SWE-bench Lite dataset."""
    print("Loading princeton-nlp/SWE-bench_Lite...")

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", streaming=True)

    samples = []
    for raw in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading"):
        instance_id = raw.get("instance_id", "")
        repo = raw.get("repo", "")
        problem_statement = raw.get("problem_statement", "")
        patch = raw.get("patch", "")
        test_patch = raw.get("test_patch", "")

        if not problem_statement or not patch:
            continue

        # Build hints section
        hints = raw.get("hints_text", "")
        hints_section = f"HINTS:\n{hints}" if hints else ""

        # Count files touched
        files_changed = _extract_files_from_patch(patch)
        if len(files_changed) < 1 or len(files_changed) > 8:
            continue

        samples.append({
            "instance_id": instance_id,
            "repo": repo,
            "problem_statement": problem_statement,
            "patch": patch,
            "test_patch": test_patch,
            "hints_section": hints_section,
            "files_changed": len(files_changed),
        })

    print(f"Loaded {len(samples)} SWE-bench Lite instances")
    return samples


def _load_local_swebench(limit: int) -> List[Dict]:
    """Load swebench tasks from local benchmark directory."""
    from local_task_loader import load_local_tasks_as_dicts

    print("Step 1: Loading SWE-bench from local tasks...")
    local_tasks = load_local_tasks_as_dicts(dataset_name="swebench", limit=limit)
    samples = []
    for t in local_tasks:
        meta = json.loads(t.get("metadata", "{}"))
        samples.append({
            "instance_id": meta.get("instance_id", t["task_dir_name"]),
            "repo": meta.get("repo", "unknown/repo"),
            "problem_statement": t["instruction"],
            "patch": "",
            "test_patch": "",
            "hints_section": "",
            "files_changed": 1,
        })
    print(f"  Loaded {len(samples)} local SWE-bench instances")
    return samples


# =============================================================================
# Pipeline
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate issue-based tasks from SWE-bench")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Max instances")
    parser.add_argument("--model", default=MODEL, help="LLM model")
    parser.add_argument("--no-upload", action="store_true", help="Skip HF upload")
    parser.add_argument("--local", action="store_true", help="Use local swebench tasks instead of HF")
    args = parser.parse_args()

    # Step 1: Load data — combine SWE-bench + harbor repos to reach 5000
    if args.local:
        samples = _load_local_swebench(args.limit)
    else:
        print("Step 1: Loading SWE-bench (full + verified)...")
        samples = load_swebench_all(args.limit)
        if len(samples) < args.limit:
            print(f"  Only {len(samples)} from SWE-bench, supplementing from harbor repos...")
            remaining = args.limit - len(samples)
            harbor_samples = _load_harbor_issue_candidates(remaining)
            samples.extend(harbor_samples)
    if not samples:
        print("No samples loaded. Exiting.")
        return

    dataset = Dataset.from_list(samples)

    # Step 2: Generate self-contained instructions
    print(f"\nStep 2: Generating self-contained instructions with {args.model}...")
    result = run_completions(
        dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": INSTRUCTION_PROMPT,
            "output_column": "new_instruction",
        },
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"  Generated {len(result_dataset)} instructions")

    # Step 3: Generate tests
    print("Step 3: Generating tests...")
    result2 = run_completions(
        result_dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": TEST_PROMPT,
            "output_column": "new_tests",
        },
        require_all_responses=False,
    )
    result_dataset = result2.dataset
    print(f"  Generated {len(result_dataset)} test suites")

    # Step 4: Create harbor task directories
    print("Step 4: Creating harbor task directories...")
    out = Path(tempfile.mkdtemp(prefix=f"{PREFIX}_"))
    test_filename = "test_issue.py"
    test_sh = create_generic_test_sh(
        test_command=f"pytest /tests/{test_filename} -v --tb=short",
        setup_commands="export PYTHONPATH=/app:${PYTHONPATH:-}",
    )
    created = 0

    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        instruction = row.get("new_instruction", "")
        tests = row.get("new_tests", "")

        if not instruction or len(instruction.strip()) < 100:
            continue
        if not tests or len(tests.strip()) < 50:
            continue

        cleaned_tests = _clean_llm_test_code(tests)
        try:
            ast.parse(cleaned_tests)
        except SyntaxError:
            continue

        test_files = {test_filename: cleaned_tests}

        create_harbor_task_directory_generic(
            output_dir=out,
            task_id=created,
            instruction=instruction,
            dockerfile=DOCKERFILE,
            test_sh=test_sh,
            test_files=test_files,
            dataset_prefix=PREFIX,
            metadata={
                "instance_id": row.get("instance_id", ""),
                "repo": row.get("repo", ""),
                "files_changed": row.get("files_changed", 0),
                "task_type": "issue",
            },
        )
        created += 1

    print(f"  Created {created} valid tasks in {out}")

    if not args.no_upload and created > 0:
        print(f"\nStep 5: Uploading to {HF_REPO}...")
        repo_url = upload_tasks_to_hf(str(out), HF_REPO)
        print(f"  Done: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Issue task generation complete!")
    print(f"  Tasks created: {created}")
    print(f"  Output: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
