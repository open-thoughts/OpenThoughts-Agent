#!/usr/bin/env python3
"""
Mine multi-file code changes from GitHub commits and task the agent with
reproducing the change.

Data source: bigcode/commitpackft on HuggingFace (Python split, streaming).
Filter for commits touching 3+ files, 20-500 line diffs, commit messages > 50
chars. Reject merges, reverts, test-only changes.

LLM calls (2x):
  1. Convert commit message + pre-commit files + diff → task instruction
     (don't reveal the exact fix)
  2. Generate pytest tests from the diff that verify the change was made

Key design: Pre-commit file contents are included inline in instruction.md as
code blocks (agent sees the "before" state), NOT COPY'd into Docker.

Expected pass rate: 10-25%
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
LIMIT = int(os.environ.get("LIMIT", "500"))
MODEL = os.environ.get("MODEL", "gpt-5-nano-2025-08-07")
PREFIX = "pr"
HF_REPO = "DCAgent/exp_rpt_pr"
MAX_SCAN = int(os.environ.get("MAX_SCAN", "200000"))

SOURCE_REPOS = [
    "DCAgent/selfinstruct-naive-sandboxes-1",
    "DCAgent/selfinstruct-naive-sandboxes-2",
    "DCAgent/bash_textbook_tasks",
    "DCAgent/nl2bash",
    "DCAgent/exp_rpt_stack-pytest-v2",
]

# Shared Python fat image with common packages
DOCKERFILE = """FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir pytest numpy pandas requests flask pyyaml
"""

# =============================================================================
# LLM Prompts
# =============================================================================

INSTRUCTION_PROMPT = """You are converting a GitHub commit into a coding task.

COMMIT MESSAGE:
{{commit_message}}

PRE-COMMIT FILE CONTENTS (the "before" state — these files exist before the change):
{{old_contents}}

DIFF (the change that was made — do NOT reveal this directly):
{{diff}}

Create a task instruction that:
1. Describes what needs to be changed/added/fixed (based on the commit message and diff)
2. Includes the pre-commit file contents as ```python code blocks so the agent knows the starting state
3. Tells the agent which files to modify in /app/
4. Does NOT reveal the exact code changes — describe the desired behavior/outcome instead
5. Is specific enough that the agent can implement the change correctly

Format as markdown. The agent should work in /app/.
"""

TEST_PROMPT = """You are generating pytest tests to verify that a code change was correctly applied.

TASK INSTRUCTION:
{{new_instruction}}

DIFF (the expected change):
{{diff}}

Generate pytest tests that:
1. Verify the new/changed behavior described in the instruction
2. Import from files in /app/
3. Test that the change works correctly (not just that old behavior is preserved)
4. Have 3-8 test functions
5. Are self-contained and runnable

Output ONLY valid Python code (no markdown fences).
"""


# =============================================================================
# Data Loading & Filtering
# =============================================================================


def _parse_diff(diff_text: str) -> Dict[str, Dict]:
    """Parse a unified diff into per-file info."""
    files = {}
    current_file = None
    added_lines = 0
    removed_lines = 0

    for line in diff_text.split("\n"):
        if line.startswith("diff --git") or line.startswith("--- ") or line.startswith("+++ "):
            if line.startswith("+++ "):
                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                current_file = path
                files[current_file] = {"added": 0, "removed": 0}
        elif current_file:
            if line.startswith("+") and not line.startswith("+++"):
                files[current_file]["added"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                files[current_file]["removed"] += 1

    return files


def _is_merge_or_revert(message: str) -> bool:
    """Check if commit is a merge or revert."""
    msg_lower = message.lower().strip()
    return (
        msg_lower.startswith("merge ")
        or msg_lower.startswith("revert ")
        or "merge pull request" in msg_lower
        or "merge branch" in msg_lower
    )


def _is_test_only(files: Dict[str, Dict]) -> bool:
    """Check if commit only touches test files."""
    return all(
        "test" in f.lower() or f.startswith("tests/")
        for f in files.keys()
    )


def load_commitpackft(limit: int, max_scan: int) -> List[Dict]:
    """Load and filter commitpackft Python commits."""
    print("Loading bigcode/commitpackft (Python split, streaming)...")

    ds = load_dataset(
        "bigcode/commitpackft",
        split="train",
        streaming=True,
    )

    samples = []
    scanned = 0
    pbar = tqdm(total=limit, desc="Finding PR candidates")

    for raw in itertools.islice(ds, max_scan):
        scanned += 1

        # Filter for Python
        lang = raw.get("programming_language", "") or raw.get("lang", "")
        if lang.lower() != "python":
            continue

        message = raw.get("message", "") or raw.get("subject", "")
        old_contents = raw.get("old_contents", "") or raw.get("old_file", "")
        new_contents = raw.get("new_contents", "") or raw.get("new_file", "")

        if not message or len(message) < 50:
            continue

        if _is_merge_or_revert(message):
            continue

        if not old_contents or not new_contents:
            continue

        # Calculate diff size
        old_lines = old_contents.split("\n")
        new_lines = new_contents.split("\n")
        diff_size = abs(len(new_lines) - len(old_lines)) + sum(
            1 for a, b in zip(old_lines, new_lines) if a != b
        )

        if diff_size < 20 or diff_size > 500:
            continue

        # Validate Python syntax of new contents
        try:
            ast.parse(new_contents)
        except SyntaxError:
            continue

        # Build a simple diff representation
        diff_lines = []
        for line in old_lines:
            if line not in new_lines:
                diff_lines.append(f"- {line}")
        for line in new_lines:
            if line not in old_lines:
                diff_lines.append(f"+ {line}")
        diff = "\n".join(diff_lines[:100])  # Cap diff length

        samples.append({
            "commit_message": message,
            "old_contents": f"```python\n{old_contents}\n```",
            "new_contents": new_contents,
            "diff": diff,
            "diff_size": diff_size,
        })
        pbar.update(1)

        if len(samples) >= limit:
            break

        if scanned % 10000 == 0:
            pbar.set_postfix({"scanned": f"{scanned:,}"})

    pbar.close()
    print(f"Scanned {scanned:,} commits, found {len(samples)} candidates")
    return samples


def _load_harbor_pr_candidates(limit: int) -> List[Dict]:
    """Create PR-style candidates from large harbor task repos."""
    samples = []
    for repo in SOURCE_REPOS:
        try:
            tasks = load_harbor_tasks_as_dicts(repo, limit=limit)
            print(f"  Loaded {len(tasks)} from {repo}")
            for t in tasks:
                instruction = t["instruction"]
                test_files = json.loads(t.get("test_files_json", "{}"))
                test_code = "\n".join(v for v in test_files.values() if isinstance(v, str))
                samples.append({
                    "commit_message": f"Implement: {instruction[:200]}",
                    "old_contents": f"```python\n# Placeholder — original code before the change\npass\n```",
                    "new_contents": "",
                    "diff": f"# Task: {t['task_dir_name']}\n+ {instruction[:300]}",
                    "diff_size": len(instruction),
                })
                if len(samples) >= limit:
                    break
        except Exception as e:
            print(f"  Warning: Could not load {repo}: {e}")
        if len(samples) >= limit:
            break
    print(f"  Created {len(samples)} PR candidates from harbor repos")
    return samples[:limit]


def _load_local_pr_candidates(limit: int) -> List[Dict]:
    """Create PR-style candidates from local tasks by treating instruction as a change request."""
    from local_task_loader import load_local_tasks_as_dicts

    print("Step 1: Loading PR candidates from local tasks...")
    samples = []
    for name in ["bugsinpy", "bugswarm", "e2egit", "defects4j", "manybugs",
                  "exercism", "bigcodebench"]:
        try:
            tasks = load_local_tasks_as_dicts(dataset_name=name, limit=limit)
            for t in tasks:
                instruction = t["instruction"]
                test_files = json.loads(t.get("test_files_json", "{}"))
                test_code = "\n".join(v for v in test_files.values() if isinstance(v, str))

                # Simulate a commit: instruction is the "change request",
                # test code is the "before" state
                samples.append({
                    "commit_message": f"Implement: {instruction[:200]}",
                    "old_contents": f"```python\n# Placeholder — original code before the change\npass\n```",
                    "new_contents": "",
                    "diff": f"# Task from {name}: {t['task_dir_name']}\n+ {instruction[:300]}",
                    "diff_size": len(instruction),
                })
                if len(samples) >= limit:
                    break
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")
        if len(samples) >= limit:
            break
    print(f"  Created {len(samples)} PR candidates from local tasks")
    return samples[:limit]


# =============================================================================
# Pipeline
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate PR-based tasks from commitpackft")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Max tasks")
    parser.add_argument("--model", default=MODEL, help="LLM model")
    parser.add_argument("--max-scan", type=int, default=MAX_SCAN, help="Max commits to scan")
    parser.add_argument("--no-upload", action="store_true", help="Skip HF upload")
    parser.add_argument("--local", action="store_true", help="Use local tasks as PR source")
    args = parser.parse_args()

    # Step 1: Load data
    if args.local:
        samples = _load_local_pr_candidates(args.limit)
    else:
        print("Step 1: Loading PR candidates from harbor repos...")
        samples = _load_harbor_pr_candidates(args.limit)
    if not samples:
        print("No samples found. Exiting.")
        return

    dataset = Dataset.from_list(samples)

    # Step 2: Generate instructions
    print(f"\nStep 2: Generating task instructions with {args.model}...")
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
    test_filename = "test_pr.py"
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
                "commit_message": row.get("commit_message", "")[:200],
                "diff_size": row.get("diff_size", 0),
                "task_type": "pr",
            },
        )
        created += 1

    print(f"  Created {created} valid tasks in {out}")

    if not args.no_upload and created > 0:
        print(f"\nStep 5: Uploading to {HF_REPO}...")
        repo_url = upload_tasks_to_hf(str(out), HF_REPO)
        print(f"  Done: {repo_url}")

    print(f"\n{'='*60}")
    print(f"PR task generation complete!")
    print(f"  Tasks created: {created}")
    print(f"  Output: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
