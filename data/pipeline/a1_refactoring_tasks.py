#!/usr/bin/env python3
"""
Generate refactoring tasks from bigcode/the-stack.

Goal: Give agent a working but messy codebase + refactoring instruction. Agent
must refactor while preserving functionality.

Data source: bigcode/the-stack (Python, streaming). Filter for files 200-800
lines with code smells: functions > 50 lines, duplicate code blocks, bare
`except:`, deeply nested callbacks.

LLM calls (3x):
  1. Analyze code and choose a refactoring type (extract module, add error
     handling, convert to async, etc.)
  2. Generate functional preservation tests (behavior must not change)
  3. Generate structural verification tests (check the refactoring actually
     happened — file existence, async def usage, class hierarchy, etc.)

Test strategy: Two-phase test.sh — both test_functional.py and
test_structure.py must pass for reward=1.

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
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from data.completions import run_completions
from data.commons import (
    create_harbor_task_directory_generic,
    get_dockerfile,
    upload_tasks_to_hf,
)
from data.exp_8_utils import _clean_llm_test_code

# =============================================================================
# Configuration
# =============================================================================
LIMIT = int(os.environ.get("LIMIT", "500"))
MODEL = os.environ.get("MODEL", "gpt-5-nano-2025-08-07")
PREFIX = "refactor"
HF_REPO = "DCAgent/exp_rpt_refactor"
MAX_SCAN = int(os.environ.get("MAX_SCAN", "500000"))

DOCKERFILE = get_dockerfile("python")

# =============================================================================
# LLM Prompts
# =============================================================================

ANALYZE_PROMPT = """You are a code reviewer analyzing messy Python code for refactoring opportunities.

CODE:
{{code}}

CODE SMELLS DETECTED:
{{smells}}

Analyze this code and create a refactoring task instruction. Choose ONE specific refactoring:
- Extract long functions into smaller, focused functions
- Replace bare except clauses with specific exception handling
- Extract duplicate code into shared helper functions
- Convert deeply nested code to use early returns or guard clauses
- Add proper type hints and docstrings
- Convert callback-heavy code to use classes or a cleaner pattern
- Split a monolithic file into logical modules

Create a clear refactoring instruction that:
1. Describes what the code currently does (functionality overview)
2. Identifies the specific code smell to fix
3. Specifies the refactoring approach to take
4. Lists constraints (must preserve all existing behavior)
5. Tells the agent to work in /app/

The original code will be available at /app/original.py. The agent should create the refactored version.

Output ONLY the task instruction in markdown format.
"""

FUNCTIONAL_TEST_PROMPT = """You are generating functional preservation tests for a refactoring task.

TASK INSTRUCTION:
{{refactoring_instruction}}

ORIGINAL CODE:
{{code}}

Generate pytest tests that verify the BEHAVIOR is preserved after refactoring:
1. Test all public functions/classes with their expected inputs and outputs
2. Test edge cases that the original code handles
3. Test that the API (function signatures, class interfaces) is preserved
4. Do NOT test internal implementation details (those will change)
5. Import from the refactored code in /app/

Have 5-10 test functions. Output ONLY valid Python code (no markdown fences).
"""

STRUCTURAL_TEST_PROMPT = """You are generating structural verification tests for a refactoring task.

TASK INSTRUCTION:
{{refactoring_instruction}}

Generate pytest tests that verify the REFACTORING was actually applied:
1. Check that the refactored code has the expected structure
2. Use introspection (inspect module, ast module, os.path checks) to verify structural changes
3. For example:
   - If extracting functions: check new functions exist
   - If adding error handling: check that bare except is gone (use ast)
   - If splitting into modules: check new files exist
   - If adding type hints: parse with ast and check annotations exist
   - If converting to class: check class definition exists

Have 3-6 test functions. Output ONLY valid Python code (no markdown fences).
"""

# =============================================================================
# Code Smell Detection
# =============================================================================


def _detect_smells(content: str) -> List[str]:
    """Detect code smells in Python code. Returns list of smell descriptions."""
    smells = []

    # Long functions (> 50 lines)
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_lines = node.end_lineno - node.lineno + 1 if node.end_lineno else 0
                if func_lines > 50:
                    smells.append(f"Long function `{node.name}` ({func_lines} lines)")
    except SyntaxError:
        pass

    # Bare except clauses
    if re.search(r"\bexcept\s*:", content):
        count = len(re.findall(r"\bexcept\s*:", content))
        smells.append(f"Bare except clauses ({count} instances)")

    # Deeply nested code (4+ indentation levels)
    max_indent = 0
    for line in content.split("\n"):
        if line.strip():
            indent = len(line) - len(line.lstrip())
            spaces = indent if line[0] == " " else indent * 4
            max_indent = max(max_indent, spaces)
    if max_indent >= 16:  # 4 levels of 4-space indent
        smells.append(f"Deeply nested code (max indent: {max_indent // 4} levels)")

    # Duplicate code blocks (simple check: repeated multi-line patterns)
    lines = content.split("\n")
    three_line_blocks = []
    for i in range(len(lines) - 2):
        block = "\n".join(line.strip() for line in lines[i : i + 3])
        if len(block) > 30:  # Non-trivial block
            three_line_blocks.append(block)
    seen = {}
    for b in three_line_blocks:
        seen[b] = seen.get(b, 0) + 1
    duplicates = sum(1 for v in seen.values() if v > 1)
    if duplicates > 0:
        smells.append(f"Duplicate code blocks ({duplicates} repeated patterns)")

    # Missing docstrings on public functions
    try:
        tree = ast.parse(content)
        missing_docs = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    if not (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, (ast.Str, ast.Constant))
                    ):
                        missing_docs += 1
        if missing_docs > 3:
            smells.append(f"Missing docstrings on {missing_docs} public functions")
    except SyntaxError:
        pass

    # No type hints
    if "def " in content and ": " not in content.split("def ")[1].split(")")[0]:
        smells.append("No type hints on function parameters")

    return smells


def _count_code_lines(content: str) -> int:
    """Count non-empty, non-comment lines."""
    return sum(
        1
        for line in content.split("\n")
        if line.strip() and not line.strip().startswith("#")
    )


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


# =============================================================================
# Data Loading
# =============================================================================


def load_messy_code(limit: int, max_scan: int) -> List[Dict]:
    """Stream the-stack looking for messy Python files with code smells."""
    print("Loading bigcode/the-stack (Python, streaming)...")
    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    samples = []
    scanned = 0
    pbar = tqdm(total=limit, desc="Finding messy code")

    for raw in itertools.islice(ds, max_scan):
        scanned += 1
        content = raw.get("content", "")
        path = raw.get("path", "")

        if not content:
            continue

        # Size filter: 200-800 lines
        code_lines = _count_code_lines(content)
        if code_lines < 200 or code_lines > 800:
            continue

        # Must be valid Python
        try:
            ast.parse(content)
        except SyntaxError:
            continue

        # No heavy deps
        if not _has_no_heavy_deps(content):
            continue

        # Must have code smells
        smells = _detect_smells(content)
        if len(smells) < 2:
            continue

        samples.append({
            "code": content,
            "path": path,
            "repo": raw.get("repository_name", "") or raw.get("repo_name", ""),
            "smells": "\n".join(f"- {s}" for s in smells),
            "num_smells": len(smells),
            "code_lines": code_lines,
        })
        pbar.update(1)

        if len(samples) >= limit:
            break

        if scanned % 10000 == 0:
            pbar.set_postfix({"scanned": f"{scanned:,}"})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(samples)} messy code candidates")
    return samples


# =============================================================================
# Two-Phase Test Script
# =============================================================================


def _create_refactor_test_sh() -> str:
    """Create two-phase test.sh: functional + structural tests must both pass."""
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
export PYTHONPATH="/app:${PYTHONPATH:-}"

# Copy original code for reference
cp /tests/original/original.py /app/original_backup.py 2>/dev/null || true

echo "=== Phase 1: Functional Preservation Tests ==="
pytest /tests/test_functional.py -v --tb=short 2>&1 | tee /logs/verifier/functional_output.txt
FUNC_EXIT=${PIPESTATUS[0]}

echo ""
echo "=== Phase 2: Structural Verification Tests ==="
pytest /tests/test_structure.py -v --tb=short 2>&1 | tee /logs/verifier/structural_output.txt
STRUCT_EXIT=${PIPESTATUS[0]}

echo ""
echo "=== Results ==="
echo "Functional tests: $([ $FUNC_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Structural tests: $([ $STRUCT_EXIT -eq 0 ] && echo PASS || echo FAIL)"

# Both must pass
if [ $FUNC_EXIT -eq 0 ] && [ $STRUCT_EXIT -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed."
    exit 1
fi
'''


def _load_local_refactor_candidates(limit: int) -> List[Dict]:
    """Create refactoring candidates from local task test code."""
    from local_task_loader import load_local_tasks_as_dicts

    print("Step 1: Loading refactoring candidates from local tasks...")
    samples = []
    for name in ["bigcodebench", "exercism", "codenet", "taco",
                  "pymethods2test", "softwareheritage", "unitsyn", "quixbugs"]:
        try:
            tasks = load_local_tasks_as_dicts(dataset_name=name, limit=limit)
            for t in tasks:
                test_files = json.loads(t.get("test_files_json", "{}"))
                # Concatenate test code as a "messy codebase"
                all_code = "\n\n".join(v for v in test_files.values() if isinstance(v, str))
                if not all_code or len(all_code) < 100:
                    continue

                # Add instruction code as part of the codebase
                code = t["instruction"] + "\n\n# Test code:\n" + all_code

                # Synthesize some smells for the prompt
                smells = [
                    "- Missing type hints on functions",
                    "- No docstrings on public functions",
                    "- Could benefit from extracting helper functions",
                ]

                samples.append({
                    "code": code,
                    "path": f"{name}/{t['task_dir_name']}",
                    "repo": name,
                    "smells": "\n".join(smells),
                    "num_smells": len(smells),
                    "code_lines": len(code.split("\n")),
                })
                if len(samples) >= limit:
                    break
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")
        if len(samples) >= limit:
            break
    print(f"  Created {len(samples)} refactoring candidates from local tasks")
    return samples[:limit]


# =============================================================================
# Pipeline
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate refactoring tasks from the-stack")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Max tasks")
    parser.add_argument("--model", default=MODEL, help="LLM model")
    parser.add_argument("--max-scan", type=int, default=MAX_SCAN, help="Max files to scan")
    parser.add_argument("--no-upload", action="store_true", help="Skip HF upload")
    parser.add_argument("--local", action="store_true", help="Use local tasks as code source")
    args = parser.parse_args()

    # Step 1: Load messy code
    if args.local:
        samples = _load_local_refactor_candidates(args.limit)
    else:
        print("Step 1: Finding messy code from the-stack...")
        samples = load_messy_code(args.limit, args.max_scan)
    if not samples:
        print("No messy code found. Exiting.")
        return

    dataset = Dataset.from_list(samples)

    # Step 2: Analyze and generate refactoring instructions
    print(f"\nStep 2: Generating refactoring instructions with {args.model}...")
    result = run_completions(
        dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": ANALYZE_PROMPT,
            "output_column": "refactoring_instruction",
        },
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"  Generated {len(result_dataset)} refactoring instructions")

    # Step 3: Generate functional preservation tests
    print("Step 3: Generating functional tests...")
    result2 = run_completions(
        result_dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": FUNCTIONAL_TEST_PROMPT,
            "output_column": "functional_tests",
        },
        require_all_responses=False,
    )
    result_dataset = result2.dataset
    print(f"  Generated {len(result_dataset)} functional test suites")

    # Step 4: Generate structural verification tests
    print("Step 4: Generating structural tests...")
    result3 = run_completions(
        result_dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": STRUCTURAL_TEST_PROMPT,
            "output_column": "structural_tests",
        },
        require_all_responses=False,
    )
    result_dataset = result3.dataset
    print(f"  Generated {len(result_dataset)} structural test suites")

    # Step 5: Create harbor task directories
    print("Step 5: Creating harbor task directories...")
    out = Path(tempfile.mkdtemp(prefix=f"{PREFIX}_"))
    test_sh = _create_refactor_test_sh()
    created = 0

    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        instruction = row.get("refactoring_instruction", "")
        func_tests = row.get("functional_tests", "")
        struct_tests = row.get("structural_tests", "")
        original_code = row.get("code", "")

        if not instruction or len(instruction.strip()) < 100:
            continue
        if not func_tests or len(func_tests.strip()) < 50:
            continue
        if not struct_tests or len(struct_tests.strip()) < 50:
            continue

        cleaned_func = _clean_llm_test_code(func_tests)
        cleaned_struct = _clean_llm_test_code(struct_tests)

        # Validate both test files
        try:
            ast.parse(cleaned_func)
            ast.parse(cleaned_struct)
        except SyntaxError:
            continue

        test_files = {
            "test_functional.py": cleaned_func,
            "test_structure.py": cleaned_struct,
            "original/original.py": original_code,
        }

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
                "path": row.get("path", ""),
                "num_smells": row.get("num_smells", 0),
                "smells": row.get("smells", ""),
                "code_lines": row.get("code_lines", 0),
                "task_type": "refactoring",
            },
        )
        created += 1

    print(f"  Created {created} valid tasks in {out}")

    if not args.no_upload and created > 0:
        print(f"\nStep 6: Uploading to {HF_REPO}...")
        repo_url = upload_tasks_to_hf(str(out), HF_REPO)
        print(f"  Done: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Refactoring task generation complete!")
    print(f"  Tasks created: {created}")
    print(f"  Output: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
