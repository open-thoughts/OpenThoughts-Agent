#!/usr/bin/env python3
"""
Generate multi-file composition tasks by chaining 3 independent single-file
tasks into a dependent multi-module project.

Data source: DCAgent/exp_rpt_stack-pytest-v2 — groups consecutive tasks into
triples.  Extends the generate_task_stacking() pattern from exp_8_utils.py.

LLM calls (2x):
  1. Compose 3 tasks into unified instruction with inter-module imports
     (core.py -> processor.py -> integration.py)
  2. Generate combined pytest tests verifying all modules + integration

Expected pass rate: 15-35%
"""

import ast
import json
import os
import sys
import tempfile
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from data.completions import run_completions
from data.commons import (
    create_harbor_task_directory_generic,
    create_generic_test_sh,
    get_dockerfile,
    load_harbor_tasks_as_dicts,
    upload_tasks_to_hf,
)
from data.exp_8_utils import _clean_llm_test_code

# =============================================================================
# Configuration
# =============================================================================
LIMIT = int(os.environ.get("LIMIT", "15000"))
MODEL = os.environ.get("MODEL", "gpt-5-nano-2025-08-07")
PREFIX = "multifile"
HF_REPO = "DCAgent/exp_rpt_multifile"
SOURCE_REPOS = [
    "DCAgent/selfinstruct-naive-sandboxes-1",
    "DCAgent/selfinstruct-naive-sandboxes-2",
    "DCAgent/bash_textbook_tasks",
    "DCAgent/nl2bash",
    "DCAgent/exp_rpt_stack-pytest-v2",
]

# Shared Dockerfile for all tasks
DOCKERFILE = get_dockerfile("python")

# =============================================================================
# LLM Prompts
# =============================================================================

COMPOSE_PROMPT = """You are designing a multi-module Python project that combines three independent coding tasks into one cohesive project with inter-module dependencies.

TASK 1:
{{instruction_1}}

TASK 2:
{{instruction_2}}

TASK 3:
{{instruction_3}}

Design a unified project with these three modules:
- `/app/core.py` — foundational utilities (derived from Task 1)
- `/app/processor.py` — data processing that IMPORTS and USES functions from core.py (derived from Task 2)
- `/app/integration.py` — integration layer that IMPORTS and USES functions from BOTH core.py AND processor.py (derived from Task 3)

Requirements:
1. Each module must contain meaningful functionality (not just pass-throughs)
2. processor.py must `from core import ...` and use at least one function from core
3. integration.py must `from core import ...` and `from processor import ...` and use functions from both
4. All three modules must work together as a coherent project
5. Keep the implementation feasible — each module should be 30-80 lines

Output ONLY the task instruction in markdown format. Include:
- Project overview
- Detailed specification for each module (core.py, processor.py, integration.py)
- The import dependencies between modules
- Example usage showing how the modules work together

The agent should create files in /app/.
"""

TEST_PROMPT = """You are writing comprehensive pytest tests for a multi-module Python project.

PROJECT INSTRUCTION:
{{new_instruction}}

Write a pytest test file that:
1. Tests each module independently (core.py, processor.py, integration.py)
2. Tests the integration between modules (e.g., processor using core, integration using both)
3. Has clear test function names like test_core_*, test_processor_*, test_integration_*
4. Imports from the modules: `from core import ...`, `from processor import ...`, `from integration import ...`
5. Has 8-15 test functions total

Output ONLY valid Python code (no markdown fences). Start with import statements.
"""


# =============================================================================
# Pipeline
# =============================================================================


def group_into_triples(tasks: list[dict]) -> list[dict]:
    """Group consecutive tasks into triples for composition."""
    grouped = []
    for i in range(0, len(tasks) - 2, 3):
        grouped.append({
            "instruction_1": tasks[i]["instruction"],
            "instruction_2": tasks[i + 1]["instruction"],
            "instruction_3": tasks[i + 2]["instruction"],
            "source_1": tasks[i]["task_dir_name"],
            "source_2": tasks[i + 1]["task_dir_name"],
            "source_3": tasks[i + 2]["task_dir_name"],
        })
    return grouped


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate multi-file composition tasks")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Max source tasks to load")
    parser.add_argument("--model", default=MODEL, help="LLM model for generation")
    parser.add_argument("--no-upload", action="store_true", help="Skip HF upload")
    parser.add_argument("--local", action="store_true", help="Use local task dirs instead of HF")
    args = parser.parse_args()

    # Step 1: Load source tasks
    if args.local:
        from local_task_loader import load_local_tasks_as_dicts
        tasks = []
        for name in ["exercism", "bigcodebench", "codenet", "taco", "quixbugs",
                      "pymethods2test", "methods2test", "softwareheritage", "unitsyn"]:
            tasks.extend(load_local_tasks_as_dicts(dataset_name=name, limit=args.limit))
        tasks = tasks[:args.limit]
        print(f"Step 1: Loaded {len(tasks)} local source tasks")
    else:
        tasks = []
        for repo in SOURCE_REPOS:
            try:
                t = load_harbor_tasks_as_dicts(repo, limit=args.limit)
                print(f"  Loaded {len(t)} from {repo}")
                tasks.extend(t)
            except Exception as e:
                print(f"  Warning: Could not load {repo}: {e}")
        tasks = tasks[:args.limit]
        print(f"  Total: {len(tasks)} source tasks")

    # Step 2: Group into triples
    print("Step 2: Grouping into triples...")
    grouped = group_into_triples(tasks)
    print(f"  Created {len(grouped)} triples")

    if not grouped:
        print("No triples created. Exiting.")
        return

    dataset = Dataset.from_list(grouped)

    # Step 3: Compose unified instructions
    print(f"Step 3: Composing unified instructions with {args.model}...")
    result = run_completions(
        dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": COMPOSE_PROMPT,
            "output_column": "new_instruction",
        },
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"  Generated {len(result_dataset)} composed instructions")

    # Step 4: Generate integration tests
    print("Step 4: Generating integration tests...")
    result2 = run_completions(
        result_dataset,
        model=args.model,
        map_type="chat",
        map_config={
            "user_message": TEST_PROMPT,
            "output_column": "combined_tests",
        },
        require_all_responses=False,
    )
    result_dataset = result2.dataset
    print(f"  Generated {len(result_dataset)} test suites")

    # Step 5: Create harbor task directories
    print("Step 5: Creating harbor task directories...")
    out = Path(tempfile.mkdtemp(prefix=f"{PREFIX}_"))
    test_filename = "test_multifile.py"
    test_sh = create_generic_test_sh(
        test_command=f"pytest /tests/{test_filename} -v --tb=short",
        setup_commands="export PYTHONPATH=/app:${PYTHONPATH:-}",
    )
    created = 0

    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        new_instruction = row.get("new_instruction", "")
        combined_tests = row.get("combined_tests", "")

        if not new_instruction or len(new_instruction.strip()) < 100:
            continue
        if not combined_tests or len(combined_tests.strip()) < 50:
            continue

        cleaned_tests = _clean_llm_test_code(combined_tests)

        try:
            ast.parse(cleaned_tests)
        except SyntaxError:
            continue

        test_files = {test_filename: cleaned_tests}

        create_harbor_task_directory_generic(
            output_dir=out,
            task_id=created,
            instruction=new_instruction,
            dockerfile=DOCKERFILE,
            test_sh=test_sh,
            test_files=test_files,
            dataset_prefix=PREFIX,
            metadata={
                "source_1": row["source_1"],
                "source_2": row["source_2"],
                "source_3": row["source_3"],
                "task_type": "multifile_composition",
            },
        )
        created += 1

    print(f"  Created {created} valid tasks in {out}")

    if not args.no_upload and created > 0:
        print(f"\nStep 6: Uploading to {HF_REPO}...")
        repo_url = upload_tasks_to_hf(str(out), HF_REPO)
        print(f"  Done: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Multi-file composition generation complete!")
    print(f"  Tasks created: {created}")
    print(f"  Output: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
