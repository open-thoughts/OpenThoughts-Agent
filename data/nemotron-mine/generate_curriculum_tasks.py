#!/usr/bin/env python3
"""
Generate curriculum variants (easy/medium/hard) of existing tasks for progressive
difficulty training.

Loads tasks from multiple existing HF datasets and rewrites each at three
difficulty levels via two LLM calls per variant:
  1. Rewrite the instruction at the target difficulty
  2. Regenerate tests matching the new difficulty

Output prefixes: curriculum-easy-, curriculum-medium-, curriculum-hard-

Expected pass rates: Easy 40-60%, Medium 20-35%, Hard 5-15%
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
from data.exp_8_utils import _clean_llm_test_code, _fix_test_sh_filename

# =============================================================================
# Configuration
# =============================================================================
LIMIT = int(os.environ.get("LIMIT", "500"))
MODEL = os.environ.get("MODEL", "gpt-5-nano-2025-08-07")

SOURCE_REPOS = [
    "DCAgent/selfinstruct-naive-sandboxes-1",
    "DCAgent/selfinstruct-naive-sandboxes-2",
    "DCAgent/bash_textbook_tasks",
    "DCAgent/nl2bash",
    "DCAgent/exp_rpt_stack-pytest-v2",
]

DIFFICULTY_CONFIGS = {
    "easy": {
        "prefix": "curriculum-easy",
        "hf_repo": "DCAgent/exp_rpt_curriculum-easy",
        "instruction_guidance": (
            "Make this task EASIER by:\n"
            "- Reducing the number of requirements (pick the 1-2 most important)\n"
            "- Adding detailed hints about the approach and algorithm\n"
            "- Providing a skeleton or partial implementation\n"
            "- Removing edge cases and performance constraints\n"
            "- Simplifying input/output formats\n"
            "Keep it as a single-file task."
        ),
        "test_guidance": (
            "Generate EASY tests that:\n"
            "- Only test the happy path (basic correct behavior)\n"
            "- Use simple, small inputs\n"
            "- Have 3-5 test functions maximum\n"
            "- Don't test edge cases or error handling"
        ),
    },
    "medium": {
        "prefix": "curriculum-medium",
        "hf_repo": "DCAgent/exp_rpt_curriculum-medium",
        "instruction_guidance": (
            "Rewrite this task at MEDIUM difficulty by:\n"
            "- Keeping all core requirements but clarifying them\n"
            "- Adding 1-2 moderate edge cases\n"
            "- Providing some hints but not a full solution approach\n"
            "- Requiring reasonable input validation\n"
            "Keep it as a single-file task."
        ),
        "test_guidance": (
            "Generate MEDIUM difficulty tests that:\n"
            "- Test core functionality thoroughly\n"
            "- Include 1-2 edge case tests\n"
            "- Have 5-8 test functions\n"
            "- Test basic error handling"
        ),
    },
    "hard": {
        "prefix": "curriculum-hard",
        "hf_repo": "DCAgent/exp_rpt_curriculum-hard",
        "instruction_guidance": (
            "Make this task HARDER by:\n"
            "- Adding performance constraints (time/space complexity)\n"
            "- Requiring handling of many edge cases (empty input, huge input, unicode, etc.)\n"
            "- Adding concurrency or multi-file requirements\n"
            "- Requiring a specific design pattern or architecture\n"
            "- Making the specification deliberately terse with fewer hints\n"
            "Keep the solution language the same."
        ),
        "test_guidance": (
            "Generate HARD tests that:\n"
            "- Include stress tests with large inputs\n"
            "- Test many edge cases (empty, None, negative, unicode, etc.)\n"
            "- Have 8-15 test functions\n"
            "- Test performance/timing where specified\n"
            "- Test error handling and graceful failures\n"
            "- Include parametrized tests where appropriate"
        ),
    },
}

# =============================================================================
# LLM Prompts
# =============================================================================

INSTRUCTION_PROMPT = """You are rewriting a coding task at a specific difficulty level.

ORIGINAL TASK INSTRUCTION:
{{instruction}}

DIFFICULTY ADJUSTMENT:
{{difficulty_guidance}}

Rewrite the task instruction for the specified difficulty. Output ONLY the new task instruction in markdown format. Keep the same programming language and general topic. The solution should be placed in /app/.
"""

TEST_PROMPT = """You are generating pytest tests for a coding task at a specific difficulty level.

TASK INSTRUCTION:
{{new_instruction}}

TEST DIFFICULTY GUIDANCE:
{{test_guidance}}

Generate a complete pytest test file. Output ONLY valid Python code (no markdown fences). The test file should:
- Import from the solution module (e.g., from solution import func_name)
- Use pytest assertions
- Be runnable with: pytest /tests/test_curriculum.py
"""


# =============================================================================
# Pipeline
# =============================================================================


def load_source_tasks(limit_per_repo: int, local: bool = False) -> list[dict]:
    """Load and merge tasks from all source repos (or local dirs)."""
    all_tasks = []

    if local:
        from local_task_loader import load_local_tasks_as_dicts
        LOCAL_DATASETS = ["exercism", "bigcodebench", "codenet", "taco", "quixbugs"]
        for name in LOCAL_DATASETS:
            try:
                tasks = load_local_tasks_as_dicts(dataset_name=name, limit=limit_per_repo)
                print(f"  Loaded {len(tasks)} from local:{name}")
                all_tasks.extend(tasks)
            except Exception as e:
                print(f"  Warning: Could not load local:{name}: {e}")
        return all_tasks

    for repo in SOURCE_REPOS:
        try:
            tasks = load_harbor_tasks_as_dicts(repo, limit=limit_per_repo)
            print(f"  Loaded {len(tasks)} from {repo}")
            all_tasks.extend(tasks)
        except Exception as e:
            print(f"  Warning: Could not load {repo}: {e}")
    return all_tasks


def generate_difficulty_variant(
    tasks: list[dict],
    difficulty: str,
    config: dict,
    model: str,
    upload: bool = True,
) -> str:
    """Generate one difficulty variant for all tasks."""
    prefix = config["prefix"]
    hf_repo = config["hf_repo"]

    print(f"\n{'='*60}")
    print(f"Generating {difficulty.upper()} variants ({len(tasks)} tasks)")
    print(f"{'='*60}")

    # Annotate tasks with difficulty metadata + truncate long instructions
    MAX_INSTRUCTION_LEN = 8000  # Prevent 2GB LLM responses
    for t in tasks:
        t["difficulty_guidance"] = config["instruction_guidance"]
        t["test_guidance"] = config["test_guidance"]
        if len(t.get("instruction", "")) > MAX_INSTRUCTION_LEN:
            t["instruction"] = t["instruction"][:MAX_INSTRUCTION_LEN] + "\n\n[... truncated for brevity]"

    dataset = Dataset.from_list(tasks)

    # Step 1: Rewrite instructions
    print("Step 1: Rewriting instructions...")
    result = run_completions(
        dataset,
        model=model,
        map_type="chat",
        map_config={
            "user_message": INSTRUCTION_PROMPT,
            "output_column": "new_instruction",
        },
        require_all_responses=False,
        generation_params={"max_completion_tokens": 4096},
    )
    result_dataset = result.dataset
    print(f"  Generated {len(result_dataset)} instruction rewrites")

    # Step 2: Generate tests
    print("Step 2: Generating tests...")
    result2 = run_completions(
        result_dataset,
        model=model,
        map_type="chat",
        map_config={
            "user_message": TEST_PROMPT,
            "output_column": "new_tests",
        },
        require_all_responses=False,
        generation_params={"max_completion_tokens": 4096},
    )
    result_dataset = result2.dataset
    print(f"  Generated {len(result_dataset)} test suites")

    # Step 3: Create harbor task directories
    print("Step 3: Creating harbor task directories...")
    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    test_filename = "test_curriculum.py"
    created = 0

    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        new_instruction = row.get("new_instruction", "")
        new_tests = row.get("new_tests", "")

        if not new_instruction or len(new_instruction.strip()) < 50:
            continue
        if not new_tests or len(new_tests.strip()) < 30:
            continue

        cleaned_tests = _clean_llm_test_code(new_tests)

        # Validate test syntax
        try:
            ast.parse(cleaned_tests)
        except SyntaxError:
            continue

        test_files = {test_filename: cleaned_tests}
        # Always generate a proper test.sh with PYTHONPATH and cleanup
        test_sh = create_generic_test_sh(
            test_command=f"pytest /tests/{test_filename} -v --tb=short",
            setup_commands="export PYTHONPATH=/app:${PYTHONPATH:-}",
        )
        # Use a known-good Dockerfile instead of source (which may be minimal)
        dockerfile = row.get("dockerfile", "")
        if not dockerfile or "pytest" not in dockerfile:
            dockerfile = get_dockerfile("python")

        create_harbor_task_directory_generic(
            output_dir=out,
            task_id=created,
            instruction=new_instruction,
            dockerfile=dockerfile,
            test_sh=test_sh,
            test_files=test_files,
            dataset_prefix=prefix,
            metadata={
                "source": row["task_dir_name"],
                "difficulty": difficulty,
                "source_repo": "mixed",
            },
        )
        created += 1

    print(f"  Created {created} valid tasks in {out}")

    if upload and created > 0:
        print(f"\nStep 4: Uploading to {hf_repo}...")
        repo_url = upload_tasks_to_hf(str(out), hf_repo)
        print(f"  Done: {repo_url}")

    return str(out)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate curriculum task variants")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Tasks per source repo")
    parser.add_argument("--model", default=MODEL, help="LLM model for generation")
    parser.add_argument("--no-upload", action="store_true", help="Skip HF upload")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which difficulty to generate",
    )
    parser.add_argument("--local", action="store_true", help="Use local task dirs instead of HF")
    args = parser.parse_args()

    print(f"Loading source tasks (limit={args.limit} per repo, local={args.local})...")
    tasks = load_source_tasks(args.limit, local=args.local)
    print(f"Total source tasks: {len(tasks)}")

    if not tasks:
        print("No source tasks loaded. Exiting.")
        return

    difficulties = (
        list(DIFFICULTY_CONFIGS.keys())
        if args.difficulty == "all"
        else [args.difficulty]
    )

    results = {}
    for diff in difficulties:
        config = DIFFICULTY_CONFIGS[diff]
        out_dir = generate_difficulty_variant(
            tasks, diff, config, args.model, upload=not args.no_upload
        )
        results[diff] = out_dir

    print(f"\n{'='*60}")
    print("Curriculum generation complete!")
    for diff, out_dir in results.items():
        print(f"  {diff}: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
