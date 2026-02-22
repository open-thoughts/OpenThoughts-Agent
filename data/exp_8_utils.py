#!/usr/bin/env python3
"""Shared utilities for Theme 8 experiment generators."""

import json
import sys
import tempfile
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from data.completions import run_completions
from data.commons import (
    create_harbor_task_directory_generic,
    load_harbor_tasks_as_dicts,
    upload_tasks_to_hf,
)

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_SOURCE_REPO = "DCAgent/exp_rpt_stack-pytest"
DEFAULT_LIMIT = 10_000


def _clean_llm_test_code(code: str) -> str:
    """Clean up common LLM artifacts in generated test code.

    Fixes: markdown fences, missing pytest import, and validates syntax.
    Returns cleaned code, or original if unfixable (caller should check syntax).
    """
    import ast
    import re

    # Strip markdown code fences (```python ... ```)
    code = re.sub(r'^[^\n]*```python\s*\n', '', code)
    code = re.sub(r'^[^\n]*```\s*\n', '', code)  # opening fence without language
    code = re.sub(r'\n```\s*$', '', code)
    # Strip leading filename line (e.g., "test_solution.py\n")
    code = re.sub(r'^test_\w+\.py\s*\n', '', code)

    # Ensure import pytest if pytest is used
    if 'pytest' in code and 'import pytest' not in code:
        code = 'import pytest\n\n' + code

    # Try to compile; if it works, return
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    # Attempt auto-fixes for common LLM escape issues
    # Fix escaped quotes in strings: \\' -> '
    fixed = code.replace("\\\\'", "'")
    # Fix escaped double quotes: \\" -> "
    fixed = fixed.replace('\\"', '"')
    try:
        ast.parse(fixed)
        return fixed
    except SyntaxError:
        pass

    # Return original (broken) - caller should handle
    return code


def _fix_test_sh_filename(test_sh: str, test_filename: str) -> str:
    """Rewrite test.sh to reference the correct test filename instead of test_solution.py."""
    import re
    # Replace pytest /tests/test_solution.py (or any test_*.py) with the correct filename
    test_sh = re.sub(
        r'pytest\s+/tests/test_\w+\.py',
        f'pytest /tests/{test_filename}',
        test_sh,
    )
    return test_sh


def generate_instruction_transform(
    prompt: str,
    prefix: str,
    hf_repo: str,
    variant_col: str,
    variant_value: str,
    source_repo: str = DEFAULT_SOURCE_REPO,
    limit: int = DEFAULT_LIMIT,
    model: str = DEFAULT_MODEL,
):
    """Category A: Single LLM call to transform instructions."""
    tasks = load_harbor_tasks_as_dicts(source_repo, limit=limit)
    print(f"Loaded {len(tasks)} source tasks")

    for t in tasks:
        t[variant_col] = variant_value
    dataset = Dataset.from_list(tasks)

    result = run_completions(
        dataset, model=model, map_type="chat",
        map_config={"user_message": prompt, "output_column": "new_instruction"},
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"Generated {len(result_dataset)} completions")

    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["new_instruction"],
            dockerfile=row["dockerfile"],
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix=prefix,
            metadata={"source": row["task_dir_name"], variant_col: row[variant_col]},
        )

    if hf_repo:
        print(f"\nUploading to {hf_repo}...")
        repo_url = upload_tasks_to_hf(str(out), hf_repo)
        print(f"Done: {repo_url}")
    return str(out)


def generate_instruction_and_tests_transform(
    instruction_prompt: str,
    test_prompt: str,
    test_filename: str,
    prefix: str,
    hf_repo: str,
    variant_col: str,
    variant_value: str,
    source_repo: str = DEFAULT_SOURCE_REPO,
    limit: int = DEFAULT_LIMIT,
    model: str = DEFAULT_MODEL,
):
    """Category B: Two LLM calls — instruction rewrite + test rewrite."""
    tasks = load_harbor_tasks_as_dicts(source_repo, limit=limit)
    print(f"Loaded {len(tasks)} source tasks")

    for t in tasks:
        t[variant_col] = variant_value
    dataset = Dataset.from_list(tasks)

    result = run_completions(
        dataset, model=model, map_type="chat",
        map_config={"user_message": instruction_prompt, "output_column": "new_instruction"},
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"Generated {len(result_dataset)} instruction completions")

    result2 = run_completions(
        result_dataset, model=model, map_type="chat",
        map_config={"user_message": test_prompt, "output_column": "new_tests"},
        require_all_responses=False,
    )
    result_dataset = result2.dataset
    print(f"Generated {len(result_dataset)} test completions")

    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        cleaned_tests = _clean_llm_test_code(row["new_tests"])
        test_files = {test_filename: cleaned_tests}
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["new_instruction"],
            dockerfile=row["dockerfile"],
            test_sh=_fix_test_sh_filename(row["test_sh"], test_filename),
            test_files=test_files,
            dataset_prefix=prefix,
            metadata={"source": row["task_dir_name"], variant_col: row[variant_col]},
        )

    print(f"\nUploading to {hf_repo}...")
    repo_url = upload_tasks_to_hf(str(out), hf_repo)
    print(f"Done: {repo_url}")
    return str(out)


def generate_error_state(
    broken_code_prompt: str,
    instruction_prompt: str,
    prefix: str,
    hf_repo: str,
    variant_col: str,
    variant_value: str,
    source_repo: str = DEFAULT_SOURCE_REPO,
    limit: int = DEFAULT_LIMIT,
    model: str = DEFAULT_MODEL,
):
    """Category C (8.6): Broken starter code + debug-and-fix instruction."""
    tasks = load_harbor_tasks_as_dicts(source_repo, limit=limit)
    print(f"Loaded {len(tasks)} source tasks")

    for t in tasks:
        t[variant_col] = variant_value
    dataset = Dataset.from_list(tasks)

    result = run_completions(
        dataset, model=model, map_type="chat",
        map_config={"user_message": broken_code_prompt, "output_column": "broken_code"},
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"Generated {len(result_dataset)} broken code completions")

    result2 = run_completions(
        result_dataset, model=model, map_type="chat",
        map_config={"user_message": instruction_prompt, "output_column": "new_instruction"},
        require_all_responses=False,
    )
    result_dataset = result2.dataset
    print(f"Generated {len(result_dataset)} instruction completions")

    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        test_files = json.loads(row["test_files_json"])
        dockerfile = row["dockerfile"].rstrip() + "\nCOPY starter_code.py /app/starter_code.py\n"

        task_dir = create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["new_instruction"],
            dockerfile=dockerfile,
            test_sh=row["test_sh"],
            test_files=test_files,
            dataset_prefix=prefix,
            metadata={"source": row["task_dir_name"], variant_col: row[variant_col]},
        )
        # Write starter_code.py to environment/ so Dockerfile COPY can find it
        (task_dir / "environment" / "starter_code.py").write_text(
            row["broken_code"], encoding="utf-8"
        )

    if hf_repo:
        print(f"\nUploading to {hf_repo}...")
        repo_url = upload_tasks_to_hf(str(out), hf_repo)
        print(f"Done: {repo_url}")
    return str(out)


def generate_task_stacking(
    compound_prompt: str,
    test_prompt: str,
    prefix: str,
    hf_repo: str,
    source_repo: str = DEFAULT_SOURCE_REPO,
    limit: int = DEFAULT_LIMIT,
    model: str = DEFAULT_MODEL,
):
    """Category C (8.9): Pair consecutive tasks into compound tasks."""
    tasks = load_harbor_tasks_as_dicts(source_repo, limit=limit)
    print(f"Loaded {len(tasks)} source tasks")

    grouped = []
    for i in range(0, len(tasks) - 1, 2):
        grouped.append({
            "instruction_1": tasks[i]["instruction"],
            "instruction_2": tasks[i + 1]["instruction"],
            "test_files_1": tasks[i]["test_files_json"],
            "test_files_2": tasks[i + 1]["test_files_json"],
            "dockerfile": tasks[i]["dockerfile"],
            "test_sh": tasks[i]["test_sh"],
            "source_1": tasks[i]["task_dir_name"],
            "source_2": tasks[i + 1]["task_dir_name"],
        })
    print(f"Grouped into {len(grouped)} pairs")

    dataset = Dataset.from_list(grouped)

    result = run_completions(
        dataset, model=model, map_type="chat",
        map_config={"user_message": compound_prompt, "output_column": "new_instruction"},
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"Generated {len(result_dataset)} compound instructions")

    result2 = run_completions(
        result_dataset, model=model, map_type="chat",
        map_config={"user_message": test_prompt, "output_column": "combined_tests"},
        require_all_responses=False,
    )
    result_dataset = result2.dataset
    print(f"Generated {len(result_dataset)} combined test suites")

    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        test_files = {"test_combined.py": _clean_llm_test_code(row["combined_tests"])}
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["new_instruction"],
            dockerfile=row["dockerfile"],
            test_sh=_fix_test_sh_filename(row["test_sh"], "test_combined.py"),
            test_files=test_files,
            dataset_prefix=prefix,
            metadata={"source_1": row["source_1"], "source_2": row["source_2"]},
        )

    print(f"\nUploading to {hf_repo}...")
    repo_url = upload_tasks_to_hf(str(out), hf_repo)
    print(f"Done: {repo_url}")
    return str(out)
