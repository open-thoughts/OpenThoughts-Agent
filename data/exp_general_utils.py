#!/usr/bin/env python3
"""Shared utilities for non-Theme-8 experiment generators (programmatic transforms)."""

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


def make_test_sh(test_cmd: str = "python -m pytest /tests/test_solution.py -v --tb=short") -> str:
    """Create a proper test.sh that checks exit codes via PIPESTATUS."""
    return f"""#!/bin/bash
set -e
mkdir -p /logs/verifier

cleanup() {{
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}}
trap cleanup EXIT

cd /app
{test_cmd} 2>&1 | tee /logs/verifier/output.txt

TEST_EXIT=${{PIPESTATUS[0]}}
if [ $TEST_EXIT -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed."
    exit 1
fi
"""


def transform_tasks(
    transform_fn: Callable[[Dict], Dict],
    prefix: str,
    hf_repo: str,
    source_repo: str = DEFAULT_SOURCE_REPO,
    limit: int = DEFAULT_LIMIT,
):
    """Load tasks, apply transform_fn to each task dict, create dirs, upload.

    transform_fn receives a task dict and returns a modified task dict.
    Must return dict with at least: instruction, dockerfile, test_sh, test_files_json.
    """
    tasks = load_harbor_tasks_as_dicts(source_repo, limit=limit)
    print(f"Loaded {len(tasks)} source tasks")

    transformed = []
    for t in tqdm(tasks, desc="Transforming"):
        result = transform_fn(t)
        if result is not None:
            transformed.append(result)
    print(f"Kept {len(transformed)} tasks after transform")

    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    for i, row in enumerate(tqdm(transformed, desc="Creating tasks")):
        metadata = row.get("metadata", {"source": row.get("task_dir_name", "")})
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"],
            dockerfile=row["dockerfile"],
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix=prefix,
            metadata=metadata,
            solution_files=row.get("solution_files"),
        )

    print(f"\nUploading to {hf_repo}...")
    repo_url = upload_tasks_to_hf(str(out), hf_repo)
    print(f"Done: {repo_url}")


def filter_tasks(
    filter_fn: Callable[[Dict], bool],
    prefix: str,
    hf_repo: str,
    source_repo: str = DEFAULT_SOURCE_REPO,
    limit: int = DEFAULT_LIMIT,
):
    """Load tasks, keep only those passing filter_fn, create dirs, upload."""
    tasks = load_harbor_tasks_as_dicts(source_repo, limit=limit)
    print(f"Loaded {len(tasks)} source tasks")

    kept = [t for t in tqdm(tasks, desc="Filtering") if filter_fn(t)]
    print(f"Kept {len(kept)} / {len(tasks)} tasks")

    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    for i, row in enumerate(tqdm(kept, desc="Creating tasks")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"],
            dockerfile=row["dockerfile"],
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix=prefix,
            metadata={"source": row["task_dir_name"]},
        )

    print(f"\nUploading to {hf_repo}...")
    repo_url = upload_tasks_to_hf(str(out), hf_repo)
    print(f"Done: {repo_url}")


def mix_datasets(
    source_repos_with_weights: List[Tuple[str, float]],
    total_limit: int,
    prefix: str,
    hf_repo: str,
):
    """Load tasks from multiple repos with weights, sample, create dirs, upload."""
    import random
    all_tasks = []
    for repo, weight in source_repos_with_weights:
        n = int(total_limit * weight)
        tasks = load_harbor_tasks_as_dicts(repo, limit=n)
        random.shuffle(tasks)
        all_tasks.extend(tasks[:n])
        print(f"Sampled {min(n, len(tasks))} from {repo} (weight={weight})")

    random.shuffle(all_tasks)
    all_tasks = all_tasks[:total_limit]
    print(f"Total mixed dataset: {len(all_tasks)} tasks")

    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    for i, row in enumerate(tqdm(all_tasks, desc="Creating tasks")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"],
            dockerfile=row["dockerfile"],
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix=prefix,
            metadata={"source": row["task_dir_name"]},
        )

    print(f"\nUploading to {hf_repo}...")
    repo_url = upload_tasks_to_hf(str(out), hf_repo)
    print(f"Done: {repo_url}")


def llm_score_and_filter(
    score_prompt: str,
    score_column: str,
    filter_fn: Callable[[Dict], bool],
    prefix: str,
    hf_repo: str,
    source_repo: str = DEFAULT_SOURCE_REPO,
    limit: int = DEFAULT_LIMIT,
    model: str = DEFAULT_MODEL,
):
    """Score tasks with LLM, filter based on scores, create dirs, upload."""
    tasks = load_harbor_tasks_as_dicts(source_repo, limit=limit)
    print(f"Loaded {len(tasks)} source tasks")

    dataset = Dataset.from_list(tasks)
    result = run_completions(
        dataset, model=model, map_type="chat",
        map_config={"user_message": score_prompt, "output_column": score_column},
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"Scored {len(result_dataset)} tasks")

    kept = [row for row in tqdm(result_dataset, desc="Filtering") if filter_fn(row)]
    print(f"Kept {len(kept)} / {len(result_dataset)} tasks")

    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    for i, row in enumerate(tqdm(kept, desc="Creating tasks")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"],
            dockerfile=row["dockerfile"],
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix=prefix,
            metadata={"source": row["task_dir_name"], score_column: row[score_column]},
        )

    print(f"\nUploading to {hf_repo}...")
    repo_url = upload_tasks_to_hf(str(out), hf_repo)
    print(f"Done: {repo_url}")


def llm_generate_and_create(
    generation_prompt: str,
    output_column: str,
    build_task_fn: Callable[[Dict], Dict],
    prefix: str,
    hf_repo: str,
    input_dataset: Dataset = None,
    source_repo: str = DEFAULT_SOURCE_REPO,
    limit: int = DEFAULT_LIMIT,
    model: str = DEFAULT_MODEL,
):
    """Run LLM generation on dataset, then build tasks from results.

    build_task_fn receives a row dict and returns a dict with:
    instruction, dockerfile, test_sh, test_files_json, metadata
    """
    if input_dataset is None:
        tasks = load_harbor_tasks_as_dicts(source_repo, limit=limit)
        print(f"Loaded {len(tasks)} source tasks")
        input_dataset = Dataset.from_list(tasks)

    result = run_completions(
        input_dataset, model=model, map_type="chat",
        map_config={"user_message": generation_prompt, "output_column": output_column},
        require_all_responses=False,
    )
    result_dataset = result.dataset
    print(f"Generated {len(result_dataset)} completions")

    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    created = 0
    for i, row in enumerate(tqdm(result_dataset, desc="Creating tasks")):
        task = build_task_fn(row)
        if task is None:
            continue
        create_harbor_task_directory_generic(
            output_dir=out, task_id=created,
            instruction=task["instruction"],
            dockerfile=task["dockerfile"],
            test_sh=task["test_sh"],
            test_files=json.loads(task["test_files_json"]) if isinstance(task["test_files_json"], str) else task["test_files_json"],
            dataset_prefix=prefix,
            metadata=task.get("metadata", {}),
            solution_files=task.get("solution_files"),
        )
        created += 1

    print(f"Created {created} tasks")
    print(f"\nUploading to {hf_repo}...")
    repo_url = upload_tasks_to_hf(str(out), hf_repo)
    print(f"Done: {repo_url}")
