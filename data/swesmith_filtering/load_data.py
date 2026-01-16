#!/usr/bin/env python3
"""
Shared data loading for SWE-smith filtering
"""

import tempfile
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset

from data.gcs_cache import gcs_cache
from data.commons import upload_tasks_to_hf
from data.swesmith.generate_with_plain_docker import create_sandboxed_task


@gcs_cache()
def load_swesmith_data() -> Tuple[List[str], List[dict]]:
    """Load SWE-smith problem statements and rows"""
    print("Loading SWE-smith dataset...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")
    print(f"Total rows in dataset: {len(ds)}")

    seen = set()
    questions = []
    rows = []
    for row in ds:
        problem = row.get("problem_statement", "")
        patch = row.get("patch", "")
        if not problem or not patch:
            continue
        if problem not in seen:
            seen.add(problem)
            questions.append(problem)
            rows.append(row)

    print(f"Loaded {len(questions)} unique SWE-smith problem statements")
    return questions, rows


def filter_and_upload_swesmith(
    questions: List[str],
    rows: List[dict],
    filtered_questions: List[str],
    repo_id: str,
    max_tasks: int = 10_000
) -> None:
    """Filter rows based on filtered questions and upload to HuggingFace"""
    question_to_row = {q: r for q, r in zip(questions, rows)}

    seen = set()
    filtered_rows = []
    for q in filtered_questions:
        if q not in seen:
            seen.add(q)
            filtered_rows.append(question_to_row[q])
        if len(filtered_rows) >= max_tasks:
            break

    print(f"Selected {len(filtered_rows)} unique tasks after filtering")

    out_root = Path(tempfile.mkdtemp())
    for idx, row in enumerate(filtered_rows):
        create_sandboxed_task(row, out_root, idx)

    upload_tasks_to_hf(str(out_root), repo_id)
