"""
Utilities for Nemotron-Terminal-Corpus task extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def extract_task_description(first_user_message: str) -> str | None:
    """Extract the task description from the first user message.

    The message format is:
        [system prompt / JSON format instructions]
        ...
        Task Description:
        [actual task instruction]

        Current terminal state:
        Current Terminal Screen:
        root@host:/app#

    Returns None if no task description is found.
    """
    # Split on "Task Description:\n" — take everything after it
    marker = "Task Description:\n"
    idx = first_user_message.find(marker)
    if idx == -1:
        # Fallback: try "Task Description:" without newline
        marker = "Task Description:"
        idx = first_user_message.find(marker)
        if idx == -1:
            return None

    after = first_user_message[idx + len(marker):]

    # Trim at "Current terminal state:" or "Current Terminal Screen:"
    for end_marker in ("Current terminal state:", "Current Terminal Screen:"):
        end_idx = after.find(end_marker)
        if end_idx != -1:
            after = after[:end_idx]
            break

    result = after.strip()
    return result if result else None


def iter_parquet_rows(parquet_path: Path, batch_size: int = 256) -> Iterator[dict]:
    """Yield rows from a parquet file using iter_batches.

    Uses iter_batches instead of read_row_group to avoid
    ArrowNotImplementedError on large nested-struct parquets.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        cols = batch.schema.names
        for i in range(batch.num_rows):
            yield {col: batch.column(col)[i].as_py() for col in cols}


# ---------------------------------------------------------------------------
# Subset configuration: maps HF repo suffix → source parquet glob + metadata
# ---------------------------------------------------------------------------

SUBSET_CONFIGS: dict[str, dict] = {
    "adapters-code": {
        "parquets": ["dataset_adapters/code.parquet"],
        "difficulty": "medium",
        "category": "code",
    },
    "adapters-math": {
        "parquets": ["dataset_adapters/math.parquet"],
        "difficulty": "medium",
        "category": "math",
    },
    "adapters-swe": {
        "parquets": ["dataset_adapters/swe.parquet"],
        "difficulty": "hard",
        "category": "software-engineering",
    },
    "skill-easy": {
        "parquets": [
            "synthetic_tasks/skill_based/easy/data_processing/data_filtered.parquet",
            "synthetic_tasks/skill_based/easy/data_querying/data_filtered.parquet",
            "synthetic_tasks/skill_based/easy/data_science/data_filtered.parquet",
            "synthetic_tasks/skill_based/easy/debugging/data_filtered.parquet",
            "synthetic_tasks/skill_based/easy/dependency_management/data_filtered.parquet",
            "synthetic_tasks/skill_based/easy/file_operations/data_filtered.parquet",
            "synthetic_tasks/skill_based/easy/scientific_computing/data_filtered.parquet",
            "synthetic_tasks/skill_based/easy/security/data_filtered.parquet",
            "synthetic_tasks/skill_based/easy/software_engineering/data_filtered.parquet",
        ],
        "difficulty": "easy",
        "category": "terminal",
    },
    "skill-medium": {
        "parquets": [
            "synthetic_tasks/skill_based/medium/data_processing/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/data_querying/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/data_science/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/debugging/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/dependency_management/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/file_operations/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/model_training/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/scientific_computing/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/security/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/software_engineering/data_filtered.parquet",
            "synthetic_tasks/skill_based/medium/system_administration/data_filtered.parquet",
        ],
        "difficulty": "medium",
        "category": "terminal",
    },
    "skill-mixed": {
        "parquets": [
            "synthetic_tasks/skill_based/mixed/data_processing/data_filtered.parquet",
            "synthetic_tasks/skill_based/mixed/data_science/data_filtered.parquet",
            "synthetic_tasks/skill_based/mixed/debugging/data_filtered.parquet",
            "synthetic_tasks/skill_based/mixed/file_operations/data_filtered.parquet",
            "synthetic_tasks/skill_based/mixed/scientific_computing/data_filtered.parquet",
            "synthetic_tasks/skill_based/mixed/security/data_filtered.parquet",
        ],
        "difficulty": "medium",
        "category": "terminal",
    },
}


def get_category_from_parquet_path(parquet_path: str) -> str:
    """Extract the skill category from a skill_based parquet path.

    E.g. 'synthetic_tasks/skill_based/easy/debugging/data_filtered.parquet'
    → 'debugging'
    """
    parts = Path(parquet_path).parts
    # The category is the parent directory of data_filtered.parquet
    if len(parts) >= 2:
        return parts[-2].replace("_", "-")
    return "terminal"
