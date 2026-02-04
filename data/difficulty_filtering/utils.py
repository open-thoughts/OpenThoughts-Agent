#!/usr/bin/env python3
"""Shared utility functions for difficulty filtering."""
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from data.gcs_cache import gcs_cache


@gcs_cache()
def generate_r2egym_hdf5() -> str:
    """Generate r2egym tasks as HDF5 file."""
    from data.r2egym.utils import load_r2egym_instances, add_r2egym_verifier_tests_to_instances
    from data.commons import generate_tasks_to_hdf5

    instances = load_r2egym_instances()
    questions = add_r2egym_verifier_tests_to_instances(instances)

    # Unpack tuples into separate lists
    instructions = [q[0] for q in questions]
    metadata = [q[1] for q in questions]
    solutions = [q[2] for q in questions]
    test_sh = [q[3] for q in questions]
    test_py = [q[4] for q in questions]
    task_toml = [q[5] for q in questions]
    dockerfiles = [q[6] for q in questions]

    return generate_tasks_to_hdf5(
        instructions=instructions,
        metadata=metadata,
        solutions=solutions,
        test_sh=test_sh,
        test_py=test_py,
        task_toml=task_toml,
        dockerfiles=dockerfiles,
        dataset_prefix="r2egym",
    )


def compute_pass_at_k(job_dir: Path) -> dict:
    """Compute pass@k scores from Harbor job results (timeouts count as failures)."""
    task_results = defaultdict(list)
    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text())
            task_name = data.get("task_name")
            if not task_name:
                continue
            # Timeouts/exceptions count as failures (reward=0)
            if data.get("exception_info"):
                task_results[task_name].append(False)
                continue
            verifier = data.get("verifier_result")
            if verifier:
                reward = verifier.get("rewards", {}).get("reward", 0.0)
                task_results[task_name].append(reward > 0)
            else:
                task_results[task_name].append(False)
        except:
            continue
    return {task: sum(r)/len(r) for task, r in task_results.items() if r}


def filter_tasks(scores: dict, min_rate: float, max_rate: float) -> list:
    """Filter tasks by pass rate range."""
    if min_rate == 0 and max_rate == 0:
        return [t for t, s in scores.items() if s == 0]
    return [t for t, s in scores.items() if min_rate < s <= max_rate]


def copy_filtered_tasks(task_dir: Path, tasks: list) -> Path:
    """Copy filtered tasks to a temp directory."""
    temp_dir = Path(tempfile.mkdtemp())
    for task in tasks:
        src = task_dir / task
        if src.exists():
            shutil.copytree(src, temp_dir / task)
    return temp_dir
