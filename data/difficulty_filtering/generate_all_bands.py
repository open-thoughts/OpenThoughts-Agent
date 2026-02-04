#!/usr/bin/env python3
"""
Generate filtered datasets for all difficulty bands based on pass@k rates.

Usage:
    python -m data.difficulty_filtering.generate_all_bands
"""
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from data.commons import upload_tasks_to_hf, extract_hdf5_to_task_paths, generate_tasks_to_hdf5
from data.r2egym.utils import load_r2egym_instances, add_r2egym_verifier_tests_to_instances
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType

from .utils import compute_pass_at_k, filter_tasks, copy_filtered_tasks

# Configuration
DATASET = "r2egym"
HF_ORG = "DCAgent"
HF_PREFIX = "exp-rdb-"
MODEL = "openai/gpt-5-nano"
AGENT = "terminus-2"
N_ATTEMPTS = 8
N_CONCURRENT = 512
JOBS_DIR = Path(__file__).parent.parent.parent / "jobs"

# Difficulty bands: (name, min_rate, max_rate, description)
# Using (min, max] ranges, except for impossible (exactly 0) and trivial (exactly 1)
DIFFICULTY_BANDS = [
    ("impossible", 0.0, 0.0, "Never passes - broken or too hard"),
    ("very_hard", 0.0, 0.25, "Rarely passes (0-25%]"),
    ("hard", 0.25, 0.50, "Passes occasionally (25-50%]"),
    ("medium", 0.50, 0.75, "Usually passes (50-75%]"),
    ("easy", 0.75, 1.0, "Almost always passes (75-100%)"),
    ("trivial", 1.0, 1.0, "Always passes - no learning signal"),
]


def filter_by_band(scores: dict, band_name: str, min_rate: float, max_rate: float) -> list:
    """Filter tasks by difficulty band."""
    if band_name == "impossible":
        # Exactly 0%
        return [t for t, s in scores.items() if s == 0.0]
    elif band_name == "trivial":
        # Exactly 100%
        return [t for t, s in scores.items() if s == 1.0]
    elif band_name == "easy":
        # (75%, 100%) - excludes 100%
        return [t for t, s in scores.items() if min_rate < s < max_rate]
    else:
        # (min, max] - exclusive min, inclusive max
        return [t for t, s in scores.items() if min_rate < s <= max_rate]


if __name__ == "__main__":
    # Load all r2egym instances
    print(f"Loading r2egym instances...", flush=True)
    instances = load_r2egym_instances()
    print(f"Loaded {len(instances)} instances", flush=True)

    questions = add_r2egym_verifier_tests_to_instances(instances)

    instructions = [q[0] for q in questions]
    metadata = [q[1] for q in questions]
    solutions = [q[2] for q in questions]
    test_sh = [q[3] for q in questions]
    test_py = [q[4] for q in questions]
    task_toml = [q[5] for q in questions]
    dockerfiles = [q[6] for q in questions]

    print("Generating HDF5...", flush=True)
    hdf5_path = generate_tasks_to_hdf5(
        instructions=instructions,
        metadata=metadata,
        solutions=solutions,
        test_sh=test_sh,
        test_py=test_py,
        task_toml=task_toml,
        dockerfiles=dockerfiles,
        dataset_prefix=DATASET,
    )

    print("Extracting tasks...", flush=True)
    task_dir = Path(extract_hdf5_to_task_paths(hdf5_path))
    print(f"Task dir: {task_dir}", flush=True)

    job_name = f"pass_at_{N_ATTEMPTS}_{DATASET}"
    print(f"Running Harbor evaluation...", flush=True)
    print(f"  job_name={job_name}", flush=True)
    print(f"  n_attempts={N_ATTEMPTS}", flush=True)
    print(f"  n_concurrent={N_CONCURRENT}", flush=True)
    print(f"  model={MODEL}", flush=True)
    print(f"  agent={AGENT}", flush=True)

    run_dataset_to_traces(
        dataset_path=task_dir,
        job_name=job_name,
        jobs_dir=JOBS_DIR,
        n_attempts=N_ATTEMPTS,
        n_concurrent=N_CONCURRENT,
        agent_name=AgentName(AGENT),
        model_name=MODEL,
        env_type=EnvironmentType("daytona"),
        force_build=True,
        delete_env=True,
        quiet=False,
    )

    print("Computing pass@k scores...", flush=True)
    scores = compute_pass_at_k(JOBS_DIR / job_name)
    print(f"Computed scores for {len(scores)} tasks", flush=True)

    # Print score distribution
    print("\nScore distribution:", flush=True)
    for task, score in sorted(scores.items(), key=lambda x: x[1]):
        print(f"  {task}: {score:.2%}", flush=True)

    # Filter and upload each band
    print("\nFiltering and uploading by difficulty band...", flush=True)
    for band_name, min_rate, max_rate, description in DIFFICULTY_BANDS:
        tasks = filter_by_band(scores, band_name, min_rate, max_rate)
        print(f"\n{band_name}: {len(tasks)} tasks ({description})", flush=True)

        if not tasks:
            print(f"  Skipping - no tasks in this band", flush=True)
            continue

        # Copy filtered tasks
        temp_dir = copy_filtered_tasks(task_dir, tasks)

        # Upload to HuggingFace
        repo_id = f"{HF_ORG}/{HF_PREFIX}{DATASET}-{band_name}"
        print(f"  Uploading to {repo_id}...", flush=True)
        upload_tasks_to_hf(str(temp_dir), repo_id)

        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"  Done: {repo_id}", flush=True)

    print("\nAll bands processed!", flush=True)
