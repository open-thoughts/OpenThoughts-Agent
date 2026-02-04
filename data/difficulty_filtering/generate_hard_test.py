#!/usr/bin/env python3
"""python -m data.difficulty_filtering.generate_hard_test"""
import shutil
import logging
from pathlib import Path

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from data.commons import upload_tasks_to_hf, extract_hdf5_to_task_paths, generate_tasks_to_hdf5
from data.r2egym.utils import load_r2egym_instances, add_r2egym_verifier_tests_to_instances
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType

from .utils import compute_pass_at_k, filter_tasks, copy_filtered_tasks

DATASET = "r2egym-test"
HF_ORG = "DCAgent"
MODEL = "openai/gpt-5-nano"
AGENT = "terminus-2"
N_ATTEMPTS = 4
N_CONCURRENT = 64
JOBS_DIR = Path(__file__).parent.parent.parent / "jobs"
SAMPLE_SIZE = 3

MIN_RATE = 0.25
MAX_RATE = 0.50

if __name__ == "__main__":
    # Generate small sample
    print(f"Loading {SAMPLE_SIZE} r2egym instances...", flush=True)
    instances = load_r2egym_instances()[:SAMPLE_SIZE]
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
    print("Running Harbor evaluation...", flush=True)
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

    print("Computing scores...")
    scores = compute_pass_at_k(JOBS_DIR / job_name)
    print(f"Scores: {scores}")

    tasks = filter_tasks(scores, MIN_RATE, MAX_RATE)
    print(f"Filtered {len(tasks)} tasks in range {MIN_RATE}-{MAX_RATE}")

    if not tasks:
        print("No tasks in range")
        exit()

    temp_dir = copy_filtered_tasks(task_dir, tasks)
    repo_id = f"{HF_ORG}/{DATASET}-pass{int(MIN_RATE*100)}_{int(MAX_RATE*100)}"
    upload_tasks_to_hf(str(temp_dir), repo_id)
    shutil.rmtree(temp_dir)
    print(f"Done: {len(tasks)} tasks -> {repo_id}")
