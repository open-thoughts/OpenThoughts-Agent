#!/usr/bin/env python3
"""python -m data.difficulty_filtering.generate_very_hard"""
import shutil
from pathlib import Path

from data.commons import upload_tasks_to_hf, extract_hdf5_to_task_paths
from scripts.harbor.run_and_export_traces import run_dataset_to_traces_hdf5
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType

from .utils import generate_r2egym_hdf5, compute_pass_at_k, filter_tasks, copy_filtered_tasks

DATASET = "r2egym"
HF_ORG = "DCAgent"
MODEL = "openai/gpt-5-nano"
AGENT = "terminus-2"
N_ATTEMPTS = 8
N_CONCURRENT = 8
JOBS_DIR = Path(__file__).parent.parent.parent / "jobs"

MIN_RATE = 0.0
MAX_RATE = 0.25

if __name__ == "__main__":
    hdf5_path = generate_r2egym_hdf5()
    task_dir = Path(extract_hdf5_to_task_paths(hdf5_path))

    upload_tasks_to_hf(str(task_dir), f"{HF_ORG}/{DATASET}")

    job_name = f"pass_at_{N_ATTEMPTS}_{DATASET}"
    run_dataset_to_traces_hdf5(
        hdf5_path=hdf5_path,
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

    scores = compute_pass_at_k(JOBS_DIR / job_name)
    tasks = filter_tasks(scores, MIN_RATE, MAX_RATE)
    if not tasks:
        print("No tasks in range")
        exit()

    temp_dir = copy_filtered_tasks(task_dir, tasks)
    repo_id = f"{HF_ORG}/{DATASET}-pass{int(MIN_RATE*100)}_{int(MAX_RATE*100)}"
    upload_tasks_to_hf(str(temp_dir), repo_id)
    shutil.rmtree(temp_dir)
    print(f"Done: {len(tasks)} tasks -> {repo_id}")
