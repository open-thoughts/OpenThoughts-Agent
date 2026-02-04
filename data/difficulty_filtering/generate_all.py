#!/usr/bin/env python3
"""
Generate r2egym difficulty-filtered datasets from scratch.

python -m data.difficulty_filtering.generate_all
"""
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from data.commons import upload_tasks_to_hf, generate_tasks_from_questions
from data.r2egym.utils import load_r2egym_instances, add_r2egym_verifier_tests_to_instances
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType

# Config
DATASET = "r2egym"
HF_ORG = "DCAgent"
MODEL = "openai/gpt-5-nano"
AGENT = "terminus-2"
N_ATTEMPTS = 8
N_CONCURRENT = 8
JOBS_DIR = Path(__file__).parent.parent.parent / "jobs"

BANDS = [
    (0.0, 0.0),      # impossible
    (0.0, 0.25),     # very_hard
    (0.25, 0.50),    # hard
    (0.50, 0.75),    # medium
    (0.75, 0.99),    # easy
    (0.99, 1.0),     # trivial
]

if __name__ == "__main__":
    # Step 1: Generate tasks
    print("Step 1: Generating tasks")
    instances = load_r2egym_instances()
    questions = add_r2egym_verifier_tests_to_instances(instances)
    task_dir = Path(generate_tasks_from_questions(questions, DATASET))
    print(f"Generated {len(list(task_dir.iterdir()))} tasks")

    # Step 2: Upload to HF
    print("Step 2: Uploading to HuggingFace")
    upload_tasks_to_hf(str(task_dir), f"{HF_ORG}/{DATASET}")

    # Step 3: Run Harbor evaluation
    print("Step 3: Running Harbor evaluation")
    job_name = f"pass_at_{N_ATTEMPTS}_{DATASET}"
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
    job_dir = JOBS_DIR / job_name

    # Step 4: Compute pass@k scores
    print("Step 4: Computing pass@k scores")
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
            if not task_name or data.get("exception_info"):
                continue
            verifier = data.get("verifier_result")
            if verifier:
                reward = verifier.get("rewards", {}).get("reward", 0.0)
                task_results[task_name].append(reward > 0)
        except:
            continue

    scores = {task: sum(r)/len(r) for task, r in task_results.items() if r}
    print(f"Computed scores for {len(scores)} tasks")

    # Step 5: Filter and upload each band
    print("Step 5: Filtering and uploading bands")
    for min_rate, max_rate in BANDS:
        tasks = [t for t, s in scores.items() if (min_rate < s <= max_rate) or (min_rate == 0 and s == 0)]
        if not tasks:
            print(f"  {int(min_rate*100)}-{int(max_rate*100)}%: no tasks")
            continue

        temp_dir = Path(tempfile.mkdtemp())
        for task in tasks:
            src = task_dir / task
            if src.exists():
                shutil.copytree(src, temp_dir / task)

        repo_id = f"{HF_ORG}/{DATASET}-pass{int(min_rate*100)}_{int(max_rate*100)}"
        upload_tasks_to_hf(str(temp_dir), repo_id)
        shutil.rmtree(temp_dir)
        print(f"  {int(min_rate*100)}-{int(max_rate*100)}%: {len(tasks)} tasks -> {repo_id}")

    print("Done!")
