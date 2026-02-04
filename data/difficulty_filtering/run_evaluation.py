#!/usr/bin/env python3
"""
Full pipeline: Generate tasks -> Upload to HF -> Run Harbor evaluation

python -m data.difficulty_filtering.run_evaluation
"""
from data.commons import upload_tasks_to_hf
from .utils import (
    DATASET, HF_ORG, N_ATTEMPTS,
    generate_r2egym_tasks,
    run_harbor_evaluation,
)

if __name__ == "__main__":
    # Step 1: Generate r2egym tasks
    print("=" * 60)
    print("Step 1: Generating r2egym tasks")
    print("=" * 60)
    task_dir = generate_r2egym_tasks()

    # Step 2: Upload to HuggingFace
    print("=" * 60)
    print("Step 2: Uploading to HuggingFace")
    print("=" * 60)
    repo_id = f"{HF_ORG}/{DATASET}"
    url = upload_tasks_to_hf(str(task_dir), repo_id)
    print(f"Uploaded to: {url}")

    # Step 3: Run Harbor evaluation
    print("=" * 60)
    print("Step 3: Running Harbor evaluation")
    print("=" * 60)
    job_dir = run_harbor_evaluation(DATASET, HF_ORG, N_ATTEMPTS)
    print(f"Evaluation complete: {job_dir}")

    print("=" * 60)
    print("Done! Now run generate_*.py scripts to filter by difficulty")
    print("=" * 60)
