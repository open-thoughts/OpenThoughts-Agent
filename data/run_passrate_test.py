#!/usr/bin/env python3
"""
Test pass rate for Harbor datasets with gpt-4.1-nano.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root and scripts/harbor to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "harbor"))

# Disable progress bars
os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'


def run_test(dataset_repo_id: str, n_tasks: int = 25, model: str = "openai/gpt-4.1-nano"):
    """Run pass rate test for a single dataset."""
    dataset_name = dataset_repo_id.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_repo_id}")
    print(f"Tasks: {n_tasks}, Model: {model}")
    print(f"{'='*60}")

    # Download dataset
    print("\n=== Downloading dataset ===")
    from tasks_parquet_converter import from_hf_dataset
    dataset_path = from_hf_dataset(dataset_repo_id)
    print(f"Dataset path: {dataset_path}")

    # Find task directories
    task_dirs = sorted(Path(dataset_path).glob("*/instruction.md"))
    total_tasks = len(task_dirs)
    print(f"Total tasks available: {total_tasks}")

    if total_tasks == 0:
        print("ERROR: No tasks found!")
        return {"dataset": dataset_repo_id, "error": "No tasks found"}

    # Create subset directory by copying (symlinks cause recursion issues with Harbor)
    import shutil
    n_tasks = min(n_tasks, total_tasks)
    subset_dir = tempfile.mkdtemp(prefix=f"{dataset_name}_subset_")

    for task_md in task_dirs[:n_tasks]:
        task_dir = task_md.parent
        target = Path(subset_dir) / task_dir.name
        shutil.copytree(task_dir, target)

    print(f"Created subset with {n_tasks} tasks at: {subset_dir}")

    # Run Harbor
    print("\n=== Running Harbor ===")
    job_name = f"passrate_{dataset_name}_{os.getpid()}"
    config_path = Path(__file__).parent.parent / "eval/tacc/dcagent_eval_config.yaml"

    cmd = [
        "harbor", "jobs", "start",
        "-p", subset_dir,
        "--n-concurrent", "8",
        "--agent", "terminus-2",
        "--model", model,
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "0",
        "--job-name", job_name,
        "--config", str(config_path),
    ]

    print(f"Command: {' '.join(cmd)}")

    # Change to dc-agent directory
    os.chdir(Path(__file__).parent.parent)

    result = subprocess.run(cmd, capture_output=False)

    # Parse results
    print("\n=== Results ===")
    result_file = Path("jobs") / job_name / "result.json"

    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)

        results = data.get("results", [])
        passed = sum(1 for r in results if r.get("reward", 0) > 0)
        total = len(results)
        rate = (passed / total * 100) if total > 0 else 0

        print(f"Dataset: {dataset_repo_id}")
        print(f"Tasks tested: {total}")
        print(f"Passed: {passed}")
        print(f"Pass rate: {rate:.1f}%")

        return {
            "dataset": dataset_repo_id,
            "passed": passed,
            "total": total,
            "pass_rate": rate
        }
    else:
        print(f"Could not find results at {result_file}")
        return {"dataset": dataset_repo_id, "error": "No results file"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", help="Dataset repo IDs to test")
    parser.add_argument("--n-tasks", type=int, default=25)
    parser.add_argument("--model", default="openai/gpt-4.1-nano")
    args = parser.parse_args()

    results = []
    for ds in args.datasets:
        try:
            result = run_test(ds, args.n_tasks, args.model)
            results.append(result)
        except Exception as e:
            print(f"ERROR testing {ds}: {e}")
            results.append({"dataset": ds, "error": str(e)})

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        if "error" in r:
            print(f"❌ {r['dataset']}: {r['error']}")
        else:
            print(f"✅ {r['dataset']}: {r['pass_rate']:.1f}% ({r['passed']}/{r['total']})")


if __name__ == "__main__":
    main()
