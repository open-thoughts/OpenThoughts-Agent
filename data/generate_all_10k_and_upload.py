#!/usr/bin/env python3
"""
Generate 10k tasks for each dataset and upload to HuggingFace.

This script:
1. Runs all existing generator scripts with limit=10000
2. Uploads each generated dataset to HuggingFace Hub
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import HfApi, create_repo, upload_folder
from tqdm import tqdm


# Dataset configurations: generator script -> HF repo name
DATASETS = {
    # Focal-Test Pair Datasets
    "generate_unitsyn_tasks.py": "DCAgent/exp_rpt_unitsyn-python",
    "generate_methods2test_tasks.py": "DCAgent/exp_rpt_methods2test",
    "generate_pymethods2test_tasks.py": "DCAgent/exp_rpt_pymethods2test",

    # Competitive Programming
    "generate_codenet_tasks.py": "DCAgent/exp_rpt_codenet",
    "generate_taco_tasks.py": "DCAgent/exp_rpt_taco",
    "generate_exercism_tasks.py": "DCAgent/exp_rpt_exercism",
    "generate_codeelo_tasks.py": "DCAgent/exp_rpt_codeelo",

    # Bug Benchmarks
    "generate_defects4j_tasks.py": "DCAgent/exp_rpt_defects4j",
    "generate_bugsinpy_tasks.py": "DCAgent/exp_rpt_bugsinpy",
    "generate_bugsinpy_mf_tasks.py": "DCAgent/exp_rpt_bugsinpy-mf",
    "generate_quixbugs_tasks.py": "DCAgent/exp_rpt_quixbugs",
    "generate_codereval_tasks.py": "DCAgent/exp_rpt_codereval",
    "generate_manybugs_tasks.py": "DCAgent/exp_rpt_manybugs",
    "generate_bugswarm_tasks.py": "DCAgent/exp_rpt_bugswarm",

    # Complex/Multi-File
    "generate_swebench_tasks.py": "DCAgent/exp_rpt_swebench",
    "generate_bigcodebench_tasks.py": "DCAgent/exp_rpt_bigcodebench",
    "generate_e2egit_tasks.py": "DCAgent/exp_rpt_e2egit",
    "generate_crosscodeeval_tasks.py": "DCAgent/exp_rpt_crosscodeeval",

    # CI/Build Mining
    "generate_travistorrent_tasks.py": "DCAgent/exp_rpt_travistorrent",
    "generate_ghactions_tasks.py": "DCAgent/exp_rpt_ghactions",
    "generate_softwareheritage_tasks.py": "DCAgent/exp_rpt_softwareheritage",
}


def upload_to_huggingface(
    task_dir: str,
    repo_id: str,
    token: Optional[str] = None,
) -> str:
    """
    Upload a task directory to HuggingFace Hub.

    Args:
        task_dir: Path to directory containing harbor tasks
        repo_id: HuggingFace repo ID (e.g., "DCAgent/exp_rpt_unitsyn")
        token: HuggingFace API token

    Returns:
        URL of the uploaded repository
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"  Created/verified repo: {repo_id}")
    except Exception as e:
        print(f"  Warning creating repo: {e}")

    # Upload the folder
    print(f"  Uploading {task_dir} to {repo_id}...")
    upload_folder(
        folder_path=task_dir,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    return f"https://huggingface.co/datasets/{repo_id}"


def run_generator(script_name: str, limit: int = 10000) -> Optional[str]:
    """
    Run a generator script and return the output directory.

    Args:
        script_name: Name of the generator script
        limit: Maximum number of tasks to generate

    Returns:
        Path to generated tasks directory, or None if failed
    """
    script_path = Path(__file__).parent / "dclm-mine" / script_name

    if not script_path.exists():
        print(f"  Warning: Script not found: {script_path}")
        return None

    print(f"  Running: {script_name} --limit {limit}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--limit", str(limit)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(script_path.parent),
        )

        if result.returncode != 0:
            print(f"  Error running {script_name}:")
            print(f"    stdout: {result.stdout[-500:] if result.stdout else 'None'}")
            print(f"    stderr: {result.stderr[-500:] if result.stderr else 'None'}")
            return None

        # Try to extract output directory from stdout
        for line in result.stdout.split("\n"):
            if "Output directory:" in line or "Task directory:" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    return parts[1].strip()

        return None

    except subprocess.TimeoutExpired:
        print(f"  Timeout running {script_name}")
        return None
    except Exception as e:
        print(f"  Exception running {script_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate 10k tasks for all datasets and upload to HuggingFace")
    parser.add_argument("--limit", type=int, default=10000, help="Number of tasks to generate per dataset")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to process (default: all)")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, only upload existing tasks")
    parser.add_argument("--task-dirs", type=str, help="JSON file mapping dataset names to task directories")
    args = parser.parse_args()

    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set")
        print("Run: source /scratch/10000/eguha3/old-dc-agent/secret.env")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Generating {args.limit} tasks for each dataset and uploading to HuggingFace")
    print(f"{'='*60}\n")

    # Filter datasets if specified
    datasets_to_process = DATASETS
    if args.datasets:
        datasets_to_process = {
            k: v for k, v in DATASETS.items()
            if any(d in k or d in v for d in args.datasets)
        }

    results = {}

    for script_name, repo_id in tqdm(datasets_to_process.items(), desc="Processing datasets"):
        dataset_name = repo_id.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")

        task_dir = None

        if not args.skip_generation:
            # Run generator
            task_dir = run_generator(script_name, limit=args.limit)
        elif args.task_dirs:
            # Load from pre-existing mapping
            with open(args.task_dirs) as f:
                task_dirs_map = json.load(f)
            task_dir = task_dirs_map.get(dataset_name)

        if task_dir and Path(task_dir).exists():
            try:
                # Upload to HuggingFace
                repo_url = upload_to_huggingface(task_dir, repo_id, hf_token)
                results[dataset_name] = {"status": "success", "url": repo_url, "task_dir": task_dir}
                print(f"  ✓ Uploaded: {repo_url}")
            except Exception as e:
                results[dataset_name] = {"status": "upload_failed", "error": str(e)}
                print(f"  ✗ Upload failed: {e}")
        else:
            results[dataset_name] = {"status": "generation_failed", "task_dir": task_dir}
            print(f"  ✗ No tasks generated")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    success = [k for k, v in results.items() if v.get("status") == "success"]
    failed = [k for k, v in results.items() if v.get("status") != "success"]

    print(f"\nSuccessful uploads ({len(success)}):")
    for name in success:
        print(f"  ✓ {name}: {results[name].get('url')}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name in failed:
            print(f"  ✗ {name}: {results[name].get('status')} - {results[name].get('error', 'no error')}")

    # Save results
    results_file = f"/scratch/10000/eguha3/dc-agent/data/upload_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
