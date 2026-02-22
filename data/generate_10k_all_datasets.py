#!/usr/bin/env python3
"""
Generate exactly 10k tasks for each dataset and upload to HuggingFace.

For each dataset:
1. Run the generator script to create tasks
2. Subsample to 10k if more tasks exist, upsample if fewer
3. Upload to HuggingFace
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
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.commons import (
    subsample_tasks_directory,
    upsample_tasks_directory,
    upload_tasks_to_hf,
)

# Dataset configurations: (script_name, hf_repo_id, has_limit_arg)
DATASETS = {
    # Focal-Test Pair Datasets (highest quality)
    "unitsyn": ("generate_unitsyn_tasks.py", "DCAgent/exp_rpt_unitsyn-python", True),
    "methods2test": ("generate_methods2test_tasks.py", "DCAgent/exp_rpt_methods2test", True),
    "pymethods2test": ("generate_pymethods2test_tasks.py", "DCAgent/exp_rpt_pymethods2test", True),

    # Competitive Programming
    "codenet": ("generate_codenet_tasks.py", "DCAgent/exp_rpt_codenet", True),
    "taco": ("generate_taco_tasks.py", "DCAgent/exp_rpt_taco", True),
    "exercism": ("generate_exercism_tasks.py", "DCAgent/exp_rpt_exercism", True),
    "codeelo": ("generate_codeelo_tasks.py", "DCAgent/exp_rpt_codeelo", True),

    # Bug Benchmarks
    "defects4j": ("generate_defects4j_tasks.py", "DCAgent/exp_rpt_defects4j", True),
    "bugsinpy": ("generate_bugsinpy_tasks.py", "DCAgent/exp_rpt_bugsinpy", True),
    "bugsinpy_mf": ("generate_bugsinpy_mf_tasks.py", "DCAgent/exp_rpt_bugsinpy-mf", True),
    "quixbugs": ("generate_quixbugs_tasks.py", "DCAgent/exp_rpt_quixbugs", True),
    "codereval": ("generate_codereval_tasks.py", "DCAgent/exp_rpt_codereval", True),
    "manybugs": ("generate_manybugs_tasks.py", "DCAgent/exp_rpt_manybugs", True),
    "bugswarm": ("generate_bugswarm_tasks.py", "DCAgent/exp_rpt_bugswarm", True),

    # Complex/Multi-File
    "swebench": ("generate_swebench_tasks.py", "DCAgent/exp_rpt_swebench", True),
    "bigcodebench": ("generate_bigcodebench_tasks.py", "DCAgent/exp_rpt_bigcodebench", True),
    "e2egit": ("generate_e2egit_tasks.py", "DCAgent/exp_rpt_e2egit", True),
    "crosscodeeval": ("generate_crosscodeeval_tasks.py", "DCAgent/exp_rpt_crosscodeeval", True),

    # CI/Build Mining
    "travistorrent": ("generate_travistorrent_tasks.py", "DCAgent/exp_rpt_travistorrent", True),
    "ghactions": ("generate_ghactions_tasks.py", "DCAgent/exp_rpt_ghactions", True),
    "softwareheritage": ("generate_softwareheritage_tasks.py", "DCAgent/exp_rpt_softwareheritage", True),
}

TARGET_TASKS = 10000


def run_generator(script_name: str, limit: int = 20000) -> Optional[str]:
    """
    Run a generator script and return the output directory.
    We generate more than needed so we can subsample.
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
            timeout=7200,  # 2 hour timeout
            cwd=str(script_path.parent),
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        )

        # Try to extract output directory from stdout
        output_dir = None
        for line in result.stdout.split("\n"):
            if "Output directory:" in line or "Task directory:" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    output_dir = parts[1].strip()
                    break
            # Also check for temp directory patterns
            if "/tmp/" in line and "_tasks_" in line:
                for word in line.split():
                    if "/tmp/" in word and "_tasks_" in word:
                        potential_dir = word.strip()
                        if Path(potential_dir).exists():
                            output_dir = potential_dir
                            break

        if result.returncode != 0:
            print(f"  Warning: Script returned non-zero: {result.returncode}")
            if result.stderr:
                print(f"  stderr (last 500 chars): {result.stderr[-500:]}")

        return output_dir

    except subprocess.TimeoutExpired:
        print(f"  Timeout running {script_name}")
        return None
    except Exception as e:
        print(f"  Exception running {script_name}: {e}")
        return None


def normalize_to_10k(task_dir: str, dataset_name: str) -> str:
    """
    Normalize a task directory to exactly 10k tasks.
    Subsample if >10k, upsample if <10k.
    """
    task_path = Path(task_dir)
    task_dirs = [d for d in task_path.iterdir() if d.is_dir()]
    current_count = len(task_dirs)

    print(f"  Current task count: {current_count}")

    if current_count > TARGET_TASKS:
        print(f"  Subsampling from {current_count} to {TARGET_TASKS}...")
        return subsample_tasks_directory(task_dir, TARGET_TASKS, dataset_prefix=f"{dataset_name}_10k")
    elif current_count < TARGET_TASKS:
        print(f"  Upsampling from {current_count} to {TARGET_TASKS}...")
        return upsample_tasks_directory(task_dir, TARGET_TASKS, dataset_prefix=f"{dataset_name}_10k")
    else:
        print(f"  Already exactly {TARGET_TASKS} tasks, no adjustment needed.")
        return task_dir


def process_dataset(dataset_name: str, script_name: str, repo_id: str) -> Dict:
    """Process a single dataset: generate, normalize, upload."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")

    result = {
        "dataset": dataset_name,
        "status": "unknown",
        "repo_id": repo_id,
    }

    # Step 1: Generate tasks (generate extra so we can subsample)
    print(f"\nStep 1: Generating tasks...")
    task_dir = run_generator(script_name, limit=20000)

    if not task_dir or not Path(task_dir).exists():
        result["status"] = "generation_failed"
        result["error"] = "No task directory generated"
        print(f"  ✗ Generation failed")
        return result

    result["original_dir"] = task_dir

    # Count generated tasks
    task_dirs = [d for d in Path(task_dir).iterdir() if d.is_dir()]
    result["original_count"] = len(task_dirs)
    print(f"  ✓ Generated {len(task_dirs)} tasks")

    # Step 2: Normalize to 10k
    print(f"\nStep 2: Normalizing to {TARGET_TASKS} tasks...")
    try:
        normalized_dir = normalize_to_10k(task_dir, dataset_name)
        result["normalized_dir"] = normalized_dir

        # Verify final count
        final_dirs = [d for d in Path(normalized_dir).iterdir() if d.is_dir()]
        result["final_count"] = len(final_dirs)
        print(f"  ✓ Final task count: {len(final_dirs)}")
    except Exception as e:
        result["status"] = "normalization_failed"
        result["error"] = str(e)
        print(f"  ✗ Normalization failed: {e}")
        return result

    # Step 3: Upload to HuggingFace
    print(f"\nStep 3: Uploading to HuggingFace ({repo_id})...")
    try:
        repo_url = upload_tasks_to_hf(normalized_dir, repo_id)
        result["status"] = "success"
        result["url"] = repo_url
        print(f"  ✓ Uploaded: {repo_url}")
    except Exception as e:
        result["status"] = "upload_failed"
        result["error"] = str(e)
        print(f"  ✗ Upload failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate 10k tasks for all datasets and upload to HuggingFace")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to process (default: all)")
    parser.add_argument("--skip", nargs="+", default=[], help="Datasets to skip")
    args = parser.parse_args()

    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set")
        print("Run: source /scratch/10000/eguha3/old-dc-agent/secret.env")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Generating {TARGET_TASKS} tasks for each dataset")
    print(f"{'='*60}")

    # Filter datasets
    datasets_to_process = DATASETS
    if args.datasets:
        datasets_to_process = {k: v for k, v in DATASETS.items() if k in args.datasets}
    if args.skip:
        datasets_to_process = {k: v for k, v in datasets_to_process.items() if k not in args.skip}

    print(f"Processing {len(datasets_to_process)} datasets: {list(datasets_to_process.keys())}")

    results = []
    for dataset_name, (script_name, repo_id, _) in datasets_to_process.items():
        result = process_dataset(dataset_name, script_name, repo_id)
        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    print(f"\nSuccessful ({len(success)}/{len(results)}):")
    for r in success:
        print(f"  ✓ {r['dataset']}: {r.get('final_count', '?')} tasks -> {r.get('url', 'N/A')}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  ✗ {r['dataset']}: {r.get('status')} - {r.get('error', 'unknown')}")

    # Save results
    results_file = f"/scratch/10000/eguha3/dc-agent/data/generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
