#!/usr/bin/env python3
"""
Generate '-large' (5000-task) versions of 14 datasets and upload to HuggingFace.

For datasets with >500 source tasks, scales up to 5000 (or source max).
Uploads as DCAgent/exp_rpt_<name>-large.

Usage:
    # Generate and upload all 14 datasets (sequential)
    python data/generate_large_datasets.py

    # Generate specific datasets
    python data/generate_large_datasets.py --datasets pytest jest dockerfile

    # Dry run (generate only, no upload)
    python data/generate_large_datasets.py --no-upload

    # Run a specific batch
    python data/generate_large_datasets.py --batch 1
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
GENERATORS_DIR = PROJECT_ROOT / "data" / "dclm-mine"
HF_ORG = "DCAgent"
SECRET_ENV = Path("/scratch/10000/eguha3/old-dc-agent/secret.env")

# ─── Dataset Configurations ────────────────────────────────────
# (short_name, generator_script, target_limit, hf_suffix)

DATASETS: List[Tuple[str, str, int, str]] = [
    # Tier 1: Can scale to 5000
    ("pytest",          "generate_pytest_tasks.py",           5000, "stack-pytest-large"),
    ("jest",            "generate_jest_tasks.py",             5000, "stack-jest-large"),
    ("dockerfile",      "generate_dockerfile_tasks.py",       5000, "stack-dockerfile-large"),
    ("selfdoc",         "generate_self_documented_tasks.py",  5000, "stack-selfdoc-large"),
    ("php",             "generate_phpunit_tasks.py",          5000, "stack-php-large"),
    ("pymethods2test",  "generate_pymethods2test_tasks.py",   5000, "pymethods2test-large"),
    ("unitsyn",         "generate_unitsyn_tasks.py",          5000, "unitsyn-python-large"),
    ("methods2test",    "generate_methods2test_tasks.py",     5000, "methods2test-large"),
    ("e2egit",          "generate_e2egit_tasks.py",           5000, "e2egit-large"),
    ("softwareheritage","generate_softwareheritage_tasks.py", 5000, "softwareheritage-large"),
    ("exercism",        "generate_exercism_tasks.py",         5000, "exercism-python-large"),
    # Tier 2: Source-limited but >500 available
    ("bugsinpy",        "generate_bugsinpy_tasks.py",         5000, "bugsinpy-large"),
    ("bigcodebench",    "generate_bigcodebench_tasks.py",     5000, "bigcodebench-large"),
    ("bugsinpy-mf",     "generate_bugsinpy_mf_tasks.py",      5000, "bugsinpy-mf-large"),
]

# Batches for parallel execution
BATCHES = {
    1: ["pytest", "jest", "dockerfile", "selfdoc", "php"],
    2: ["pymethods2test", "unitsyn", "methods2test", "e2egit", "softwareheritage"],
    3: ["exercism", "bugsinpy", "bigcodebench", "bugsinpy-mf"],
}


def run_generator(
    script_name: str,
    limit: int,
    extra_args: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Run a generator script with SKIP_HF_UPLOAD=1 and return the output directory.

    Parses 'Output directory: <path>' from stdout.
    """
    script_path = GENERATORS_DIR / script_name
    if not script_path.exists():
        print(f"  [ERROR] Script not found: {script_path}")
        return None

    cmd = [sys.executable, str(script_path), "--limit", str(limit)]

    # Add --no-upload for generators that support it
    generators_with_no_upload = {
        "generate_pymethods2test_tasks.py",
        "generate_unitsyn_tasks.py",
        "generate_e2egit_tasks.py",
        "generate_exercism_tasks.py",
        "generate_bugsinpy_tasks.py",
        "generate_bigcodebench_tasks.py",
        "generate_bugsinpy_mf_tasks.py",
    }
    if script_name in generators_with_no_upload:
        cmd.append("--no-upload")

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["SKIP_HF_UPLOAD"] = "1"
    env["CURATOR_DISABLE_CACHE"] = "true"

    # Source secret.env to get HF_TOKEN and other API keys
    if SECRET_ENV.exists():
        for line in SECRET_ENV.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                env[key] = val

    print(f"  Running: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
            env=env,
            cwd=str(GENERATORS_DIR),
        )

        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.0f}s (exit code {result.returncode})")

        if result.returncode != 0:
            print(f"  [ERROR] Generator failed:")
            # Print last 1000 chars of stderr
            if result.stderr:
                print(f"  stderr (last 1000 chars): {result.stderr[-1000:]}")
            if result.stdout:
                print(f"  stdout (last 500 chars): {result.stdout[-500:]}")
            return None

        # Parse output directory from stdout
        task_dir = _extract_output_dir(result.stdout)
        if task_dir:
            print(f"  Output directory: {task_dir}")
            return task_dir

        # Fallback: look for SKIP_HF_UPLOAD message which also prints the dir
        task_dir = _extract_skip_upload_dir(result.stdout)
        if task_dir:
            print(f"  Output directory (from skip msg): {task_dir}")
            return task_dir

        print(f"  [WARNING] Could not extract output directory from stdout")
        if result.stdout:
            # Print last lines to help debug
            lines = result.stdout.strip().split("\n")
            for line in lines[-20:]:
                print(f"    | {line}")
        return None

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timed out after 2 hours")
        return None
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        return None


def _extract_output_dir(stdout: str) -> Optional[str]:
    """Extract output directory path from generator stdout."""
    patterns = [
        r"Output directory:\s*(.+)",
        r"Task directory:\s*(.+)",
        r"Generating harbor tasks in:\s*(.+)",
    ]
    # Search from the end of output (most recent mention)
    for line in reversed(stdout.strip().split("\n")):
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                path = match.group(1).strip()
                if Path(path).exists():
                    return path
    return None


def _extract_skip_upload_dir(stdout: str) -> Optional[str]:
    """Extract directory from SKIP_HF_UPLOAD message."""
    pattern = r"\[SKIP_HF_UPLOAD\].*Task directory:\s*(.+)"
    for line in stdout.strip().split("\n"):
        match = re.search(pattern, line)
        if match:
            path = match.group(1).strip()
            if Path(path).exists():
                return path
    return None


def count_tasks(task_dir: str) -> int:
    """Count number of tasks in a directory by counting instruction.md files."""
    return len(list(Path(task_dir).rglob("instruction.md")))


def _load_secret_env():
    """Load API keys from secret.env into os.environ."""
    if SECRET_ENV.exists():
        for line in SECRET_ENV.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                os.environ.setdefault(key, val)


def upload_dataset(task_dir: str, repo_suffix: str) -> Optional[str]:
    """Upload a task directory to HuggingFace as a -large dataset."""
    repo_id = f"{HF_ORG}/exp_rpt_{repo_suffix}"

    # Import the upload function
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.commons import upload_tasks_to_hf

    # Ensure we have API keys loaded
    _load_secret_env()

    print(f"  Uploading to {repo_id}...")
    try:
        # Temporarily unset SKIP_HF_UPLOAD for the actual upload
        old_val = os.environ.pop("SKIP_HF_UPLOAD", None)
        repo_url = upload_tasks_to_hf(
            task_dir, repo_id,
            commit_message=f"Upload {repo_suffix} dataset ({count_tasks(task_dir)} tasks)",
        )
        if old_val is not None:
            os.environ["SKIP_HF_UPLOAD"] = old_val
        print(f"  Uploaded: {repo_url}")
        return repo_url
    except Exception as e:
        print(f"  [ERROR] Upload failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate -large (5000-task) dataset versions and upload to HuggingFace"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="Specific datasets to generate (by short name, e.g. pytest jest)"
    )
    parser.add_argument(
        "--batch", type=int, choices=[1, 2, 3],
        help="Run a specific batch (1=stack, 2=large-source, 3=remaining)"
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip HuggingFace upload (generate only)"
    )
    parser.add_argument(
        "--upload-only", type=str,
        help="JSON file mapping short_name -> task_dir (skip generation, upload only)"
    )
    args = parser.parse_args()

    # Determine which datasets to process
    datasets = DATASETS
    if args.batch:
        batch_names = BATCHES[args.batch]
        datasets = [d for d in DATASETS if d[0] in batch_names]
    elif args.datasets:
        datasets = [d for d in DATASETS if d[0] in args.datasets]

    print(f"{'='*70}")
    print(f"Generating -large datasets ({len(datasets)} total)")
    print(f"{'='*70}")
    for short_name, script, limit, suffix in datasets:
        print(f"  {short_name:<20} -> {HF_ORG}/exp_rpt_{suffix}  (limit={limit})")
    print()

    # Load pre-existing task dirs if upload-only mode
    existing_dirs = {}
    if args.upload_only:
        with open(args.upload_only) as f:
            existing_dirs = json.load(f)

    results = {}
    for short_name, script_name, limit, hf_suffix in datasets:
        print(f"\n{'='*70}")
        print(f"[{short_name}] Generating {limit} tasks -> {HF_ORG}/exp_rpt_{hf_suffix}")
        print(f"{'='*70}")

        # Step 1: Generate (or use existing)
        task_dir = existing_dirs.get(short_name)
        if not task_dir:
            task_dir = run_generator(script_name, limit)

        if not task_dir or not Path(task_dir).exists():
            results[short_name] = {
                "status": "generation_failed",
                "error": "No output directory",
            }
            print(f"  [FAILED] No output directory for {short_name}")
            continue

        n_tasks = count_tasks(task_dir)
        print(f"  Generated {n_tasks} tasks in {task_dir}")

        # Step 2: Upload
        repo_url = None
        if not args.no_upload:
            repo_url = upload_dataset(task_dir, hf_suffix)

        results[short_name] = {
            "status": "ok" if (repo_url or args.no_upload) else "upload_failed",
            "task_dir": task_dir,
            "n_tasks": n_tasks,
            "repo_url": repo_url,
            "hf_suffix": hf_suffix,
        }

    # ─── Summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("GENERATION SUMMARY")
    print(f"{'='*70}")

    ok = 0
    total_tasks = 0
    for short_name in [d[0] for d in datasets]:
        r = results.get(short_name, {})
        status = r.get("status", "unknown")
        n = r.get("n_tasks", 0)
        td = r.get("task_dir", "")
        url = r.get("repo_url", "")

        if status == "ok":
            ok += 1
            total_tasks += n
            print(f"  OK    {short_name:<20} {n:>5} tasks  {url or td}")
        else:
            print(f"  FAIL  {short_name:<20} {r.get('error', status)}")

    print(f"\n{ok}/{len(datasets)} succeeded, {total_tasks} total tasks generated")

    # Save results for potential re-upload
    results_file = PROJECT_ROOT / "data" / f"large_dataset_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

    # Save task_dirs mapping for --upload-only re-runs
    task_dirs_map = {k: v.get("task_dir") for k, v in results.items() if v.get("task_dir")}
    dirs_file = PROJECT_ROOT / "data" / "large_dataset_dirs.json"
    with open(dirs_file, "w") as f:
        json.dump(task_dirs_map, f, indent=2)
    print(f"Task dirs saved to: {dirs_file}")


if __name__ == "__main__":
    main()
