#!/usr/bin/env python3
"""Upsample small datasets to 10K tasks and upload to HuggingFace.

Reads dataset_analysis_report.jsonl, upsamples datasets with <10K tasks,
uploads as DCAgent/{name}_10k, and writes launch_list.txt.

Run from login node (needs internet + HF_TOKEN).

Usage:
    source ~/data_gen_secrets.env
    export PYTHONPATH="/e/scratch/jureap59/feuer1/OpenThoughts-Agent:/e/scratch/jureap59/feuer1/OpenThoughts-Agent/scripts/harbor:/e/scratch/jureap59/etash/harbor/src:/e/scratch/jureap59/feuer1/OpenThoughts-Agent/data:$PYTHONPATH"
    /lib/ld-linux-aarch64.so.1 /e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python3.12 prepare_datasets.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

# Redirect temp files and HF cache to playground (scratch has quota limits)
PLAYGROUND_TMP = "/e/data1/datasets/playground/mmlaion/shared/guha1/tmp_glm47"
os.environ["TMPDIR"] = PLAYGROUND_TMP
os.environ["HF_HOME"] = os.path.join(PLAYGROUND_TMP, "hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(PLAYGROUND_TMP, "hf_cache", "datasets")

from tasks_parquet_converter import from_hf_dataset
from commons import upsample_tasks_directory, upload_tasks_to_hf

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_PATH = SCRIPT_DIR / "dataset_analysis_report.jsonl"
LAUNCH_LIST_PATH = SCRIPT_DIR / "launch_list.txt"
PROGRESS_PATH = SCRIPT_DIR / "prepare_progress.jsonl"

TARGET_TASKS = 10000

HF_DATASETS_DIR = os.path.join(PLAYGROUND_TMP, "hf_datasets")
HF_CACHE_DIR = os.path.join(PLAYGROUND_TMP, "hf_cache")


def _cleanup_hf_caches():
    """Remove HF dataset cache and extracted tasks to free disk space."""
    for d in [HF_DATASETS_DIR, HF_CACHE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)


def get_upsampled_repo_id(repo_id: str) -> str:
    """Derive the upsampled repo ID: DCAgent/{dataset_name}_10k."""
    name = repo_id.split("/")[-1]
    return f"DCAgent/{name}_10k"


def load_progress() -> set:
    """Load already-processed repo IDs from progress file."""
    done = set()
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add(rec["repo_id"])
    return done


def save_progress(repo_id: str, status: str, output_repo: str = None):
    """Append a progress record."""
    with open(PROGRESS_PATH, "a") as f:
        f.write(json.dumps({
            "repo_id": repo_id,
            "status": status,
            "output_repo": output_repo,
        }) + "\n")


def main():
    if not REPORT_PATH.exists():
        print(f"ERROR: {REPORT_PATH} not found. Run analyze_datasets.py first.")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Run: source ~/data_gen_secrets.env")
        sys.exit(1)

    # Load analysis report
    records = []
    with open(REPORT_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter to actionable datasets
    actionable = [r for r in records if not r["skip_reason"] and not r["error"]]
    print(f"Total records: {len(records)}, actionable: {len(actionable)}")

    done = load_progress()
    launch_entries = []

    for i, rec in enumerate(actionable, 1):
        repo_id = rec["repo_id"]

        if rec["needs_upsample"]:
            upsampled_repo = get_upsampled_repo_id(repo_id)

            if repo_id in done:
                print(f"[{i}/{len(actionable)}] {repo_id} -> already done, using {upsampled_repo}")
                launch_entries.append(upsampled_repo)
                continue

            print(f"\n[{i}/{len(actionable)}] Upsampling {repo_id} ({rec['num_tasks']} -> {TARGET_TASKS}) ...")
            try:
                task_base = from_hf_dataset(repo_id)
                upsampled_path = upsample_tasks_directory(task_base, TARGET_TASKS)

                print(f"  Uploading to {upsampled_repo} ...")
                upload_tasks_to_hf(upsampled_path, upsampled_repo, token=hf_token)

                # Clean up upsampled temp dir and original extraction
                if upsampled_path != task_base and Path(upsampled_path).exists():
                    shutil.rmtree(upsampled_path, ignore_errors=True)
                if Path(task_base).exists():
                    shutil.rmtree(task_base, ignore_errors=True)

                save_progress(repo_id, "uploaded", upsampled_repo)
                launch_entries.append(upsampled_repo)
                print(f"  -> OK: {upsampled_repo}")

            except Exception as e:
                print(f"  -> ERROR: {e}")
                save_progress(repo_id, f"error: {e}")
                # Still add original to launch list as fallback
                launch_entries.append(repo_id)
            finally:
                # Aggressively clean up HF cache to free disk space
                _cleanup_hf_caches()
        else:
            # Already >= 10K tasks, use original
            launch_entries.append(repo_id)
            if repo_id not in done:
                save_progress(repo_id, "no_upsample_needed", repo_id)

    # Write launch list
    with open(LAUNCH_LIST_PATH, "w") as f:
        for entry in launch_entries:
            f.write(entry + "\n")

    print(f"\n{'=' * 60}")
    print(f"Launch list written to {LAUNCH_LIST_PATH}")
    print(f"Total datasets to launch: {len(launch_entries)}")
    upsample_count = sum(1 for r in actionable if r["needs_upsample"])
    print(f"  Upsampled: {upsample_count}")
    print(f"  Original (>= 10K): {len(launch_entries) - upsample_count}")


if __name__ == "__main__":
    main()
