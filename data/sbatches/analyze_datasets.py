#!/usr/bin/env python3
"""Analyze all GLM-4.7 candidate datasets: count tasks and unique dockerfiles.

Run from login node (needs internet for HF downloads).

Usage:
    source ~/data_gen_secrets.env
    export PYTHONPATH="/e/scratch/jureap59/feuer1/OpenThoughts-Agent:/e/scratch/jureap59/feuer1/OpenThoughts-Agent/scripts/harbor:/e/scratch/jureap59/etash/harbor/src:/e/scratch/jureap59/feuer1/OpenThoughts-Agent/data:$PYTHONPATH"
    /lib/ld-linux-aarch64.so.1 /e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python3.12 analyze_datasets.py
"""

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

# Redirect temp files and HF cache to playground (scratch has quota limits)
PLAYGROUND_TMP = "/e/data1/datasets/playground/mmlaion/shared/guha1/tmp_glm47"
os.environ["TMPDIR"] = PLAYGROUND_TMP
os.environ["HF_HOME"] = os.path.join(PLAYGROUND_TMP, "hf_cache")

from tasks_parquet_converter import from_hf_dataset, find_tasks
from harbor.utils.container_cache import analyze_task_dockerfiles

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_PATH = SCRIPT_DIR / "dataset_analysis_report.jsonl"

MAX_UNIQUE_ENVS = 8

DATASETS = [
    "DCAgent/exp_rpt_stack-bash",
    "DCAgent/exp_rpt_stack-bash-withtests",
    "DCAgent/exp_rpt_stack-ruby",
    "DCAgent/exp_rpt_stack-go",
    "DCAgent/exp_rpt_stack-junit",
    "DCAgent/exp_rpt_stack-csharp",
    "DCAgent/exp_rpt_stack-rust",
    "DCAgent/exp_rpt_stack-cpp",
    "DCAgent/exp_rpt_codenet-python",
    "DCAgent/exp_rpt_ghactions",
    "DCAgent/exp_rpt_bugswarm",
    "DCAgent/exp_rpt_crosscodeeval-java",
    "DCAgent/exp_rpt_crosscodeeval-typescript",
    "DCAgent/exp_rpt_crosscodeeval-csharp",
    "DCAgent/exp_rpt_taco",
    "DCAgent/exp_rpt_exercism-python",
    "DCAgent2/logsmith-easy-500",
    "DCAgent/exp_rpt_codeelo-v2",
    "DCAgent/exp_rpt_crosscodeeval-python-v2",
    "DCAgent/exp_rle_proportional",
    "DCAgent/exp_rle_adversarial",
    "DCAgent/exp_rle_2skill",
    "DCAgent/exp_rle_partial_ambiguity",
    "DCAgent/exp_rle_structural_debug",
    "DCAgent/exp_rle_curated",
    "DCAgent/exp_rle_error_report",
    "DCAgent/exp_rle_minimal_instructions",
    "DCAgent/exp_rle_github_issue",
    "DCAgent/exp_rle_detailed",
    "DCAgent/exp_rle_expert",
    "DCAgent/exp_rle_heavy_padding",
    "DCAgent/exp_rle_moderate_padding",
    "DCAgent/exp_flat25_baseline",
    "DCAgent/exp_flat25_speed_bonus",
    "DCAgent/exp_flat25_subtle_debug",
    "DCAgent/exp_flat25_pseudocode",
    "DCAgent/exp_flat25_stackoverflow",
    "DCAgent/exp_rpt_pymethods2test-v3",
    "DCAgent/exp_rpt_unitsyn-python-v3",
    "DCAgent/exp_rpt_defects4j-v3",
    "DCAgent/exp_rpt_manybugs-v2",
    "DCAgent/exp_rpt_nemotron-junit",
    "DCAgent/exp_rpt_nemotron-cpp",
    "DCAgent/exp_rpt_stack-pytest-large",
    "DCAgent/exp_rpt_stack-jest-large",
    "DCAgent/exp_rpt_stack-dockerfile-large",
    "DCAgent/exp_rpt_stack-selfdoc-large",
    "DCAgent/exp_rpt_stack-php-large",
    "DCAgent/exp_rpt_pymethods2test-large",
    "DCAgent/exp_rpt_unitsyn-python-large",
    "DCAgent/exp_rpt_methods2test-large",
    "DCAgent/exp_rpt_e2egit-large",
    "DCAgent/exp_rpt_softwareheritage-large",
    "DCAgent/exp_rpt_bigcodebench-large",
    "DCAgent/exp_rpt_bugsinpy-mf-large",
    "DCAgent/stackexchange-tezos-sandboxes-armo-rm",
    "DCAgent/stackexchange-overflow-sandboxes-armo-rm",
    "DCAgent/stackexchange-tezos-sandboxes-skywork",
    "DCAgent/stackexchange-overflow-sandboxes-skywork",
    "DCAgent/stackexchange-tezos-sandboxes-skywork-response",
    "DCAgent/stackexchange-overflow-sandboxes-skywork-response",
    "DCAgent/exp_rpt_nemotron-bash-v2",
    "DCAgent/exp_rpt_bugsinpy-v4",
    "DCAgent/exp_rpt_nemotron-pytest-gpt5mini-v2",
    "DCAgent/exp_rpt_pr",
    "DCAgent/exp_rpt_multifile",
    "DCAgent/exp_rpt_scaffold",
    "DCAgent/exp_rpt_issue",
    "DCAgent/exp_rpt_curriculum-easy",
    "DCAgent/exp_rpt_curriculum-medium",
    "DCAgent/exp_rpt_curriculum-hard",
]


def analyze_one(repo_id: str) -> dict:
    """Analyze a single dataset and return a report record."""
    record = {
        "repo_id": repo_id,
        "num_tasks": 0,
        "unique_envs": 0,
        "hash_counts": {},
        "needs_upsample": False,
        "skip_reason": None,
        "error": None,
    }

    task_base = None
    try:
        task_base = from_hf_dataset(repo_id)
        task_dirs = find_tasks(task_base, recursive=True)
        record["num_tasks"] = len(task_dirs)

        if len(task_dirs) == 0:
            record["skip_reason"] = "no_tasks"
            return record

        stats = analyze_task_dockerfiles(task_dirs)
        record["unique_envs"] = stats.unique_hashes
        record["hash_counts"] = dict(stats.hash_counts)

        if stats.unique_hashes > MAX_UNIQUE_ENVS:
            record["skip_reason"] = f"too_many_envs ({stats.unique_hashes})"
        elif record["num_tasks"] < 10000:
            record["needs_upsample"] = True

    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        # Clean up extracted tasks to save disk space (HF cache stays for reuse)
        if task_base and Path(task_base).exists():
            shutil.rmtree(task_base, ignore_errors=True)

    return record


def main():
    # Load already-processed datasets to support resuming
    done = set()
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add(rec["repo_id"])
        print(f"Resuming: {len(done)} datasets already analyzed")

    remaining = [ds for ds in DATASETS if ds not in done]
    print(f"Analyzing {len(remaining)} datasets ({len(done)} already done)")

    for i, repo_id in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] Analyzing {repo_id} ...")
        record = analyze_one(repo_id)

        with open(REPORT_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        status = "OK"
        if record["skip_reason"]:
            status = f"SKIP ({record['skip_reason']})"
        elif record["error"]:
            status = f"ERROR ({record['error'][:80]})"
        print(
            f"  -> {status} | tasks={record['num_tasks']} "
            f"unique_envs={record['unique_envs']} "
            f"upsample={record['needs_upsample']}"
        )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_records = []
    with open(REPORT_PATH) as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    ok = [r for r in all_records if not r["skip_reason"] and not r["error"]]
    skipped = [r for r in all_records if r["skip_reason"]]
    errored = [r for r in all_records if r["error"]]
    need_upsample = [r for r in ok if r["needs_upsample"]]

    print(f"Total: {len(all_records)}")
    print(f"  OK: {len(ok)} ({len(need_upsample)} need upsample)")
    print(f"  Skipped: {len(skipped)}")
    for r in skipped:
        print(f"    - {r['repo_id']}: {r['skip_reason']}")
    print(f"  Errors: {len(errored)}")
    for r in errored:
        print(f"    - {r['repo_id']}: {r['error'][:100]}")

    # Collect all unique env hashes across OK datasets
    all_hashes = set()
    for r in ok:
        all_hashes.update(r["hash_counts"].keys())
    print(f"\nTotal unique env hashes (for snapshot creation): {len(all_hashes)}")


if __name__ == "__main__":
    main()
