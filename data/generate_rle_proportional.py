#!/usr/bin/env python3
"""
Generate a 500-task RLE dataset for the proportional reward experiment.

Loads tasks from DCAgent/exp_rpt_stack-pytest, replaces test.sh with a
proportional reward version (fraction of tests passed instead of binary
pass/fail), creates task directories, and uploads to HuggingFace as
DCAgent/exp_rle_proportional.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root and data dir to path
sys.path.insert(0, "/scratch/10000/eguha3/dc-agent")
sys.path.insert(0, "/scratch/10000/eguha3/dc-agent/data")

from data.commons import (
    load_harbor_tasks_as_dicts,
    create_harbor_task_directory_generic,
    upload_tasks_to_hf,
)

# -- Configuration --
SOURCE_REPO = "DCAgent/exp_rpt_stack-pytest"
TARGET_REPO = "DCAgent/exp_rle_proportional"
NUM_TASKS = 500
DATASET_PREFIX = "exp-rle-proportional"
MANIFEST_PATH = Path("/scratch/10000/eguha3/dc-agent/data/rle_manifest.json")

# -- Proportional reward test.sh --
PROPORTIONAL_TEST_SH = r"""#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier

# Create virtual environment if needed (fixes PEP 668 on Python 3.12+)
if [ ! -d /app/.venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv /app/.venv
fi

# Activate virtual environment
source /app/.venv/bin/activate

# Install pytest if not available
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install --quiet pytest
fi

# Install whitelisted test dependencies (not agent-created packages)
WHITELIST="requests numpy pandas scipy scikit-learn sklearn torch tensorflow keras httpx aiohttp ddtrace django flask fastapi matplotlib seaborn pillow pydantic pytest-mock requests-mock faker pyyaml pytz cryptography bcrypt hypothesis"
for pkg in $WHITELIST; do
    python3 -c "import ${pkg//-/_}" 2>/dev/null || pip install --quiet "$pkg" 2>/dev/null || true
done

cd /app
export PYTHONPATH="/app:${PYTHONPATH:-}"

# Run all test files, count pass/fail
TOTAL=0
PASSED=0

for test_file in /tests/test_*.py; do
    [ -f "$test_file" ] || continue
    TEST_COUNT=$(grep -c "def test_" "$test_file" 2>/dev/null || true)
    TEST_COUNT=${TEST_COUNT:-0}
    TOTAL=$((TOTAL + TEST_COUNT))

    RESULT=$(python -m pytest "$test_file" -v --tb=no 2>&1 || true)
    FILE_PASSED=$(echo "$RESULT" | grep -c " PASSED" || true)
    FILE_PASSED=${FILE_PASSED:-0}
    PASSED=$((PASSED + FILE_PASSED))
done

if [ $TOTAL -eq 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi

REWARD=$(python3 -c "print(round($PASSED / $TOTAL, 4))")
echo "$REWARD" > /logs/verifier/reward.txt
echo "Passed $PASSED / $TOTAL tests (reward=$REWARD)"

if [ "$PASSED" -eq "$TOTAL" ]; then
    exit 0
else
    exit 1
fi
""".lstrip()


def main():
    print(f"Loading tasks from {SOURCE_REPO} (limit={NUM_TASKS})...")
    tasks = load_harbor_tasks_as_dicts(SOURCE_REPO, limit=NUM_TASKS)
    print(f"  Loaded {len(tasks)} tasks from source repo.")

    if len(tasks) < NUM_TASKS:
        print(f"  WARNING: Only {len(tasks)} tasks available (requested {NUM_TASKS}).")

    actual_count = min(len(tasks), NUM_TASKS)
    tasks = tasks[:actual_count]
    print(f"  Using {actual_count} tasks.")

    # Create output directory
    output_dir = Path(tempfile.mkdtemp(
        prefix=f"{DATASET_PREFIX}_",
        dir="/scratch/10000/eguha3/data_cache",
    ))
    print(f"  Output directory: {output_dir}")

    # Create task directories with proportional test.sh
    created = 0
    for i, task in enumerate(tasks):
        test_files = json.loads(task.get("test_files_json", "{}"))

        metadata = {}
        if task.get("metadata"):
            try:
                metadata = json.loads(task["metadata"])
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        # Tag with source info
        metadata["source_repo"] = SOURCE_REPO
        metadata["source_task"] = task.get("task_dir_name", f"task-{i}")
        metadata["reward_type"] = "proportional"

        create_harbor_task_directory_generic(
            output_dir=output_dir,
            task_id=i,
            instruction=task["instruction"],
            dockerfile=task["dockerfile"],
            test_sh=PROPORTIONAL_TEST_SH,
            test_files=test_files,
            dataset_prefix=DATASET_PREFIX,
            metadata=metadata,
        )
        created += 1

        if (created % 100) == 0:
            print(f"    Created {created}/{actual_count} tasks...")

    print(f"  Created {created} task directories.")

    # Upload to HuggingFace
    print(f"Uploading to {TARGET_REPO}...")
    url = upload_tasks_to_hf(
        dataset_path=str(output_dir),
        repo_id=TARGET_REPO,
        commit_message=f"Upload {created}-task proportional reward RLE dataset",
    )
    print(f"  Upload complete: {url}")

    # Update manifest
    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError):
            manifest = {}

    manifest["proportional"] = str(output_dir)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest updated: {MANIFEST_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print(f"DONE")
    print(f"  Tasks created:    {created}")
    print(f"  Output directory: {output_dir}")
    print(f"  HuggingFace repo: {TARGET_REPO}")
    print(f"  Manifest:         {MANIFEST_PATH}")
    print("=" * 60)

    return str(output_dir), created


if __name__ == "__main__":
    main()
