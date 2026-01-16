#!/usr/bin/env python3
"""
Generate swesmith dataset in sandboxes style - deduplicated and subsampled
"""

import random
import tempfile
from pathlib import Path

from datasets import load_dataset
from data.commons import upload_tasks_to_hf, subsample_tasks_directory, upload_traces_to_hf
from data.swesmith.generate_with_plain_docker import create_sandboxed_task
from scripts.harbor.run_and_export_traces import run_dataset_to_traces


def main() -> None:
    """Main function - load all swesmith, dedup, and generate tasks"""
    # Load all data
    print("Loading swesmith dataset...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")
    print(f"Total rows in dataset: {len(ds)}")

    # Extract problem statements and deduplicate
    seen = set()
    unique_rows = []
    for i, row in enumerate(ds):
        problem = row.get("problem_statement", "")
        patch = row.get("patch", "")
        if not problem or not patch:
            continue
        if problem not in seen:
            seen.add(problem)
            unique_rows.append(row)

    print(f"After dedup: {len(unique_rows)} unique tasks")

    # Shuffle and select up to 31.6k
    random.shuffle(unique_rows)
    selected = unique_rows[:31_600]
    print(f"Selected {len(selected)} tasks")

    # Generate task directories
    out_root = Path(tempfile.mkdtemp())

    for idx, row in enumerate(selected):
        create_sandboxed_task(row, out_root, idx)

    print(f"Generated tasks at: {out_root}")

    upload_tasks_to_hf(str(out_root), "DCAgent/exp-psu-swesmith-31K")


if __name__ == "__main__":
    main()
