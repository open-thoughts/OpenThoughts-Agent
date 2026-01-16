#!/usr/bin/env python3
"""Export and upload expert traces from job directories to HuggingFace."""

import sys
from pathlib import Path

from harbor.utils.traces_utils import export_traces, push_dataset
from datasets import concatenate_datasets


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <hf_repo_id> <job_dir1> [job_dir2] ...", file=sys.stderr)
        sys.exit(1)

    repo_id = sys.argv[1]
    job_dirs = [Path(d) for d in sys.argv[2:] if d.strip()]

    if not job_dirs:
        print("ERROR: No job directories provided")
        sys.exit(1)

    print(f"Found {len(job_dirs)} job directories to export")
    print(f"Uploading to: {repo_id}")

    all_datasets = []
    for job_dir in job_dirs:
        if job_dir.exists():
            print(f"Exporting traces from {job_dir}...")
            ds = export_traces(
                root=job_dir,
                recursive=True,
                episodes='last',
                push=False,
                verbose=True,
                embed_tools_in_conversation = False,
            )
            if len(ds) > 0:
                all_datasets.append(ds)
                print(f"  Collected {len(ds)} rows from {job_dir}")
        else:
            print(f"WARNING: Job directory does not exist: {job_dir}")

    if all_datasets:
        print(f"Merging {len(all_datasets)} datasets...")
        merged_ds = concatenate_datasets(all_datasets)
        print(f"Total merged rows: {len(merged_ds)}")
        print(f"Pushing to {repo_id}...")
        push_dataset(merged_ds, repo_id)
        print(f"Successfully uploaded {len(merged_ds)} traces to {repo_id}")
    else:
        print("ERROR: No datasets collected from any shard")
        sys.exit(1)


if __name__ == "__main__":
    main()
