#!/usr/bin/env python3
"""Download HF dataset and shard into multiple parts for parallel processing."""

import sys
import shutil
import tempfile
from pathlib import Path

from tasks_parquet_converter import from_hf_dataset


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <hf_repo_id> <num_shards> <output_file>", file=sys.stderr)
        sys.exit(1)

    hf_repo_id = sys.argv[1]
    num_shards = int(sys.argv[2])
    output_file = sys.argv[3]

    # Download dataset (uses fingerprint-based caching)
    print(f"Downloading dataset: {hf_repo_id}")
    dataset_path = from_hf_dataset(hf_repo_id)
    fingerprint = dataset_path.name  # fingerprint is the last path component
    print(f"Dataset path: {dataset_path}")
    print(f"Fingerprint: {fingerprint}")

    # Determine shard directory based on fingerprint
    shard_dir = Path(tempfile.gettempdir()) / "hf_datasets_shards" / f"{fingerprint}_{num_shards}shards"
    shard_marker = shard_dir / ".sharding_complete"

    if shard_dir.exists() and shard_marker.exists():
        print(f"Shards already exist at: {shard_dir}")
    else:
        print(f"Creating shards at: {shard_dir}")
        shard_dir.mkdir(parents=True, exist_ok=True)

        task_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        print(f"Found {len(task_dirs)} tasks to shard into {num_shards} parts")

        for shard_idx in range(num_shards):
            shard_path = shard_dir / f"shard_{shard_idx}"
            shard_path.mkdir(exist_ok=True)
            shard_tasks = task_dirs[shard_idx::num_shards]
            print(f"Shard {shard_idx}: {len(shard_tasks)} tasks")
            for task_dir in shard_tasks:
                dest = shard_path / task_dir.name
                if not dest.exists():
                    shutil.copytree(task_dir, dest)

        # Write marker file
        shard_marker.write_text(f"fingerprint={fingerprint}\nnum_shards={num_shards}\n")
        print("Dataset sharding complete!")

    # Write output to file for bash to source
    with open(output_file, 'w') as f:
        f.write(f"SHARD_DIR={shard_dir}\n")
        f.write(f"DATASET_FINGERPRINT={fingerprint}\n")


if __name__ == "__main__":
    main()
