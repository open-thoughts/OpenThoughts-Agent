#!/usr/bin/env python3
"""
Split dev_set_71_tasks into two parts, upsample, and upload to HuggingFace in parquet format.
"""

import sys
import shutil
import tempfile
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from huggingface_hub import snapshot_download
from scripts.harbor.tasks_parquet_converter import from_hf_dataset_hdf5
from data.commons import (
    upload_tasks_to_hf,
    upsample_tasks_hdf5,
    extract_hdf5_to_task_paths,
)

# Configuration
SOURCE_REPO = "DCAgent/dev_set_71_tasks"
PART1_REPO = "DCAgent/dev_set_part1"
PART2_REPO = "DCAgent/dev_set_part2"
PART1_UPSAMPLED_REPO = "DCAgent/dev_set_part1_10k"
UPSAMPLE_TARGET = 10_000


def main():
    # Step 1: Download source dataset
    print(f"Step 1: Downloading {SOURCE_REPO}...")
    source_path = Path(snapshot_download(SOURCE_REPO, repo_type="dataset"))
    print(f"Downloaded to: {source_path}")

    # Step 2: Get all task folders and split
    print("\nStep 2: Splitting tasks...")
    task_folders = sorted([d for d in source_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
    mid_point = len(task_folders) // 2
    part1_tasks = task_folders[:mid_point]
    part2_tasks = task_folders[mid_point:]
    print(f"Found {len(task_folders)} tasks -> Part 1: {len(part1_tasks)}, Part 2: {len(part2_tasks)}")

    # Step 3: Create temp directories and copy tasks
    print("\nStep 3: Copying tasks to temp directories...")
    tmpdir = Path(tempfile.mkdtemp())
    part1_dir = tmpdir / "part1"
    part2_dir = tmpdir / "part2"
    part1_dir.mkdir()
    part2_dir.mkdir()

    for task_folder in part1_tasks:
        shutil.copytree(task_folder, part1_dir / task_folder.name, symlinks=False)
    print(f"Copied {len(part1_tasks)} tasks to part1")

    for task_folder in part2_tasks:
        shutil.copytree(task_folder, part2_dir / task_folder.name, symlinks=False)
    print(f"Copied {len(part2_tasks)} tasks to part2")

    # Step 4: Upload part1 (parquet)
    print(f"\nStep 4: Uploading {PART1_REPO}...")
    upload_tasks_to_hf(str(part1_dir), PART1_REPO)

    # Step 5: Upload part2 (parquet)
    print(f"\nStep 5: Uploading {PART2_REPO}...")
    upload_tasks_to_hf(str(part2_dir), PART2_REPO)

    # Step 6: Download part1 and convert to HDF5
    print(f"\nStep 6: Converting {PART1_REPO} to HDF5...")
    hdf5_path = from_hf_dataset_hdf5(PART1_REPO)
    print(f"HDF5: {hdf5_path}")

    # Step 7: Upsample HDF5
    print(f"\nStep 7: Upsampling to {UPSAMPLE_TARGET} tasks...")
    upsampled_hdf5 = upsample_tasks_hdf5(hdf5_path, UPSAMPLE_TARGET)
    print(f"Upsampled HDF5: {upsampled_hdf5}")

    # Step 8: Extract HDF5 to task directories
    print(f"\nStep 8: Extracting HDF5 to task directories...")
    upsampled_task_dir = extract_hdf5_to_task_paths(upsampled_hdf5)
    print(f"Extracted to: {upsampled_task_dir}")

    # Step 9: Upload upsampled dataset
    print(f"\nStep 9: Uploading {PART1_UPSAMPLED_REPO}...")
    upload_tasks_to_hf(upsampled_task_dir, PART1_UPSAMPLED_REPO)

    # Done
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Part 1: https://huggingface.co/datasets/{PART1_REPO}")
    print(f"Part 2: https://huggingface.co/datasets/{PART2_REPO}")
    print(f"Part 1 (10k): https://huggingface.co/datasets/{PART1_UPSAMPLED_REPO}")


if __name__ == "__main__":
    main()
