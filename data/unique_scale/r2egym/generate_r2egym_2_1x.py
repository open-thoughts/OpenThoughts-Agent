#!/usr/bin/env python3
"""
Generate R2E-Gym tasks with 2_1x upsampling.
~4762 unique instructions → upsampled to 10,000 total.
"""

import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.r2egym.utils import load_r2egym_instances, create_r2egym_instruction, create_r2egym_task_toml
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, create_standard_dockerfile

TARGET_TOTAL = 10_000
UPSAMPLE_RATIO = 2.1
NUM_UNIQUE = int(TARGET_TOTAL / UPSAMPLE_RATIO)  # ~4762
DATASET_SUFFIX = "2_1x"


def filter_unique_instructions(
    instructions: list,
    metadata: list,
    dockerfiles: list,
    task_tomls: list,
) -> tuple[list, list, list, list]:
    """Filter lists to keep only entries with unique instructions."""
    seen_instructions = set()
    unique_instructions = []
    unique_metadata = []
    unique_dockerfiles = []
    unique_task_tomls = []

    for inst, meta, dockerfile, task_toml in zip(instructions, metadata, dockerfiles, task_tomls):
        if inst not in seen_instructions:
            seen_instructions.add(inst)
            unique_instructions.append(inst)
            unique_metadata.append(meta)
            unique_dockerfiles.append(dockerfile)
            unique_task_tomls.append(task_toml)

    return unique_instructions, unique_metadata, unique_dockerfiles, unique_task_tomls


def downsample_data(
    instructions: list,
    metadata: list,
    dockerfiles: list,
    task_tomls: list,
    num_samples: int,
) -> tuple[list, list, list, list]:
    """Downsample to num_samples unique instructions."""
    if len(instructions) <= num_samples:
        return instructions, metadata, dockerfiles, task_tomls

    indices = random.sample(range(len(instructions)), num_samples)
    indices.sort()

    return (
        [instructions[i] for i in indices],
        [metadata[i] for i in indices],
        [dockerfiles[i] for i in indices],
        [task_tomls[i] for i in indices],
    )


def upsample_to_target(
    instructions: list,
    metadata: list,
    dockerfiles: list,
    task_tomls: list,
    target_count: int,
) -> tuple[list, list, list, list]:
    """Upsample data by duplicating entries to reach target count."""
    upsampled_instructions = []
    upsampled_metadata = []
    upsampled_dockerfiles = []
    upsampled_task_tomls = []

    idx = 0
    while len(upsampled_instructions) < target_count:
        upsampled_instructions.append(instructions[idx % len(instructions)])
        upsampled_metadata.append(metadata[idx % len(metadata)])
        upsampled_dockerfiles.append(dockerfiles[idx % len(dockerfiles)])
        upsampled_task_tomls.append(task_tomls[idx % len(task_tomls)])
        idx += 1

    return upsampled_instructions, upsampled_metadata, upsampled_dockerfiles, upsampled_task_tomls


def main() -> str:
    random.seed(42)

    instances = load_r2egym_instances()
    dockerfile = create_standard_dockerfile()
    task_toml = create_r2egym_task_toml()

    instructions = [create_r2egym_instruction(inst) for inst in instances]
    metadata = instances
    dockerfiles = [dockerfile] * len(instances)
    task_tomls = [task_toml] * len(instances)

    # Filter to unique
    instructions, metadata, dockerfiles, task_tomls = filter_unique_instructions(
        instructions, metadata, dockerfiles, task_tomls
    )
    print(f"Total unique instructions available: {len(instructions)}")

    # Downsample to target number of unique instructions
    instructions, metadata, dockerfiles, task_tomls = downsample_data(
        instructions, metadata, dockerfiles, task_tomls, NUM_UNIQUE
    )
    print(f"Downsampled to {len(instructions)} unique instructions")

    # Upsample to target total
    instructions, metadata, dockerfiles, task_tomls = upsample_to_target(
        instructions, metadata, dockerfiles, task_tomls, TARGET_TOTAL
    )
    print(f"Upsampled to {len(instructions)} total instructions ({UPSAMPLE_RATIO}x)")

    hdf5_path = generate_tasks_to_hdf5(
        instructions=instructions,
        metadata=metadata,
        dockerfiles=dockerfiles,
        task_toml=task_tomls,
        dataset_prefix=f"r2egym_wo_docker_{DATASET_SUFFIX}",
    )

    extracted_dir = extract_hdf5_to_task_paths(hdf5_path)
    upload_tasks_to_hf(extracted_dir, f"DCAgent/exp-uns-r2egym-{DATASET_SUFFIX}")
    return extracted_dir


if __name__ == "__main__":
    print(main())
