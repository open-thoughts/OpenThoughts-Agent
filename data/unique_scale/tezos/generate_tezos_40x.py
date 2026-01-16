#!/usr/bin/env python3
"""
Generate Tezos tasks with 40x upsampling.
~250 unique instructions → upsampled to 10,000 total.
"""

import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data.stackexchange.generate_codereview import download_and_extract_dataset, parse_posts_xml, extract_questions_from_data
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, create_standard_dockerfile, create_standard_task_toml

TARGET_TOTAL = 10_000
UPSAMPLE_RATIO = 40
NUM_UNIQUE = int(TARGET_TOTAL / UPSAMPLE_RATIO)  # ~250
DATASET_SUFFIX = "40x"


def downsample_data(
    instructions: list,
    metadata: list,
    num_samples: int,
) -> tuple[list, list]:
    """Downsample to num_samples unique instructions."""
    if len(instructions) <= num_samples:
        return instructions, metadata

    indices = random.sample(range(len(instructions)), num_samples)
    indices.sort()

    return (
        [instructions[i] for i in indices],
        [metadata[i] for i in indices],
    )


def upsample_to_target(
    instructions: list,
    metadata: list,
    target_count: int,
) -> tuple[list, list]:
    """Upsample data by duplicating entries to reach target count."""
    upsampled_instructions = []
    upsampled_metadata = []

    idx = 0
    while len(upsampled_instructions) < target_count:
        upsampled_instructions.append(instructions[idx % len(instructions)])
        upsampled_metadata.append(metadata[idx % len(metadata)])
        idx += 1

    return upsampled_instructions, upsampled_metadata


def main() -> str:
    random.seed(42)

    # Load tezos data
    posts_xml_path = download_and_extract_dataset("https://archive.org/download/stackexchange/tezos.stackexchange.com.7z")
    questions_data = parse_posts_xml(posts_xml_path)
    questions = extract_questions_from_data(questions_data)

    # Extract instructions and metadata
    instructions = [q if isinstance(q, str) else q[0] for q in questions]
    metadata = [q[1] if isinstance(q, tuple) and len(q) > 1 else {} for q in questions]

    print(f"Total unique instructions available: {len(instructions)}")

    # Downsample to target number of unique instructions
    instructions, metadata = downsample_data(instructions, metadata, NUM_UNIQUE)
    print(f"Downsampled to {len(instructions)} unique instructions")

    # Upsample to target total
    instructions, metadata = upsample_to_target(instructions, metadata, TARGET_TOTAL)
    print(f"Upsampled to {len(instructions)} total instructions ({UPSAMPLE_RATIO}x)")

    dockerfile = create_standard_dockerfile()
    task_toml = create_standard_task_toml()

    hdf5_path = generate_tasks_to_hdf5(
        instructions=instructions,
        metadata=metadata,
        dockerfiles=[dockerfile] * len(instructions),
        task_toml=[task_toml] * len(instructions),
        dataset_prefix=f"tezos_{DATASET_SUFFIX}",
    )

    extracted_dir = extract_hdf5_to_task_paths(hdf5_path)
    upload_tasks_to_hf(extracted_dir, f"DCAgent/exp-uns-tezos-{DATASET_SUFFIX}")
    return extracted_dir


if __name__ == "__main__":
    print(main())
