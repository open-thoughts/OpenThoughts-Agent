#!/usr/bin/env python3
"""
Generate R2E-Gym tasks with plain Ubuntu Docker (no verification).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.r2egym.utils import load_r2egym_instances, create_r2egym_instruction, create_r2egym_task_toml
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, create_standard_dockerfile


def filter_unique_instructions(
    instructions: list,
    metadata: list,
    dockerfiles: list,
    task_tomls: list,
) -> tuple[list, list, list, list]:
    """
    Filter lists to keep only entries with unique instructions.

    Returns the first occurrence of each unique instruction along with
    its corresponding metadata, dockerfile, and task_toml.
    """
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



    instances = load_r2egym_instances()
    dockerfile = create_standard_dockerfile()
    task_toml = create_r2egym_task_toml()

    instructions = [create_r2egym_instruction(inst) for inst in instances]
    metadata = instances
    dockerfiles = [dockerfile] * len(instances)
    task_tomls = [task_toml] * len(instances)

    instructions, metadata, dockerfiles, task_tomls = filter_unique_instructions(
        instructions, metadata, dockerfiles, task_tomls
    )


    ratio = 10_000 / len(instructions)

    
    hdf5_path = generate_tasks_to_hdf5(
        instructions=instructions,
        metadata=metadata,
        dockerfiles=dockerfiles,
        task_toml=task_tomls,
        dataset_prefix="r2egym_wo_docker",
    )

    extracted_paths = extract_hdf5_to_task_paths(hdf5_path)
    extracted_dir = str(extracted_paths[0]).rsplit('/', 1)[0]
    upload_tasks_to_hf(extracted_dir, "DCAgent/exp-swd-r2egym-wo-docker")
    return extracted_dir


if __name__ == "__main__":
    print(main())
