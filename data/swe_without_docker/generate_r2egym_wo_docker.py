#!/usr/bin/env python3
"""
Generate R2E-Gym tasks with plain Ubuntu Docker (no verification).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.r2egym.utils import load_r2egym_instances, create_r2egym_instruction, create_r2egym_task_toml
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, create_standard_dockerfile


def main() -> str:
    instances = load_r2egym_instances()
    dockerfile = create_standard_dockerfile()
    task_toml = create_r2egym_task_toml()

    instructions = [create_r2egym_instruction(inst) for inst in instances]
    metadata = instances
    dockerfiles = [dockerfile] * len(instances)
    task_tomls = [task_toml] * len(instances)


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
