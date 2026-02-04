#!/usr/bin/env python3
"""
Generate r2egym tasks with artificial constraints added using an LLM.
"""

import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data.synthetic_harden.utils import add_constraints
from data.r2egym.utils import load_r2egym_instances, create_r2egym_instruction, create_r2egym_task_toml, create_r2egym_dockerfile
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, upsample_list


def main() -> str:
    random.seed(42)

    # Load instances and keep them paired with instructions
    print("Loading R2E-Gym instances...")
    instances = load_r2egym_instances()
    pairs = [(inst, create_r2egym_instruction(inst)) for inst in instances]
    print(f"Loaded {len(pairs)} R2E-Gym instances")

    # Shuffle and upsample
    random.shuffle(pairs)
    pairs = upsample_list(pairs, 10_000)

    # Extract instructions for constraint adding
    instructions = [pair[1] for pair in pairs]
    instances_upsampled = [pair[0] for pair in pairs]

    # Add artificial constraints using the LLM
    print("Adding artificial constraints with LLM...")
    constrained_instructions = add_constraints(instructions, model="gpt-5-nano")
    print(f"Generated {len(constrained_instructions)} constrained instructions")

    # Use R2E-Gym dockerfiles for each instance
    dockerfiles = [create_r2egym_dockerfile(inst) for inst in instances_upsampled]
    task_toml = create_r2egym_task_toml()

    # Create metadata from R2E-Gym instances
    metadata_list = [
        {
            "instance_id": inst.get("instance_id"),
            "docker_image": inst.get("docker_image"),
            "base_commit": inst.get("base_commit"),
            "repo_name": inst.get("repo_name"),
            "expected_output_json": inst.get("expected_output_json", "{}"),
            "source": "r2egym_constrained"
        }
        for inst in instances_upsampled
    ]

    hdf5_path = generate_tasks_to_hdf5(
        instructions=constrained_instructions,
        metadata=metadata_list,
        dockerfiles=dockerfiles,
        task_toml=[task_toml] * len(constrained_instructions),
        dataset_prefix="r2egym_constrained",
    )

    extracted_dir = extract_hdf5_to_task_paths(hdf5_path)
    upload_tasks_to_hf(extracted_dir, "DCAgent/exp-syh-r2egym-askllm-constrained")

    return extracted_dir


if __name__ == "__main__":
    print(main())
