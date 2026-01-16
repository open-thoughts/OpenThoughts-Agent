#!/usr/bin/env python3
"""
Generate harder versions of r2egym tasks using an LLM.
"""

import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data.synthetic_harden.utils import harden_instructions
from data.r2egym.utils import load_r2egym_instances, create_r2egym_instruction, create_r2egym_task_toml
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, create_standard_dockerfile, upsample_list


def main() -> str:
    random.seed(42)

    instances = load_r2egym_instances()
    instructions = upsample_list([create_r2egym_instruction(inst) for inst in instances], 10_000)
    hardened_instructions = harden_instructions(instructions, model="gpt-4o-mini")

    dockerfile = create_standard_dockerfile()
    task_toml = create_r2egym_task_toml()

    hdf5_path = generate_tasks_to_hdf5(
        instructions=hardened_instructions,
        metadata=[{}] * len(hardened_instructions),
        dockerfiles=[dockerfile] * len(hardened_instructions),
        task_toml=[task_toml] * len(hardened_instructions),
        dataset_prefix="r2egym_hardened",
    )

    extracted_dir = extract_hdf5_to_task_paths(hdf5_path)
    upload_tasks_to_hf(extracted_dir, "DCAgent/exp-syh-r2egym-askllm-hardened")
    return extracted_dir


if __name__ == "__main__":
    print(main())
