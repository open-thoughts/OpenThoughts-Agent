#!/usr/bin/env python3
"""
Generate harder versions of r2egym tasks by mixing with SWEsmith questions using an LLM.
"""

import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from datasets import load_dataset
from data.synthetic_harden.utils import mix_questions
from data.r2egym.utils import load_r2egym_instances, create_r2egym_instruction, create_r2egym_task_toml, create_r2egym_dockerfile
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, upsample_list


def load_swesmith_problem_statements() -> list[str]:
    """Load problem statements from SWEsmith dataset."""
    print("Loading SWEsmith dataset from HuggingFace...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")

    problem_statements = []
    for row in ds:
        ps = row.get("problem_statement")
        if ps and ps.strip():
            problem_statements.append(ps)

    print(f"Loaded {len(problem_statements)} SWEsmith problem statements")
    return problem_statements


def main() -> str:
    random.seed(42)

    # Load R2E-Gym instances (keep instances paired with instructions for dockerfile generation)
    print("Loading R2E-Gym instances...")
    r2egym_instances = load_r2egym_instances()
    # Create list of (instance, instruction) pairs
    r2egym_pairs = [(inst, create_r2egym_instruction(inst)) for inst in r2egym_instances]
    print(f"Loaded {len(r2egym_pairs)} R2E-Gym instances")

    # Load SWEsmith problem statements
    swesmith_problems = load_swesmith_problem_statements()

    # Shuffle both lists for random pairing
    random.shuffle(r2egym_pairs)
    random.shuffle(swesmith_problems)

    # Upsample to target 10,000 mixed questions (keeping instance pairing)
    r2egym_pairs = upsample_list(r2egym_pairs, 10_000)
    swesmith_problems = upsample_list(swesmith_problems, 10_000)

    # Extract instructions for mixing
    r2egym_instructions = [pair[1] for pair in r2egym_pairs]
    r2egym_instances_upsampled = [pair[0] for pair in r2egym_pairs]

    # Mix the questions using the LLM
    print("Mixing questions with LLM...")
    mixed_instructions = mix_questions(
        instructions_a=r2egym_instructions,
        instructions_b=swesmith_problems,
        source_a="R2E-Gym",
        source_b="SWEsmith",
        model="gpt-5-nano",
    )
    print(f"Generated {len(mixed_instructions)} mixed instructions")

    # Use R2E-Gym dockerfiles for each instance
    dockerfiles = [create_r2egym_dockerfile(inst) for inst in r2egym_instances_upsampled]
    task_toml = create_r2egym_task_toml()

    # Create metadata from R2E-Gym instances
    metadata_list = [
        {
            "instance_id": inst.get("instance_id"),
            "docker_image": inst.get("docker_image"),
            "base_commit": inst.get("base_commit"),
            "repo_name": inst.get("repo_name"),
            "expected_output_json": inst.get("expected_output_json", "{}"),
            "source": "r2egym_swesmith_mixed"
        }
        for inst in r2egym_instances_upsampled
    ]

    hdf5_path = generate_tasks_to_hdf5(
        instructions=mixed_instructions,
        metadata=metadata_list,
        dockerfiles=dockerfiles,
        task_toml=[task_toml] * len(mixed_instructions),
        dataset_prefix="r2egym_swesmith_mixed",
    )

    extracted_dir = extract_hdf5_to_task_paths(hdf5_path)
    upload_tasks_to_hf(extracted_dir, "DCAgent/exp-syh-r2egym-swesmith-mixed")

    return extracted_dir


if __name__ == "__main__":
    print(main())
