#!/usr/bin/env python3
"""
Generate tezos tasks with artificial constraints added using an LLM.
"""

import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data.synthetic_harden.utils import add_constraints
from data.stackexchange.generate_codereview import download_and_extract_dataset, parse_posts_xml, extract_questions_from_data
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, create_standard_dockerfile, create_standard_task_toml, upsample_list


TEZOS_URL = "https://archive.org/download/stackexchange/tezos.stackexchange.com.7z"


def main() -> str:
    random.seed(42)

    # Load Tezos questions
    print("Loading Tezos questions...")
    tezos_posts_xml = download_and_extract_dataset(TEZOS_URL)
    tezos_data = parse_posts_xml(tezos_posts_xml)
    tezos_questions = extract_questions_from_data(tezos_data)
    print(f"Loaded {len(tezos_questions)} Tezos questions")

    # Shuffle and upsample
    random.shuffle(tezos_questions)
    tezos_questions = upsample_list(tezos_questions, 10_000)

    # Add artificial constraints using the LLM
    print("Adding artificial constraints with LLM...")
    constrained_instructions = add_constraints(tezos_questions, model="gpt-5-nano")
    print(f"Generated {len(constrained_instructions)} constrained instructions")

    dockerfile = create_standard_dockerfile()
    task_toml = create_standard_task_toml()

    hdf5_path = generate_tasks_to_hdf5(
        instructions=constrained_instructions,
        metadata=[{}] * len(constrained_instructions),
        dockerfiles=[dockerfile] * len(constrained_instructions),
        task_toml=[task_toml] * len(constrained_instructions),
        dataset_prefix="tezos_constrained",
    )

    extracted_dir = extract_hdf5_to_task_paths(hdf5_path)
    upload_tasks_to_hf(extracted_dir, "DCAgent/exp-syh-tezos-askllm-constrained")

    return extracted_dir


if __name__ == "__main__":
    print(main())
