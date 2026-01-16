#!/usr/bin/env python3
"""
Generate harder versions of tezos tasks using an LLM.
"""

import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data.synthetic_harden.utils import harden_instructions
from data.stackexchange.generate_codereview import download_and_extract_dataset, parse_posts_xml, extract_questions_from_data
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, create_standard_dockerfile, create_standard_task_toml, upsample_list

def main() -> str:
    random.seed(42)

    posts_xml_path = download_and_extract_dataset("https://archive.org/download/stackexchange/tezos.stackexchange.com.7z")
    questions_data = parse_posts_xml(posts_xml_path)
    instructions = upsample_list(extract_questions_from_data(questions_data), 10_000)
    hardened_instructions = harden_instructions(instructions, model="gpt-4o-mini")

    dockerfile = create_standard_dockerfile()
    task_toml = create_standard_task_toml()

    hdf5_path = generate_tasks_to_hdf5(
        instructions=hardened_instructions,
        metadata=[{}] * len(hardened_instructions),
        dockerfiles=[dockerfile] * len(hardened_instructions),
        task_toml=[task_toml] * len(hardened_instructions),
        dataset_prefix="tezos_hardened",
    )

    extracted_dir = extract_hdf5_to_task_paths(hdf5_path)
    upload_tasks_to_hf(extracted_dir, "DCAgent/exp-syh-tezos-askllm-hardened")

    return extracted_dir


if __name__ == "__main__":
    print(main())
