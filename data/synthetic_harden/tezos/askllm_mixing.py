
#!/usr/bin/env python3
"""
Generate harder versions of tezos tasks by mixing with StackOverflow questions using an LLM.
"""

import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data.synthetic_harden.utils import mix_questions
from data.stackexchange.generate_codereview import download_and_extract_dataset, parse_posts_xml, extract_questions_from_data
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf, create_standard_dockerfile, create_standard_task_toml, upsample_list


TEZOS_URL = "https://archive.org/download/stackexchange/tezos.stackexchange.com.7z"
STACKOVERFLOW_URL = "https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z"


def main() -> str:
    random.seed(42)

    # Load Tezos questions
    print("Loading Tezos questions...")
    tezos_posts_xml = download_and_extract_dataset(TEZOS_URL)
    tezos_data = parse_posts_xml(tezos_posts_xml)
    tezos_questions = extract_questions_from_data(tezos_data)
    print(f"Loaded {len(tezos_questions)} Tezos questions")

    # Load StackOverflow questions (with limit to avoid memory issues)
    print("Loading StackOverflow questions...")
    so_posts_xml = download_and_extract_dataset(STACKOVERFLOW_URL)
    so_data = parse_posts_xml(so_posts_xml, limit=10_000)
    so_questions = extract_questions_from_data(so_data)
    print(f"Loaded {len(so_questions)} StackOverflow questions")

    # Shuffle both lists for random pairing
    random.shuffle(tezos_questions)
    random.shuffle(so_questions)

    # Upsample to target 10,000 mixed questions
    tezos_questions = upsample_list(tezos_questions, 10_000)
    so_questions = upsample_list(so_questions, 10_000)

    # Mix the questions using the LLM
    print("Mixing questions with LLM...")
    mixed_instructions = mix_questions(
        instructions_a=tezos_questions,
        instructions_b=so_questions,
        source_a="Tezos StackExchange",
        source_b="StackOverflow",
        model="gpt-5-nano",
    )
    print(f"Generated {len(mixed_instructions)} mixed instructions")

    dockerfile = create_standard_dockerfile()
    task_toml = create_standard_task_toml()

    hdf5_path = generate_tasks_to_hdf5(
        instructions=mixed_instructions,
        metadata=[{}] * len(mixed_instructions),
        dockerfiles=[dockerfile] * len(mixed_instructions),
        task_toml=[task_toml] * len(mixed_instructions),
        dataset_prefix="tezos_stackoverflow_mixed",
    )

    extracted_dir = extract_hdf5_to_task_paths(hdf5_path)
    upload_tasks_to_hf(extracted_dir, "DCAgent/exp-syh-tezos-stackoverflow-mixed")

    return extracted_dir


if __name__ == "__main__":
    print(main())
