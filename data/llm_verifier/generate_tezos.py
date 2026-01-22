#!/usr/bin/env python3
"""
Generate LLM Verifier dataset for Tezos StackExchange.

This script loads Tezos questions from the StackExchange archive and adds LLM verifier
tests with a configurable judge model.

Usage:
    python -m data.llm_verifier.generate_tezos
"""

from data.stackexchange.generate_codereview import (
    download_and_extract_dataset,
    parse_posts_xml,
    extract_questions_from_data,
)
from data.commons import (
    generate_tasks_from_questions_hdf5,
    upload_tasks_to_hf,
    extract_hdf5_to_task_paths,
)
from data.llm_verifier.utils import add_llm_verifier_tests_to_questions

# Tezos StackExchange archive URL
TEZOS_URL = "https://archive.org/download/stackexchange/tezos.stackexchange.com.7z"


def main(model: str = "gpt-5-nano") -> None:
    """
    Main function - generates LLM verifier dataset for Tezos StackExchange.

    Args:
        model: The OpenAI model to use for judging (default: gpt-5-nano)
    """
    print(f"Generating Tezos LLM verifier dataset with model={model}")

    # Download and extract Tezos StackExchange data
    posts_xml_path = download_and_extract_dataset(TEZOS_URL)

    # Parse posts and extract questions
    questions_data = parse_posts_xml(posts_xml_path)
    questions = extract_questions_from_data(questions_data)

    print(f"Loaded {len(questions)} questions from Tezos StackExchange")

    # Add LLM verifier tests with the specified model
    questions_with_tests = add_llm_verifier_tests_to_questions(questions, model=model)

    # Generate tasks to HDF5 for faster operations
    model_suffix = model.replace("-", "_")
    dataset_prefix = f"llm_verifier_tezos_{model_suffix}"
    hdf5_path = generate_tasks_from_questions_hdf5(
        questions_with_tests,
        dataset_prefix=dataset_prefix
    )

    print(f"Generated HDF5 file: {hdf5_path}")

    # Extract HDF5 to task directories for HuggingFace upload
    task_dir = extract_hdf5_to_task_paths(hdf5_path)

    # Upload to HuggingFace with model name in repo
    repo_name = f"DCAgent/exp_llmve_llm-verifier-tezos-{model_suffix}"
    upload_tasks_to_hf(task_dir, repo_name)

    print(f"Successfully uploaded to {repo_name}")


if __name__ == "__main__":
    main()
