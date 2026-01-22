#!/usr/bin/env python3
"""
Generate LLM Verifier dataset for StackOverflow.

This script loads StackOverflow questions from the StackExchange archive and adds
LLM verifier tests.

Usage:
    python -m data.llm_verifier.generate_stackoverflow
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

# StackOverflow archive URL
STACKOVERFLOW_URL = "https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z"


def main(model: str = "gpt-5-nano") -> None:
    """
    Main function - generates LLM verifier dataset for StackOverflow.

    Args:
        model: The OpenAI model to use for judging (default: gpt-5-nano)
    """
    print(f"Generating StackOverflow LLM verifier dataset with model={model}")

    # Download and extract StackOverflow data
    posts_xml_path = download_and_extract_dataset(STACKOVERFLOW_URL)

    # Parse posts and extract questions (limit to 10k for manageability)
    questions_data = parse_posts_xml(posts_xml_path, limit=10_000)
    questions = extract_questions_from_data(questions_data)

    print(f"Loaded {len(questions)} questions from StackOverflow")

    # Add LLM verifier tests
    questions_with_tests = add_llm_verifier_tests_to_questions(questions, model=model)

    # Generate tasks to HDF5 for faster operations
    dataset_prefix = "llm_verifier_stackoverflow"
    hdf5_path = generate_tasks_from_questions_hdf5(
        questions_with_tests,
        dataset_prefix=dataset_prefix
    )

    print(f"Generated HDF5 file: {hdf5_path}")

    # Extract HDF5 to task directories for HuggingFace upload
    task_dir = extract_hdf5_to_task_paths(hdf5_path)

    # Upload to HuggingFace
    repo_name = "DCAgent/exp_llmve_llm-verifier-stackoverflow"
    upload_tasks_to_hf(task_dir, repo_name)

    print(f"Successfully uploaded to {repo_name}")


if __name__ == "__main__":
    main()
