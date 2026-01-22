#!/usr/bin/env python3
"""
Generate LLM Verifier dataset for SWE-smith.

This script loads SWE-smith instances from HuggingFace and adds LLM verifier tests.

Usage:
    python -m data.llm_verifier.generate_swesmith
"""

from datasets import load_dataset

from data.commons import (
    generate_tasks_from_questions_hdf5,
    upload_tasks_to_hf,
    extract_hdf5_to_task_paths,
)
from data.llm_verifier.utils import add_llm_verifier_tests_to_questions


def load_swesmith_instances() -> list[str]:
    """
    Load SWE-smith instances from HuggingFace dataset.

    Returns:
        List of problem statements from SWE-smith
    """
    print("Loading SWE-smith dataset from HuggingFace...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")

    questions = []
    for row in ds:
        problem_statement = row.get("problem_statement", "")
        if problem_statement and problem_statement.strip():
            questions.append(problem_statement)

    print(f"Loaded {len(questions)} instances from SWE-smith")
    return questions


def main(model: str = "gpt-5-nano") -> None:
    """
    Main function - generates LLM verifier dataset for SWE-smith.

    Args:
        model: The OpenAI model to use for judging (default: gpt-5-nano)
    """
    print(f"Generating SWE-smith LLM verifier dataset with model={model}")

    # Load SWE-smith instances
    questions = load_swesmith_instances()

    # Add LLM verifier tests
    questions_with_tests = add_llm_verifier_tests_to_questions(questions, model=model)

    # Generate tasks to HDF5 for faster operations
    dataset_prefix = "llm_verifier_swesmith"
    hdf5_path = generate_tasks_from_questions_hdf5(
        questions_with_tests,
        dataset_prefix=dataset_prefix
    )

    print(f"Generated HDF5 file: {hdf5_path}")

    # Extract HDF5 to task directories for HuggingFace upload
    task_dir = extract_hdf5_to_task_paths(hdf5_path)

    # Upload to HuggingFace
    repo_name = "DCAgent/exp_llmve_llm-verifier-swesmith"
    upload_tasks_to_hf(task_dir, repo_name)

    print(f"Successfully uploaded to {repo_name}")


if __name__ == "__main__":
    main()
