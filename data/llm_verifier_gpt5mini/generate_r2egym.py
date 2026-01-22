#!/usr/bin/env python3
"""
Generate LLM Verifier dataset for R2E-Gym using gpt-5-mini.

Usage:
    python -m data.llm_verifier_gpt5mini.generate_r2egym
"""

from datasets import load_dataset

from data.commons import (
    generate_tasks_from_questions_hdf5,
    upload_tasks_to_hf,
    extract_hdf5_to_task_paths,
)
from data.llm_verifier.utils import add_llm_verifier_tests_to_questions

# Model for this variant
MODEL = "gpt-5-mini"


def load_r2egym_instances() -> list[str]:
    """
    Load R2E-Gym instances from HuggingFace dataset.

    Returns:
        List of problem statements/instructions from R2E-Gym
    """
    print("Loading R2E-Gym dataset from HuggingFace...")
    ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")

    questions = []
    for row in ds:
        # R2E-Gym contains problem statements with context
        problem_statement = row.get("problem_statement", "")
        if problem_statement and problem_statement.strip():
            questions.append(problem_statement)

    print(f"Loaded {len(questions)} instances from R2E-Gym")
    return questions


def main() -> None:
    """Main function - generates LLM verifier dataset for R2E-Gym."""
    print(f"Generating R2E-Gym LLM verifier dataset with model={MODEL}")

    # Load R2E-Gym instances
    questions = load_r2egym_instances()

    # Add LLM verifier tests with the specified model
    questions_with_tests = add_llm_verifier_tests_to_questions(questions, model=MODEL)

    # Generate tasks to HDF5 for faster operations
    model_suffix = MODEL.replace("-", "_")
    dataset_prefix = f"llm_verifier_r2egym_{model_suffix}"
    hdf5_path = generate_tasks_from_questions_hdf5(
        questions_with_tests,
        dataset_prefix=dataset_prefix
    )

    print(f"Generated HDF5 file: {hdf5_path}")

    # Extract HDF5 to task directories for HuggingFace upload
    task_dir = extract_hdf5_to_task_paths(hdf5_path)

    # Upload to HuggingFace with model name in repo
    repo_name = f"DCAgent/exp_llmve_llm-verifier-r2egym-{model_suffix}"
    upload_tasks_to_hf(task_dir, repo_name)

    print(f"Successfully uploaded to {repo_name}")


if __name__ == "__main__":
    main()
