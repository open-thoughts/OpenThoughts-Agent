#!/usr/bin/env python3
"""
Generate code_feedback dataset from neulab/agent-data-collection in sandboxes style
Always upsamples to 10k tasks, then subsamples to 10k (in case dataset is larger).
"""

from typing import List
from datasets import load_dataset
from data.commons import generate_tasks_from_questions, upsample_tasks_directory, subsample_tasks_directory, upload_tasks_to_hf, upload_traces_to_hf
from scripts.harbor.run_and_export_traces import run_dataset_to_traces


def extract_code_feedback_questions() -> List[str]:
    """
    Load code_feedback subset directly from raw jsonl file and extract questions.
    """
    print("Loading code_feedback dataset from raw jsonl...")
    ds = load_dataset("json", data_files="hf://datasets/neulab/agent-data-collection/code_feedback/full_raw.jsonl")
    dataset = ds["train"]
    print(f"Loaded {len(dataset)} examples")

    questions = []
    for item in dataset:
        messages = item.get("messages", [])
        if messages:
            first_msg = messages[0]
            content = first_msg.get("content", "") or first_msg.get("value", "")
            questions.append(content)

    print(f"Extracted {len(questions)} questions from code_feedback")
    return questions


def main() -> None:
    # Extract questions from the subset (using custom loader)
    questions = extract_code_feedback_questions()
    
    # Generate tasks from questions
    print("Generating tasks from questions...")
    tasks_dir = generate_tasks_from_questions(questions, dataset_prefix="code-feedback")
    
    # Upsample to 10k (if less than 10k)
    print("Upsampling to 10k tasks...")
    tasks_dir = upsample_tasks_directory(
        source_dir=tasks_dir,
        num_samples=10_000,
        dataset_prefix="code-feedback",
    )
    
    # Subsample to 10k (in case more than 10k)
    print("Subsampling to 10k tasks...")
    tasks_dir = subsample_tasks_directory(
        source_dir=tasks_dir,
        num_samples=10_000,
        dataset_prefix="code-feedback",
    )
    
    # Upload tasks to HF
    print("Uploading tasks to HuggingFace...")
    upload_tasks_to_hf(
        dataset_path=tasks_dir,
        repo_id="DCAgent/neulab-code-feedback-sandboxes",
        commit_message="Upload code_feedback sandboxes (10k tasks)",
    )
    
    # Generate and upload traces
    # print("Generating traces...")
    # hf_dataset = run_dataset_to_traces(
    #     tasks_dir,
    #     model_name="gpt-5-nano-2025-08-07",
    #     agent_name="terminus-2",
    #     n_concurrent=256,
    #     agent_kwargs={"max_episodes": 8},
    # )
    
    # print("Uploading traces...")
    # upload_traces_to_hf(
    #     hf_dataset,
    #     "DCAgent/neulab-code-feedback-sandboxes-traces-terminus-2",
    #     "SFT"
    # )
    
    print("Done!")


if __name__ == "__main__":
    main()
