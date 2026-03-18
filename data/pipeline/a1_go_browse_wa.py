#!/usr/bin/env python3
"""
Generate go-browse-wa dataset from neulab/agent-data-collection in sandboxes style
Always upsamples to 10k tasks, then subsamples to 10k (in case dataset is larger).
"""

from typing import List
from datasets import load_dataset
from data.commons import generate_tasks_from_questions, upsample_tasks_directory, subsample_tasks_directory, upload_tasks_to_hf, upload_traces_to_hf
from scripts.harbor.run_and_export_traces import run_dataset_to_traces


def extract_go_browse_wa_questions() -> List[str]:
    """
    Load go-browse-wa subset directly from raw jsonl file and extract questions.
    """
    print("Loading go-browse-wa dataset from raw jsonl...")
    ds = load_dataset("json", data_files="hf://datasets/neulab/agent-data-collection/go-browse-wa/full_raw.jsonl")
    dataset = ds["train"]
    print(f"Loaded {len(dataset)} examples")

    questions = []
    for item in dataset:
        traj_data = item.get("traj_data", {})
        if isinstance(traj_data, dict):
            goal = traj_data.get("goal", "")
            questions.append(goal)

    print(f"Extracted {len(questions)} questions from go-browse-wa")
    return questions


def main() -> None:
    # Extract questions from the subset (using custom loader)
    questions = extract_go_browse_wa_questions()
    
    # Generate tasks from questions
    print("Generating tasks from questions...")
    tasks_dir = generate_tasks_from_questions(questions, dataset_prefix="go-browse-wa")
    
    # Upsample to 10k (if less than 10k)
    print("Upsampling to 10k tasks...")
    tasks_dir = upsample_tasks_directory(
        source_dir=tasks_dir,
        num_samples=10_000,
        dataset_prefix="go-browse-wa",
    )
    
    # Subsample to 10k (in case more than 10k)
    print("Subsampling to 10k tasks...")
    tasks_dir = subsample_tasks_directory(
        source_dir=tasks_dir,
        num_samples=10_000,
        dataset_prefix="go-browse-wa",
    )
    
    # Upload tasks to HF
    print("Uploading tasks to HuggingFace...")
    upload_tasks_to_hf(
        dataset_path=tasks_dir,
        repo_id="DCAgent/neulab-go-browse-wa-sandboxes",
        commit_message="Upload go-browse-wa sandboxes (10k tasks)",
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
    #     "DCAgent/neulab-go-browse-wa-sandboxes-traces-terminus-2",
    #     "SFT"
    # )
    
    print("Done!")


if __name__ == "__main__":
    main()
