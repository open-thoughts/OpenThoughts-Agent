#!/usr/bin/env python3
"""
GPT Long Response Filtering strategy for Freelancer dataset
Filters for questions that generate detailed, comprehensive responses
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from data.freelancer.generate import scrape_freelancer_projects
from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, subsample_tasks_directory, select_top_n_by_score
from data.freelancer_filtering.gpt_short_response_filter import generate_gpt_responses, calculate_response_lengths
from data.commons import upload_tasks_to_hf, subsample_tasks_directory, upload_traces_to_hf
from scripts.harbor.run_and_export_traces import run_dataset_to_traces



def main() -> None:
    """Main function - GPT Long Response Filtering strategy"""
    # Get freelancer project descriptions
    questions = scrape_freelancer_projects()

    print(f"Processing {len(questions)} questions...")

    # Generate GPT responses and calculate their lengths
    responses = generate_gpt_responses(questions, model="gpt-5-nano-2025-08-07")
    response_lengths = calculate_response_lengths(responses)

    # Select top questions by LONG response length using common function directly in main
    filtered_questions = select_top_n_by_score(
        questions,
        response_lengths,
        n=1_000,  # Keep top 40% longest
        descending=True      # LONGER responses (higher scores) are better
    )

    upsampled_questions = filtered_questions * 10
    final_dataset_dir = generate_tasks_from_questions(upsampled_questions, "freelancer_long_response_filtered")
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/freelancer-long-response-filtered-sandboxes")
    hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    upload_traces_to_hf(hf_dataset, "DCAgent/freelancer-long-instruction-filter", "SFT")

if __name__ == "__main__":
    main()
