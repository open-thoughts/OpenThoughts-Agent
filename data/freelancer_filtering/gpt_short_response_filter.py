#!/usr/bin/env python3
"""
GPT Short Response Filtering strategy for Freelancer dataset
Filters for questions that generate concise, efficient responses
"""

import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from datasets import Dataset

sys.path.append(str(Path(__file__).parent.parent.parent))
from data.freelancer.generate import scrape_freelancer_projects
from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, subsample_tasks_directory, select_top_n_by_score
from data.gcs_cache import gcs_cache
from data.completions import run_completions
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.commons import upload_traces_to_hf


@gcs_cache()
def generate_gpt_responses(questions: List[str], model: str = "gpt-5-nano-2025-08-07") -> List[str]:
    """Generate GPT responses for each question
    
    Args:
        questions: List of questions/instructions to respond to
        model: Model to use for generation
    
    Returns:
        List of responses
    """
    print(f"Generating responses for {len(questions)} questions with {model}...")
    
    # Create dataset
    dataset = Dataset.from_dict({"question": questions})
    
    # Define response generation prompt - encourage concise responses
    map_config = {
        "user_message": """{{question}}""",
        "system_message": "You are a helpful AI assistant. Provide clear, direct, and concise responses. Be efficient with your words while still being helpful.",
        "output_column": "gpt_response"
    }
    
    # Run completions
    result = run_completions(
        dataset=dataset,
        model=model,
        map_type="chat",
        map_config=map_config,
        max_requests_per_minute=500
    )
    
    # Extract responses
    responses = result.dataset['gpt_response']
    return responses


def calculate_response_lengths(responses: List[str]) -> List[int]:
    """Calculate lengths of responses and print statistics
    
    Args:
        responses: List of response strings
    
    Returns:
        List of response lengths
    """
    response_lengths = [len(response) for response in responses]
    
    # Print statistics
    print(f"Response length statistics:")
    print(f"  Min: {min(response_lengths)} chars")
    print(f"  Max: {max(response_lengths)} chars")
    print(f"  Mean: {np.mean(response_lengths):.1f} chars")
    print(f"  Median: {np.median(response_lengths):.1f} chars")
    
    return response_lengths




def main() -> None:
    """Main function - GPT Short Response Filtering strategy"""
    # Get freelancer project descriptions
    questions = scrape_freelancer_projects()

    # Generate GPT responses and calculate their lengths
    responses = generate_gpt_responses(questions, model="gpt-5-nano-2025-08-07")
    response_lengths = calculate_response_lengths(responses)

    # Select top questions by SHORT response length using common function directly in main
    filtered_questions = select_top_n_by_score(
        questions,
        response_lengths,
        n=1_000,  # Keep top 40% shortest
        descending=False     # SHORTER responses (lower scores) are better
    )

    upsampled_questions = filtered_questions * 10
    final_dataset_dir = generate_tasks_from_questions(upsampled_questions, "freelancer_short_response_filtered")
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/freelancer-short-response-filtered-sandboxes")
    hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    upload_traces_to_hf(hf_dataset, "DCAgent/freelancer-short-instruction-filter", "SFT")

if __name__ == "__main__":
    main()