#!/usr/bin/env python3
"""
Random Filter strategy for Freelancer dataset
Randomly selects a subset of questions
"""

import sys
import random
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent.parent))
from data.freelancer.generate import scrape_freelancer_projects
from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, subsample_tasks_directory, select_top_n_by_score
from data.gcs_cache import gcs_cache
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.commons import upload_traces_to_hf


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    print(f"Random seed set to {seed}")


def random_sample_questions(questions: List[str], sample_percentage: float = 0.3) -> List[str]:
    """Randomly sample a percentage of questions"""
    # Generate random scores for each question
    random_scores = [random.random() for _ in questions]
    
    # Use common selection function with random scores
    sampled_questions = select_top_n_by_score(
        questions,
        random_scores,
        top_percentage=sample_percentage,
        descending=True  # Doesn't matter for random scores
    )
    
    print(f"Randomly sampled {len(sampled_questions)}/{len(questions)} questions ({sample_percentage*100:.1f}%)")
    return sampled_questions


def stratified_random_sample_by_domain(
    conversations: List[dict],
    questions: List[str],
    sample_percentage: float = 0.3
) -> List[str]:
    """Randomly sample questions with stratification by domain"""
    # Group questions by domain
    domain_questions = {}
    
    for conversation, question in zip(conversations, questions):
        domain = conversation['domain']
        if domain not in domain_questions:
            domain_questions[domain] = []
        domain_questions[domain].append(question)
    
    # Sample from each domain
    sampled_questions = []
    for domain, domain_q_list in domain_questions.items():
        num_to_sample = max(1, int(len(domain_q_list) * sample_percentage))
        sampled = random.sample(domain_q_list, min(num_to_sample, len(domain_q_list)))
        sampled_questions.extend(sampled)
        print(f"  Domain {domain}: Sampled {len(sampled)}/{len(domain_q_list)} questions")
    
    # Shuffle final list
    random.shuffle(sampled_questions)
    
    print(f"Total: Randomly sampled {len(sampled_questions)}/{len(questions)} questions")
    return sampled_questions


def weighted_random_sample(
    questions: List[str],
    weights: List[float] = None,
    sample_percentage: float = 0.3
) -> List[str]:
    """Randomly sample questions with optional weights"""
    if weights is None:
        # Use uniform weights if none provided
        weights = [1.0] * len(questions)
    
    num_to_sample = int(len(questions) * sample_percentage)
    
    # Weighted sampling without replacement
    sampled_questions = random.choices(
        questions,
        weights=weights,
        k=num_to_sample
    )
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sampled = []
    for q in sampled_questions:
        if q not in seen:
            seen.add(q)
            unique_sampled.append(q)
    
    print(f"Weighted random sample: {len(unique_sampled)}/{len(questions)} questions")
    return unique_sampled


def create_diversity_weights(questions: List[str]) -> List[float]:
    """Create weights based on question diversity (length variation)"""
    lengths = [len(q) for q in questions]
    mean_length = sum(lengths) / len(lengths)
    
    # Give higher weight to questions with lengths different from mean
    weights = []
    for length in lengths:
        deviation = abs(length - mean_length)
        weight = 1.0 + (deviation / mean_length)  # Higher weight for more deviation
        weights.append(weight)
    
    return weights


def main() -> None:
    """Main function - Random Filter strategy"""
    # Set seed for reproducibility
    set_random_seed(42)

    # Get freelancer project descriptions
    questions = scrape_freelancer_projects()

    # Generate random scores for each question
    random_scores = [random.random() for _ in questions]

    # Use common selection function with random scores directly in main
    filtered_questions = select_top_n_by_score(
        questions,
        random_scores,
        n=1_000,  # Keep 30% of questions
        descending=True      # Doesn't matter for random scores
    )

    # Generate and upload filtered dataset
    upsampled_questions = filtered_questions * 10
    final_dataset_dir = generate_tasks_from_questions(upsampled_questions, "freelancer_random_filtered")
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/freelancer-random-instruction-filter-sandboxes")
    hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    upload_traces_to_hf(hf_dataset, "DCAgent/freelancer-random-instruction-filter-traces-terminus-2", "SFT")

if __name__ == "__main__":
    main()