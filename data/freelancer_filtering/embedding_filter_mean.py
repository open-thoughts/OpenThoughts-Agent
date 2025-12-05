#!/usr/bin/env python3
"""
Embedding Filter Mean strategy for Freelancer dataset
Filters based on mean similarity to positive/negative embeddings
"""

import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from data.freelancer.generate import scrape_freelancer_projects
from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, subsample_tasks_directory, select_top_n_by_score, download_dcagent_dev_set_instructions
from data.code_contests.generate import load_code_contests_questions
from data.gcs_cache import gcs_cache
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.commons import upload_traces_to_hf


@gcs_cache()
def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Compute embeddings for a list of texts"""
    print(f"Computing embeddings for {len(texts)} texts...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

@gcs_cache()
def calculate_embedding_scores(
    questions: List[str],
    question_embeddings: np.ndarray,
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray
) -> List[float]:
    """Calculate embedding scores based on mean similarity"""
    print("Calculating embedding scores...")
    
    # Calculate mean similarity with positive embeddings
    positive_similarities = np.dot(question_embeddings, positive_embeddings.T)
    mean_positive_scores = np.mean(positive_similarities, axis=1)
    
    # Calculate mean similarity with negative embeddings
    negative_similarities = np.dot(question_embeddings, negative_embeddings.T)
    mean_negative_scores = np.mean(negative_similarities, axis=1)
    
    # Calculate difference scores
    difference_scores = mean_positive_scores - mean_negative_scores
    
    return difference_scores.tolist()




def main() -> None:
    """Main function - Embedding Filter Mean strategy
    
    Args:
    """
    # Get freelancer project descriptions
    questions = scrape_freelancer_projects()

    # Create positive and negative samples
    positive_samples = download_dcagent_dev_set_instructions()
    negative_samples = load_code_contests_questions()

    # Compute embeddings
    question_embeddings = compute_embeddings(questions)
    positive_embeddings = compute_embeddings(positive_samples)
    negative_embeddings = compute_embeddings(negative_samples)

    # Calculate scores and filter
    scores = calculate_embedding_scores(
        questions,
        question_embeddings,
        positive_embeddings,
        negative_embeddings
    )
    # Select top questions based on embedding scores using common function
    filtered_questions = select_top_n_by_score(
        questions,
        scores,
        n=1_000,
        descending=True  # Higher scores (more similar to positive) are better
    )

    # Generate and upload filtered dataset
    upsampled_questions = filtered_questions * 10
    final_dataset_dir = generate_tasks_from_questions(upsampled_questions, "freelancer_embedding_mean_filtered")
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/freelancer-embedding-mean-instruction-filter-sandboxes")
    hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    upload_traces_to_hf(hf_dataset, "DCAgent/freelancer-embedding-mean-instruction-filter", "SFT")


if __name__ == "__main__":
    main()