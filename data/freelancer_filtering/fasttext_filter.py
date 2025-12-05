#!/usr/bin/env python3
"""
FastText filtering strategy for Freelancer dataset
Uses FastText embeddings to score and rank questions
"""

import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from data.freelancer.generate import scrape_freelancer_projects
from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, subsample_tasks_directory, select_top_n_by_score, download_dcagent_dev_set_instructions
from data.code_contests.generate import load_code_contests_questions
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.commons import upload_traces_to_hf

def calculate_fasttext_scores(questions: List[str], fasttext_model, positive_samples: List[str], negative_samples: List[str]) -> List[float]:
    """Calculate FastText-based scores for questions using similarity to positive/negative samples"""
    print("Calculating FastText scores...")
    
    # Get embeddings for positive and negative samples
    positive_embeddings = [fasttext_model.get_sentence_vector(sample.replace('\n', ' ')) for sample in positive_samples]
    negative_embeddings = [fasttext_model.get_sentence_vector(sample.replace('\n', ' ')) for sample in negative_samples]
    
    scores = []
    for question in questions:
        # Get sentence vector from FastText model
        question_embedding = fasttext_model.get_sentence_vector(question.replace('\n', ' '))
        
        # Calculate mean similarity with positive samples
        positive_similarities = [np.dot(question_embedding, pos_emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(pos_emb) + 1e-8) 
                                for pos_emb in positive_embeddings]
        mean_positive_score = np.mean(positive_similarities)
        
        # Calculate mean similarity with negative samples
        negative_similarities = [np.dot(question_embedding, neg_emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(neg_emb) + 1e-8)
                                for neg_emb in negative_embeddings]
        mean_negative_score = np.mean(negative_similarities)
        
        # Score is the difference between positive and negative similarities
        score = mean_positive_score - mean_negative_score
        scores.append(score)
    
    return scores


def main(fasttext_model=None) -> None:
    """Main function - FastText filtering strategy

    Args:
        fasttext_model: Pre-trained FastText model (optional).
                       If None, will train a new model.
                       Example: fasttext.load_model('path/to/model.bin')
    """
    # Get freelancer project descriptions
    questions = scrape_freelancer_projects()

    # Create positive and negative samples
    positive_samples = download_dcagent_dev_set_instructions()
    negative_samples = load_code_contests_questions()

    # Train or use provided FastText model
    if fasttext_model is None:
        print("No model provided. Training a new FastText model...")
        from data.freelancer_filtering.train_fasttext import main as train_model
        fasttext_model = train_model()
        print("Model training completed.")
    
    # Calculate FastText scores using similarity to positive/negative samples
    scores = calculate_fasttext_scores(questions, fasttext_model, positive_samples, negative_samples)

    # Select top questions by FastText score
    filtered_questions = select_top_n_by_score(
        questions,
        scores,
        n=1_000,  
        descending=True      # Higher scores are better
    )
    upsampled_questions = filtered_questions * 10
    final_dataset_dir = generate_tasks_from_questions(upsampled_questions, "freelancer_fasttext_filtered")
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/freelancer-fasttext-filtered-sandboxes")
    hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    upload_traces_to_hf(hf_dataset, "DCAgent/freelancer-fasttext-instruction-filter", "SFT")


if __name__ == "__main__":
    # Train a new model and use it for filtering
    main(fasttext_model=None)