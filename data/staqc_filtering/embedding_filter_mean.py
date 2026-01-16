#!/usr/bin/env python3
"""
Embedding Filter Mean strategy for Stack Overflow dataset
"""

from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, select_top_n_by_score, download_dcagent_dev_set_instructions
from data.code_contests.generate import load_code_contests_questions
from data.freelancer_filtering.embedding_filter_mean import compute_embeddings, calculate_embedding_scores
from data.staqc_filtering.load_data import load_staqc_questions


def main() -> None:
    questions = load_staqc_questions()

    positive_samples = download_dcagent_dev_set_instructions()
    negative_samples = load_code_contests_questions()

    question_embeddings = compute_embeddings(questions)
    positive_embeddings = compute_embeddings(positive_samples)
    negative_embeddings = compute_embeddings(negative_samples)

    scores = calculate_embedding_scores(questions, question_embeddings, positive_embeddings, negative_embeddings)

    filtered_questions = select_top_n_by_score(questions, scores, n=10_000, descending=True)
    filtered_questions = list(dict.fromkeys(filtered_questions))[:10_000]

    final_dataset_dir = generate_tasks_from_questions(filtered_questions, "staqc_embedding_mean_filtered")
    upload_tasks_to_hf(final_dataset_dir, "DCAgent/exp-gfi-staqc-embedding-mean-filtered-10K")


if __name__ == "__main__":
    main()
