#!/usr/bin/env python3
"""
Embedding Filter Mean strategy for SWE-smith dataset
"""

from data.commons import select_top_n_by_score, download_dcagent_dev_set_instructions
from data.code_contests.generate import load_code_contests_questions
from data.freelancer_filtering.embedding_filter_mean import compute_embeddings, calculate_embedding_scores
from data.swesmith_filtering.load_data import load_swesmith_data, filter_and_upload_swesmith


def main() -> None:
    questions, rows = load_swesmith_data()

    positive_samples = download_dcagent_dev_set_instructions()
    negative_samples = load_code_contests_questions()

    question_embeddings = compute_embeddings(questions)
    positive_embeddings = compute_embeddings(positive_samples)
    negative_embeddings = compute_embeddings(negative_samples)

    scores = calculate_embedding_scores(questions, question_embeddings, positive_embeddings, negative_embeddings)

    filtered_questions = select_top_n_by_score(questions, scores, n=10_000, descending=True)
    filter_and_upload_swesmith(questions, rows, filtered_questions, "DCAgent/exp-gfi-swesmith-embedding-mean-filtered-10K")


if __name__ == "__main__":
    main()
