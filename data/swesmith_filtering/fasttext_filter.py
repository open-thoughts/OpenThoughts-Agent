#!/usr/bin/env python3
"""
FastText filtering strategy for SWE-smith dataset
"""

from data.commons import select_top_n_by_score, download_dcagent_dev_set_instructions
from data.code_contests.generate import load_code_contests_questions
from data.freelancer_filtering.fasttext_filter import calculate_fasttext_scores
from data.swesmith_filtering.load_data import load_swesmith_data, filter_and_upload_swesmith


def main(fasttext_model=None) -> None:
    questions, rows = load_swesmith_data()

    positive_samples = download_dcagent_dev_set_instructions()
    negative_samples = load_code_contests_questions()

    if fasttext_model is None:
        from data.freelancer_filtering.train_fasttext import main as train_model
        fasttext_model = train_model()

    scores = calculate_fasttext_scores(questions, fasttext_model, positive_samples, negative_samples)

    filtered_questions = select_top_n_by_score(questions, scores, n=10_000, descending=True)
    filter_and_upload_swesmith(questions, rows, filtered_questions, "DCAgent/exp-gfi-swesmith-fasttext-filtered-10K")


if __name__ == "__main__":
    main(fasttext_model=None)
