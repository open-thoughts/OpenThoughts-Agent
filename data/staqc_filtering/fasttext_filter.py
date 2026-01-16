#!/usr/bin/env python3
"""
FastText filtering strategy for Stack Overflow dataset
"""

from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, select_top_n_by_score, download_dcagent_dev_set_instructions
from data.code_contests.generate import load_code_contests_questions
from data.freelancer_filtering.fasttext_filter import calculate_fasttext_scores
from data.staqc_filtering.load_data import load_staqc_questions


def main(fasttext_model=None) -> None:
    questions = load_staqc_questions()

    positive_samples = download_dcagent_dev_set_instructions()
    negative_samples = load_code_contests_questions()

    if fasttext_model is None:
        from data.freelancer_filtering.train_fasttext import main as train_model
        fasttext_model = train_model()

    scores = calculate_fasttext_scores(questions, fasttext_model, positive_samples, negative_samples)

    filtered_questions = select_top_n_by_score(questions, scores, n=10_000, descending=True)
    filtered_questions = list(dict.fromkeys(filtered_questions))[:10_000]

    final_dataset_dir = generate_tasks_from_questions(filtered_questions, "staqc_fasttext_filtered")
    upload_tasks_to_hf(final_dataset_dir, "DCAgent/exp-gfi-staqc-fasttext-filtered-10K")


if __name__ == "__main__":
    main(fasttext_model=None)
