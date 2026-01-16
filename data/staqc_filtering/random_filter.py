#!/usr/bin/env python3
"""
Random Filter strategy for Stack Overflow dataset
"""

import random

from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, select_top_n_by_score
from data.staqc_filtering.load_data import load_staqc_questions


def main() -> None:
    random.seed(42)
    questions = load_staqc_questions()

    random_scores = [random.random() for _ in questions]

    filtered_questions = select_top_n_by_score(questions, random_scores, n=10_000, descending=True)
    filtered_questions = list(dict.fromkeys(filtered_questions))[:10_000]

    final_dataset_dir = generate_tasks_from_questions(filtered_questions, "staqc_random_filtered")
    upload_tasks_to_hf(final_dataset_dir, "DCAgent/exp-gfi-staqc-random-filtered-10K")


if __name__ == "__main__":
    main()
