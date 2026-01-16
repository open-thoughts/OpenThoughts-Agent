#!/usr/bin/env python3
"""
Random Filter strategy for SWE-smith dataset
"""

import random

from data.commons import select_top_n_by_score
from data.swesmith_filtering.load_data import load_swesmith_data, filter_and_upload_swesmith


def main() -> None:
    random.seed(42)
    questions, rows = load_swesmith_data()

    random_scores = [random.random() for _ in questions]

    filtered_questions = select_top_n_by_score(questions, random_scores, n=10_000, descending=True)
    filter_and_upload_swesmith(questions, rows, filtered_questions, "DCAgent/exp-gfi-swesmith-random-filtered-10K")


if __name__ == "__main__":
    main()
