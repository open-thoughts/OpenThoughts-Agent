#!/usr/bin/env python3
"""
AskLLM filtering strategy for SWE-smith dataset
"""

from data.commons import select_top_n_by_score
from data.freelancer_filtering.ask_llm_filter import rate_questions_with_llm
from data.swesmith_filtering.load_data import load_swesmith_data, filter_and_upload_swesmith


def main() -> None:
    questions, rows = load_swesmith_data()
    ratings = rate_questions_with_llm(questions, model="gpt-4o-mini")

    filtered_questions = select_top_n_by_score(questions, ratings, n=10_000, descending=True)
    filter_and_upload_swesmith(questions, rows, filtered_questions, "DCAgent/exp-gfi-swesmith-askllm-filtered-10K")


if __name__ == "__main__":
    main()
