#!/usr/bin/env python3
"""
GPT Short Response Filtering strategy for SWE-smith dataset
"""

from data.commons import select_top_n_by_score
from data.freelancer_filtering.gpt_short_response_filter import generate_gpt_responses, calculate_response_lengths
from data.swesmith_filtering.load_data import load_swesmith_data, filter_and_upload_swesmith


def main() -> None:
    questions, rows = load_swesmith_data()

    responses = generate_gpt_responses(questions, model="gpt-5-nano-2025-08-07")
    response_lengths = calculate_response_lengths(responses)

    filtered_questions = select_top_n_by_score(questions, response_lengths, n=10_000, descending=False)
    filter_and_upload_swesmith(questions, rows, filtered_questions, "DCAgent/exp-gfi-swesmith-short-response-filtered-10K")


if __name__ == "__main__":
    main()
