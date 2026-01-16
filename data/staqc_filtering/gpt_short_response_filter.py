#!/usr/bin/env python3
"""
GPT Short Response Filtering strategy for Stack Overflow dataset
"""

from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, select_top_n_by_score
from data.freelancer_filtering.gpt_short_response_filter import generate_gpt_responses, calculate_response_lengths
from data.staqc_filtering.load_data import load_staqc_questions


def main() -> None:
    questions = load_staqc_questions()

    responses = generate_gpt_responses(questions, model="gpt-5-nano-2025-08-07")
    response_lengths = calculate_response_lengths(responses)

    filtered_questions = select_top_n_by_score(questions, response_lengths, n=10_000, descending=False)
    filtered_questions = list(dict.fromkeys(filtered_questions))[:10_000]

    final_dataset_dir = generate_tasks_from_questions(filtered_questions, "staqc_short_response_filtered")
    upload_tasks_to_hf(final_dataset_dir, "DCAgent/exp-gfi-staqc-short-response-filtered-10K")


if __name__ == "__main__":
    main()
