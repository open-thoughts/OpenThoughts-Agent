#!/usr/bin/env python3
"""
AskLLM filtering strategy for Stack Overflow dataset
"""

from data.commons import generate_tasks_from_questions, upload_tasks_to_hf, select_top_n_by_score
from data.freelancer_filtering.ask_llm_filter import rate_questions_with_llm
from data.staqc_filtering.load_data import load_staqc_questions


def main() -> None:
    questions = load_staqc_questions()
    ratings = rate_questions_with_llm(questions, model="gpt-4o-mini")

    filtered_questions = select_top_n_by_score(questions, ratings, n=10_000, descending=True)
    filtered_questions = list(dict.fromkeys(filtered_questions))[:10_000]

    final_dataset_dir = generate_tasks_from_questions(filtered_questions, "staqc_askllm_filtered")
    upload_tasks_to_hf(final_dataset_dir, "DCAgent/exp-gfi-staqc-askllm-filtered-10K")


if __name__ == "__main__":
    main()
