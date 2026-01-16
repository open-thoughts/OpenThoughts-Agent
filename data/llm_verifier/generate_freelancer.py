#!/usr/bin/env python3
"""
Generate Freelancer projects dataset with LLM verifier tests.
"""

from data.freelancer.generate import scrape_freelancer_projects
from data.commons import generate_tasks_from_questions, upsample_tasks_directory, upload_tasks_to_hf
from data.llm_verifier.utils import add_llm_verifier_tests_to_questions


def main() -> None:
    questions = scrape_freelancer_projects()
    questions_with_tests = add_llm_verifier_tests_to_questions(questions)
    dataset_dir = generate_tasks_from_questions(questions_with_tests, "llm_verifier_freelancer")
    upload_tasks_to_hf(dataset_dir, "DCAgent/exp_llmve_llm-verifier-freelancer-sandboxes")


if __name__ == "__main__":
    main()
