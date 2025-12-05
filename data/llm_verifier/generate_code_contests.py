#!/usr/bin/env python3
"""
Generate LLM Verifier dataset - tasks that use OpenAI as a judge to verify completion
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Import from parent package
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.commons import (
    upload_tasks_to_hf,
    subsample_tasks_directory,
    generate_tasks_from_questions,
    upload_traces_to_hf
)
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.gcs_cache import gcs_cache
from data.code_contests.generate import load_code_contests_questions
from data.llm_verifier.utils import add_llm_verifier_tests_to_questions

def main() -> None:
    """Main function - coordinates the pipeline"""
    # Load questions from freelancer using existing function
    code_contests_questions = load_code_contests_questions()
    # Add LLM verifier test files to questions
    questions_with_tests = add_llm_verifier_tests_to_questions(code_contests_questions)

    # Generate tasks with LLM verifier using commons function
    final_dataset_dir = generate_tasks_from_questions(questions_with_tests, "llm_verifier")

    # Subsample to manageable size
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)

    print(subsampled_dataset_dir)
    # # Upload to HuggingFace
    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/llm-verifier-code-contests")

    # # Run dataset to generate traces
    # hf_dataset = run_dataset_to_traces(
    #     subsampled_dataset_dir,
    #     model_name="gpt-5-nano-2025-08-07",
    #     agent_name="terminus-2",
    #     n_concurrent=256,
    #     agent_kwargs={"max_episodes": 8}
    # )

    # # Upload traces
    # upload_traces_to_hf(
    #     hf_dataset,
    #     "mlfoundations-dev/llm-verifier-freelancer-traces-terminus-2",
    #     "SFT"
    # )


if __name__ == "__main__":
    main()
