#!/usr/bin/env python3
"""
Generate LLM Verifier dataset for DCAgent dev set - tasks that use OpenAI as a judge to verify completion
"""

import sys
from pathlib import Path

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
from data.llm_verifier.utils import (
    copy_all_files_from_dataset,
    write_test_files_to_tasks,
    download_dataset_instructions
)
from data.llm_verifier.utils import add_llm_verifier_tests_to_questions


def main() -> None:
    """Main function - coordinates the pipeline"""
    dataset_name = "DCAgent/dev_set_71_tasks"
    dataset_prefix = "llm_verifier_dcagent"
    
    # Copy all files from source dataset
    source_dir = copy_all_files_from_dataset(dataset_name, dataset_prefix)
    
    # Download instructions to embed in test files
    questions = download_dataset_instructions(dataset_name)
    
    # Copy to new directory and write test files
    output_dir = write_test_files_to_tasks(source_dir, questions)
    
    print(f"Generated tasks in: {output_dir}")

    # Upload to HuggingFace
    upload_tasks_to_hf(output_dir, "DCAgent/llm-verifier-dcagent-dev-set")

    # # Run dataset to generate traces
    # hf_dataset = run_dataset_to_traces(
    #     final_dataset_dir,
    #     model_name="gpt-5-nano-2025-08-07",
    #     agent_name="terminus-2",
    #     n_concurrent=256,
    #     agent_kwargs={"max_episodes": 8}
    # )

    # # Upload traces
    # upload_traces_to_hf(
    #     hf_dataset,
    #     "mlfoundations-dev/llm-verifier-dcagent-dev-set-traces-terminus-2",
    #     "SFT"
    # )


if __name__ == "__main__":
    main()
