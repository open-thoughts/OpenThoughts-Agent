import os
import tempfile
import sys
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset

# Import from parent package
from data.commons import upload_tasks_to_hf, generate_tasks_from_questions, subsample_tasks_directory, upload_traces_to_hf
from data.stackexchange.generate_codereview import download_and_extract_dataset, parse_posts_xml, extract_questions_from_data
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.commons import upload_traces_to_hf

def main() -> None:
    """Main function - coordinates the pipeline with temp directories"""
    # Load data
    posts_xml_path = download_and_extract_dataset("https://archive.org/download/stackexchange/unix.stackexchange.com.7z")
    questions_data = parse_posts_xml(posts_xml_path)
    questions = extract_questions_from_data(questions_data)
    final_dataset_dir = generate_tasks_from_questions(questions, "unix")
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    upload_tasks_to_hf(subsampled_dataset_dir, "mlfoundations-dev/stackexchange-unix-sandboxes")
    hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=1024, agent_kwargs={"max_episodes": 8})
    upload_traces_to_hf(hf_dataset, "mlfoundations-dev/stackexchange-unix-sandboxes-traces-terminus-2", "SFT")

if __name__ == "__main__":
    main()