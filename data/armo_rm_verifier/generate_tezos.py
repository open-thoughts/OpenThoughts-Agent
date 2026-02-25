#!/usr/bin/env python3
"""
Generate StackExchange Tezos dataset with ArmoRM verifier enabled by default.

Sample usage:
    # Full run (Traces + Upload)
    python3 data/armo_rm_verifier/generate_tezos.py

    # Local test (No Traces, No Upload)
    python3 data/armo_rm_verifier/generate_tezos.py --skip_traces --skip_upload

    # Task generation only (No Traces, with Upload)
    python3 data/armo_rm_verifier/generate_tezos.py --skip_traces
"""

import os
import tempfile
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from parent package
from data.commons import (
    upload_tasks_to_hf, 
    generate_tasks_from_questions, 
    subsample_tasks_directory, 
    upsample_tasks_directory, 
    upload_traces_to_hf
)
from data.stackexchange.generate_codereview import (
    download_and_extract_dataset, 
    parse_posts_xml, 
    extract_questions_from_data
)
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.armo_rm_verifier.armorm_verifier import inject_armorm_verifier

def main() -> None:
    """Main function - generates StackExchange Tezos tasks with ArmoRM enabled by default"""
    parser = argparse.ArgumentParser(description="Generate StackExchange Tezos dataset with ArmoRM")
    parser.add_argument("--skip_traces", action="store_true", help="Skip trace generation")
    parser.add_argument("--skip_upload", action="store_true", help="Skip upload to Hugging Face")
    args = parser.parse_args()
    
    # 1. Load data
    print("Downloading and parsing data...")
    posts_xml_path = download_and_extract_dataset("https://archive.org/download/stackexchange/tezos.stackexchange.com.7z")
    questions_data = parse_posts_xml(posts_xml_path)
    questions = extract_questions_from_data(questions_data)
    
    # 2. Task Generation
    print("Generating base tasks...")
    final_dataset_dir = generate_tasks_from_questions(questions, "tezos")

    # 3. ArmoRM Verifier Injection (Enabled by default)
    print("Injecting local ArmoRM verifier...")
    inject_armorm_verifier(final_dataset_dir)
    suffix = "-armo-rm"

    # 4. Standard Post-Processing
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    upsampled_dataset_dir = upsample_tasks_directory(subsampled_dataset_dir, 10_000)
    
    final_tasks_dir = upsampled_dataset_dir

    # 5. Trace Generation (Optional)
    if not args.skip_traces:
        print("Generating traces using teacher model...")
        hf_dataset = run_dataset_to_traces(
            final_tasks_dir, 
            model_name="gpt-5-nano-2025-08-07", 
            agent_name="terminus-2", 
            n_concurrent=256, 
            agent_kwargs={"max_episodes": 8}
        )
        
        if not args.skip_upload:
            print("Uploading traces to Hugging Face...")
            upload_traces_to_hf(hf_dataset, f"DCAgent/stackexchange-tezos-sandboxes-traces-terminus-2{suffix}", "SFT")
    else:
        print("Skipping trace generation.")
    
    # 6. Upload Tasks
    if not args.skip_upload:
        print("Uploading tasks to Hugging Face...")
        upload_tasks_to_hf(final_tasks_dir, f"DCAgent/stackexchange-tezos-sandboxes{suffix}")
    else:
        print(f"Upload skipped. Final tasks are available in: {final_tasks_dir}")
    
    print("Generation Complete!")

if __name__ == "__main__":
    main()
