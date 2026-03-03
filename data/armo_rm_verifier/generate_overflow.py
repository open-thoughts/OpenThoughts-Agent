#!/usr/bin/env python3
"""
Generate Stack Overflow dataset with ArmoRM Reward verifier.
This version extracts from an existing HF tasks dataset instead of raw XML.

Sample usage:
    # Use standard multi-turn trajectory verifier
    python3 data/armo_rm_verifier/generate_overflow.py --verifier_type standard

    # Use response.txt based verifier
    python3 data/armo_rm_verifier/generate_overflow.py --verifier_type response

    # Task generation only (No Traces, no Upload)
    python3 data/armo_rm_verifier/generate_overflow.py --skip_upload
"""

import os
import tempfile
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from parent package
from data.commons import (
    upload_tasks_to_hf, 
    download_hf_dataset
)
from scripts.harbor import tasks_parquet_converter as tpc

# Import both ArmoRM verifiers
from data.armo_rm_verifier.armorm_verifier import inject_armorm_verifier
from data.armo_rm_verifier.armorm_response_verifier import inject_armorm_response_verifier

def main() -> None:
    """Main function - processes StackOverflow tasks with chosen ArmoRM Reward verifier"""
    parser = argparse.ArgumentParser(description="Generate Stack Overflow dataset with ArmoRM Reward")
    parser.add_argument(
        "--verifier_type", 
        choices=["standard", "response"], 
        default="standard",
        help="Which ArmoRM verifier to use: 'standard' (trajectory) or 'response' (response.txt)"
    )
    parser.add_argument("--skip_upload", action="store_true", help="Skip upload to Hugging Face")
    args = parser.parse_args()
    
    source_repo = "mlfoundations-dev/stackexchange-overflow-sandboxes"
    
    print(f"Step 1: Downloading source tasks from {source_repo}...")
    snapshot_dir = Path(download_hf_dataset(source_repo))
    
    # 2. Extract tasks from parquet
    print("Step 2: Extracting tasks from parquet files...")
    parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {snapshot_dir}")
    
    output_dir = Path(tempfile.mkdtemp(prefix="overflow_armorm_"))
    print(f"Extracting to: {output_dir}")
    
    # Use the converter directly to extract
    tpc.from_parquet(
        parquet_path=str(parquet_files[0]),
        base=str(output_dir),
        on_exist="overwrite"
    )

    # 3. Verifier Injection
    if args.verifier_type == "response":
        print("Step 3: Injecting local ArmoRM Response verifier into extracted tasks...")
        inject_armorm_response_verifier(str(output_dir))
        suffix = "-armo-rm-response"
    else:
        print("Step 3: Injecting local ArmoRM Standard verifier into extracted tasks...")
        inject_armorm_verifier(str(output_dir))
        suffix = "-armo-rm"

    target_repo = f"DCAgent/stackexchange-overflow-sandboxes{suffix}"

    # 4. Upload Tasks
    if not args.skip_upload:
        print(f"Step 4: Uploading ArmoRM-verified tasks to {target_repo}...")
        upload_tasks_to_hf(str(output_dir), target_repo)
        print(f"Success! Tasks uploaded to: https://huggingface.co/datasets/{target_repo}")
    else:
        print(f"Upload skipped. Local tasks are available in: {output_dir}")
    
    print("Generation Complete!")

if __name__ == "__main__":
    main()
