#!/usr/bin/env python3
"""
Generate verified Self-Instruct dataset by adding LLM-authored functional verifiers.
"""

import tempfile
import sys
import argparse
from pathlib import Path

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
from data.self_instruct.self_instruct_verifier import inject_self_instruct_verifier

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Verified Self-Instruct dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--skip_upload", action="store_true", help="Skip HF upload")
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="Authoring model")
    args = parser.parse_args()
    
    source_repo = "DCAgent/selfinstruct-naive-sandboxes-2"
    target_repo = "DCAgent/selfinstruct-naive-sandboxes-2-verified"
    
    # 1. Download
    print(f"Step 1: Downloading source tasks from {source_repo}...")
    snapshot_dir = Path(download_hf_dataset(source_repo))
    
    # 2. Extract tasks
    output_dir = PROJECT_ROOT / "data" / "self_instruct" / "workdir"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not any(output_dir.iterdir()):
        print(f"Step 2: Extracting tasks to: {output_dir}")
        parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
        tpc.from_parquet(parquet_path=str(parquet_files[0]), base=str(output_dir), on_exist="overwrite")
    else:
        print(f"Step 2: Reusing tasks in {output_dir}")

    # 3. Inject Verifiers
    print(f"Step 3: Authoring functional verifiers using {args.model}...")
    task_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    if args.limit:
        task_dirs = task_dirs[:args.limit]
        
    instructions = [ (d / "instruction.md").read_text() for d in task_dirs ]
    
    inject_self_instruct_verifier(str(output_dir), instructions, model_name=args.model)

    # 4. Upload
    if not args.skip_upload:
        print(f"Step 4: Uploading to {target_repo}...")
        upload_tasks_to_hf(str(output_dir), target_repo)
    else:
        print(f"Upload skipped. Tasks available in: {output_dir}")

if __name__ == "__main__":
    main()
