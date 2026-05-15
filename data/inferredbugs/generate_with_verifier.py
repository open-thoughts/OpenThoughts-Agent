#!/usr/bin/env python3
"""
Generate verified InferredBugs dataset by adding LLM-authored verifiers to an existing tasks dataset.
"""

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
from data.inferredbugs.structural_verifier import inject_inferredbugs_verifier

def update_instructions_with_requirement(dataset_dir: str):
    """
    Appends the Deliverable Requirement to each instruction.md file.
    This ensures the agent knows where to save the fix for the structural verifier.
    """
    tasks_root = Path(dataset_dir)
    print(f"Updating instructions in: {tasks_root}")
    
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
            
        instr_path = task_dir / "instruction.md"
        if instr_path.exists():
            content = instr_path.read_text()
            
            # Simple heuristic to find the target file from the instruction text
            import re
            file_match = re.search(r"\*\*File:\*\* `(.*?)`", content) or re.search(r"###\s*File:\s*\n(.*)", content)
            
            if file_match:
                target_file = file_match.group(1)
                requirement = f"""

## Deliverable Requirement
Write the complete, corrected version of the file to: `/app/{target_file}`.
The directory does not exist yet. You must create it and write the file:
```bash
mkdir -p /app/{target_file.rsplit('/', 1)[0]}
cat > /app/{target_file} << 'ENDOFFILE'
<full corrected file content here>
ENDOFFILE
```
"""
                if "Deliverable Requirement" not in content:
                    instr_path.write_text(content + requirement)

def main() -> None:
    """Main function - processes InferredBugs tasks with LLM-authored verifiers"""
    parser = argparse.ArgumentParser(description="Generate Verified InferredBugs dataset")
    parser.add_argument("--skip_upload", action="store_true", help="Skip upload to Hugging Face")
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="LLM to use for authoring verifiers")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of tasks to process")
    args = parser.parse_args()
    
    source_repo = "mlfoundations-dev/inferredbugs-sandboxes"
    target_repo = "DCAgent/inferredbugs-sandboxes-verifier"
    
    # 1. Download
    print(f"Step 1: Downloading source tasks from {source_repo}...")
    snapshot_dir = Path(download_hf_dataset(source_repo))
    
    # 2. Extract tasks
    print("Step 2: Extracting tasks from parquet files...")
    parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {snapshot_dir}")
    
    # Use a FIXED directory to allow for resumption and skip logic
    output_dir = PROJECT_ROOT / "data" / "inferredbugs" / "workdir"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {output_dir}")
    
    # Extract only if the directory is empty, to support resumption
    if not any(output_dir.iterdir()):
        print(f"Extracting tasks to: {output_dir}")
        tpc.from_parquet(
            parquet_path=str(parquet_files[0]),
            base=str(output_dir),
            on_exist="overwrite"
        )
    else:
        print(f"Directory {output_dir} not empty. Skipping extraction to support resumption.")

    # 3. Update Instructions
    print("Step 3: Appending Deliverable Requirements to instructions...")
    update_instructions_with_requirement(str(output_dir))

    # 4. Inject Authored Verifiers
    print(f"Step 4: Authoring and injecting structural verifiers using {args.model}...")
    # We need to collect the instruction texts again to pass to the authoring engine
    task_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    
    # Apply limit if specified
    if args.limit:
        print(f"Limiting processing to first {args.limit} tasks.")
        task_dirs = task_dirs[:args.limit]

    instructions = []
    for d in task_dirs:
        instr_file = d / "instruction.md"
        instructions.append(instr_file.read_text() if instr_file.exists() else "")

    inject_inferredbugs_verifier(str(output_dir), instructions, model_name=args.model)

    # 5. Upload
    if not args.skip_upload:
        print(f"Step 5: Uploading verified tasks to {target_repo}...")
        upload_tasks_to_hf(str(output_dir), target_repo)
        print(f"Success! Tasks uploaded to: https://huggingface.co/datasets/{target_repo}")
    else:
        print(f"Upload skipped. Local tasks available in: {output_dir}")
    
    print("Verified InferredBugs Generation Complete!")

if __name__ == "__main__":
    main()
