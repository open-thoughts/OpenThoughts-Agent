#!/usr/bin/env python3

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Set

def get_instruction_content(task_path: Path) -> str:
    """Read the instruction.md file from a task directory."""
    # Try different possible paths for instruction.md
    possible_paths = [
        task_path / "task_description" / "instruction.md",
        task_path / "instructions.md",
        task_path / "instruction.md"
    ]
    
    for instruction_path in possible_paths:
        if instruction_path.exists():
            try:
                with open(instruction_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Error reading {instruction_path}: {e}")
    return ""

def compute_hash(content: str) -> str:
    """Compute hash of instruction content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def remove_duplicates(dry_run=True):
    """Remove duplicate tasks from successful_runs_again that exist in successful_runs."""
    
    dir1 = Path("/Users/etash/Documents/ot-agent/successful_runs")
    dir2 = Path("/Users/etash/Documents/ot-agent/successful_runs_again")
    
    print(f"{'='*80}")
    print(f"{'DRY RUN MODE' if dry_run else 'ACTUAL REMOVAL MODE'}")
    print(f"{'='*80}")
    print(f"Source directory (keep): {dir1}")
    print(f"Target directory (remove duplicates from): {dir2}")
    print("-" * 80)
    
    # Build index of all tasks in successful_runs
    source_tasks = {}  # {task_id: content_hash}
    source_hashes = set()  # All content hashes from source
    
    for task_dir in dir1.iterdir():
        if task_dir.is_dir() and not task_dir.name.startswith('.'):
            content = get_instruction_content(task_dir)
            if content:
                content_hash = compute_hash(content)
                source_tasks[task_dir.name] = content_hash
                source_hashes.add(content_hash)
    
    print(f"Found {len(source_tasks)} tasks in successful_runs")
    
    # Check tasks in successful_runs_again
    to_remove_by_id = []  # Same task ID with same content
    to_remove_by_content = []  # Different task ID but same content
    unique_tasks = []
    
    for task_dir in dir2.iterdir():
        if task_dir.is_dir() and not task_dir.name.startswith('.'):
            content = get_instruction_content(task_dir)
            if content:
                content_hash = compute_hash(content)
                
                # Check if same task ID exists with same content
                if task_dir.name in source_tasks:
                    if source_tasks[task_dir.name] == content_hash:
                        to_remove_by_id.append(task_dir.name)
                    else:
                        # Same ID but different content - keep it
                        unique_tasks.append(task_dir.name)
                # Check if same content exists (but different ID)
                elif content_hash in source_hashes:
                    to_remove_by_content.append(task_dir.name)
                else:
                    unique_tasks.append(task_dir.name)
    
    print(f"\nAnalysis Results:")
    print(f"  - Tasks to remove (same ID, same content): {len(to_remove_by_id)}")
    print(f"  - Tasks to remove (different ID, same content): {len(to_remove_by_content)}")
    print(f"  - Unique tasks to keep: {len(unique_tasks)}")
    
    all_to_remove = to_remove_by_id + to_remove_by_content
    
    if all_to_remove:
        print(f"\n{'='*80}")
        print(f"TASKS TO BE REMOVED ({len(all_to_remove)} total):")
        print(f"{'='*80}")
        
        if to_remove_by_id:
            print("\nDuplicates by ID (exact same task ID and content):")
            for task_id in sorted(to_remove_by_id)[:10]:  # Show first 10
                print(f"  - {task_id}")
            if len(to_remove_by_id) > 10:
                print(f"  ... and {len(to_remove_by_id) - 10} more")
        
        if to_remove_by_content:
            print("\nDuplicates by content (different ID but same content):")
            for task_id in sorted(to_remove_by_content):
                print(f"  - {task_id}")
        
        if not dry_run:
            print(f"\n{'='*80}")
            print("REMOVING DUPLICATES...")
            print(f"{'='*80}")
            
            removed_count = 0
            failed_count = 0
            
            for task_id in all_to_remove:
                task_path = dir2 / task_id
                try:
                    shutil.rmtree(task_path)
                    removed_count += 1
                    print(f"✓ Removed: {task_id}")
                except Exception as e:
                    failed_count += 1
                    print(f"✗ Failed to remove {task_id}: {e}")
            
            print(f"\nRemoval complete:")
            print(f"  - Successfully removed: {removed_count}")
            print(f"  - Failed: {failed_count}")
        else:
            print(f"\n{'='*80}")
            print("DRY RUN COMPLETE - No files were actually removed")
            print("To perform actual removal, run with dry_run=False")
            print(f"{'='*80}")
    else:
        print("\nNo duplicates found to remove!")
    
    return all_to_remove

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--remove":
        print("Running in ACTUAL REMOVAL mode...")
        confirm = input("Are you sure you want to remove duplicates? Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            remove_duplicates(dry_run=False)
        else:
            print("Cancelled.")
    else:
        print("Running in DRY RUN mode (no files will be removed)...")
        print("To actually remove files, run: python remove_duplicates.py --remove\n")
        remove_duplicates(dry_run=True)