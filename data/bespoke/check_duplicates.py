#!/usr/bin/env python3

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import difflib

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

def compute_similarity(text1: str, text2: str) -> float:
    """Compute similarity ratio between two texts."""
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def check_duplicates_within_folder(directory: Path, dir_name: str):
    """Check for duplicates within a single folder."""
    print(f"\n{'='*80}")
    print(f"CHECKING DUPLICATES WITHIN: {dir_name}")
    print('='*80)
    
    tasks_info = {}
    hash_to_tasks = defaultdict(list)
    
    # Scan directory
    for task_dir in directory.iterdir():
        if task_dir.is_dir() and not task_dir.name.startswith('.'):
            content = get_instruction_content(task_dir)
            if content:
                task_hash = compute_hash(content)
                tasks_info[task_dir.name] = (content, task_hash)
                hash_to_tasks[task_hash].append(task_dir.name)
    
    print(f"Found {len(tasks_info)} tasks with instructions")
    
    # Find duplicates by content
    duplicate_count = 0
    for task_hash, task_ids in hash_to_tasks.items():
        if len(task_ids) > 1:
            duplicate_count += 1
            print(f"\nDuplicate Group {duplicate_count} (identical content):")
            for task_id in task_ids:
                print(f"  - {task_id}")
    
    if duplicate_count == 0:
        print("No exact duplicates found within this folder")
    
    # Check for high similarity
    print(f"\nHigh similarity pairs (>90%) in {dir_name}:")
    task_list = list(tasks_info.items())
    similar_pairs = []
    
    for i in range(len(task_list)):
        for j in range(i + 1, len(task_list)):
            task1_id, (content1, hash1) = task_list[i]
            task2_id, (content2, hash2) = task_list[j]
            
            if hash1 != hash2:  # Skip exact duplicates
                similarity = compute_similarity(content1, content2)
                if similarity > 0.90:
                    similar_pairs.append((task1_id, task2_id, similarity))
    
    if similar_pairs:
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        for task1, task2, sim in similar_pairs[:10]:
            print(f"  - {task1[:20]}... <-> {task2[:20]}... : {sim:.2%}")
    else:
        print("  No highly similar pairs found (>90%)")
    
    return len(tasks_info), len(hash_to_tasks), duplicate_count

def find_duplicates():
    """Find duplicate tasks between and within successful_runs and successful_runs_again."""
    
    dir1 = Path("/Users/etash/Documents/ot-agent/successful_runs")
    dir2 = Path("/Users/etash/Documents/ot-agent/successful_runs_again")
    
    # Check duplicates within each folder first
    stats1 = check_duplicates_within_folder(dir1, "successful_runs")
    stats2 = check_duplicates_within_folder(dir2, "successful_runs_again")
    
    print(f"\n{'='*80}")
    print("CHECKING DUPLICATES BETWEEN FOLDERS")
    print('='*80)
    
    # Store task info: {task_id: (directory, content, hash)}
    tasks_info = {}
    
    # Scan first directory
    for task_dir in dir1.iterdir():
        if task_dir.is_dir() and not task_dir.name.startswith('.'):
            content = get_instruction_content(task_dir)
            if content:
                task_hash = compute_hash(content)
                tasks_info[task_dir.name] = ("successful_runs", content, task_hash)
    
    print(f"Found {len(tasks_info)} tasks with instructions in successful_runs")
    
    # Scan second directory
    dir2_count = 0
    for task_dir in dir2.iterdir():
        if task_dir.is_dir() and not task_dir.name.startswith('.'):
            content = get_instruction_content(task_dir)
            if content:
                dir2_count += 1
                task_hash = compute_hash(content)
                
                # Check if this task ID exists in first directory
                if task_dir.name in tasks_info:
                    prev_dir, prev_content, prev_hash = tasks_info[task_dir.name]
                    
                    # Check for exact duplicate
                    if prev_hash == task_hash:
                        print(f"\n✓ EXACT DUPLICATE: {task_dir.name}")
                        print(f"  - Found in both directories with identical content")
                    else:
                        # Check similarity
                        similarity = compute_similarity(prev_content, content)
                        if similarity > 0.9:  # 90% similar
                            print(f"\n⚠ SIMILAR TASK: {task_dir.name}")
                            print(f"  - Similarity: {similarity:.2%}")
                            print(f"  - Different content but highly similar")
                else:
                    # New task not in first directory
                    tasks_info[task_dir.name] = ("successful_runs_again", content, task_hash)
    
    print(f"\nFound {dir2_count} tasks with instructions in successful_runs_again")
    
    # Find tasks by content hash (different IDs but same content)
    hash_to_tasks = defaultdict(list)
    for task_id, (directory, content, task_hash) in tasks_info.items():
        hash_to_tasks[task_hash].append((task_id, directory))
    
    print("\n" + "=" * 80)
    print("DUPLICATE CONTENT WITH DIFFERENT TASK IDS:")
    print("=" * 80)
    
    duplicate_groups = 0
    for task_hash, tasks in hash_to_tasks.items():
        if len(tasks) > 1:
            duplicate_groups += 1
            print(f"\nDuplicate Group {duplicate_groups}:")
            for task_id, directory in tasks:
                print(f"  - {task_id} (from {directory})")
    
    if duplicate_groups == 0:
        print("No duplicate content found with different task IDs")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    
    tasks_only_in_1 = sum(1 for tid, (d, _, _) in tasks_info.items() 
                          if d == "successful_runs" and tid not in 
                          [t for t in dir2.iterdir() if t.is_dir()])
    tasks_only_in_2 = sum(1 for tid, (d, _, _) in tasks_info.items() 
                          if d == "successful_runs_again" and tid not in 
                          [t for t in dir1.iterdir() if t.is_dir()])
    
    print(f"Total unique task IDs: {len(tasks_info)}")
    print(f"Total unique instruction contents: {len(hash_to_tasks)}")
    print(f"Duplicate content groups: {duplicate_groups}")
    
    # Check for high similarity between different tasks
    print("\n" + "=" * 80)
    print("CHECKING FOR HIGH SIMILARITY (>85%) BETWEEN DIFFERENT TASKS:")
    print("=" * 80)
    
    task_list = list(tasks_info.items())
    similar_pairs = []
    
    for i in range(len(task_list)):
        for j in range(i + 1, len(task_list)):
            task1_id, (dir1, content1, hash1) = task_list[i]
            task2_id, (dir2, content2, hash2) = task_list[j]
            
            if hash1 != hash2:  # Skip exact duplicates already found
                similarity = compute_similarity(content1, content2)
                if similarity > 0.85:
                    similar_pairs.append((task1_id, task2_id, similarity, dir1, dir2))
    
    if similar_pairs:
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        for task1, task2, sim, d1, d2 in similar_pairs[:10]:  # Show top 10
            print(f"\n  {task1} ({d1}) <-> {task2} ({d2})")
            print(f"  Similarity: {sim:.2%}")
    else:
        print("No highly similar tasks found (threshold: 85%)")

if __name__ == "__main__":
    find_duplicates()