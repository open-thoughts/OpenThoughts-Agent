#!/usr/bin/env python3
"""
Regenerate benchmark tasks with well-specified instructions.

This script:
1. Reads existing benchmark tasks
2. Extracts function signatures from test files
3. Generates improved, well-specified instructions
4. Writes them back to instruction.md files
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.commons import (
    extract_function_signatures_from_test,
    generate_well_specified_instruction,
)

BENCHMARK_DIR = Path("/scratch/10000/eguha3/dc-agent/data/benchmark_tasks_by_dataset")


def get_task_title_from_dirname(dirname: str) -> str:
    """Extract a human-readable title from directory name."""
    # e.g., "ghactions_0000_parse_workflow" -> "Parse Workflow"
    parts = dirname.split('_')
    if len(parts) >= 3:
        # Skip prefix and number
        title_parts = parts[2:]
        return ' '.join(word.capitalize() for word in title_parts)
    return dirname.replace('_', ' ').title()


def regenerate_task_instruction(task_dir: Path) -> dict:
    """
    Regenerate instruction for a single task.

    Returns dict with status info.
    """
    result = {
        "task": task_dir.name,
        "status": "unchanged",
        "old_preview": "",
        "new_preview": "",
        "functions_found": [],
        "classes_found": [],
    }

    instruction_path = task_dir / "instruction.md"
    test_path = task_dir / "tests" / "test_solution.py"

    if not instruction_path.exists():
        result["status"] = "error"
        result["error"] = "No instruction.md found"
        return result

    if not test_path.exists():
        result["status"] = "error"
        result["error"] = "No test_solution.py found"
        return result

    # Read files
    old_instruction = instruction_path.read_text(encoding="utf-8")
    test_code = test_path.read_text(encoding="utf-8")

    result["old_preview"] = old_instruction[:150].replace('\n', ' ')

    # Check if already well-specified (has backtick function name)
    import re
    if re.search(r'`\w+\([^)]*\)`', old_instruction):
        result["status"] = "already_specified"
        return result

    # Extract function signatures
    extracted = extract_function_signatures_from_test(test_code)
    result["functions_found"] = [f["name"] for f in extracted["functions"]]
    result["classes_found"] = [c["name"] for c in extracted["classes"]]

    if not extracted["functions"] and not extracted["classes"]:
        result["status"] = "no_signatures_found"
        return result

    # Generate title from directory name
    title = get_task_title_from_dirname(task_dir.name)

    # Generate new instruction
    new_instruction = generate_well_specified_instruction(
        test_code=test_code,
        task_title=title,
    )

    result["new_preview"] = new_instruction[:200].replace('\n', ' ')

    # Write new instruction
    instruction_path.write_text(new_instruction, encoding="utf-8")
    result["status"] = "improved"

    return result


def regenerate_dataset(dataset_name: str) -> dict:
    """Regenerate instructions for all tasks in a dataset."""
    dataset_dir = BENCHMARK_DIR / dataset_name

    if not dataset_dir.exists():
        return {"error": f"Dataset {dataset_name} not found"}

    results = {
        "dataset": dataset_name,
        "total": 0,
        "improved": 0,
        "already_specified": 0,
        "no_signatures": 0,
        "errors": 0,
        "tasks": []
    }

    for task_dir in sorted(dataset_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        results["total"] += 1
        task_result = regenerate_task_instruction(task_dir)
        results["tasks"].append(task_result)

        if task_result["status"] == "improved":
            results["improved"] += 1
        elif task_result["status"] == "already_specified":
            results["already_specified"] += 1
        elif task_result["status"] == "no_signatures_found":
            results["no_signatures"] += 1
        else:
            results["errors"] += 1

    return results


def main():
    """Regenerate all benchmark datasets."""
    # All 21 datasets
    datasets = [
        # Low pass rate - need most improvement
        "ghactions", "bugsinpy_mf", "manybugs", "bugswarm",
        "travistorrent", "e2egit", "defects4j", "softwareheritage",
        "codereval", "crosscodeeval",
        # Medium pass rate
        "codeelo", "bigcodebench", "exercism", "bugsinpy",
        "quixbugs", "swebench",
        # High pass rate - might already be good
        "pymethods2test", "methods2test", "unitsyn", "taco", "codenet",
    ]

    all_results = []
    total_improved = 0
    total_tasks = 0

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print('='*60)

        results = regenerate_dataset(dataset)

        if "error" in results:
            print(f"  ERROR: {results['error']}")
            continue

        all_results.append(results)
        total_improved += results["improved"]
        total_tasks += results["total"]

        print(f"  Total tasks: {results['total']}")
        print(f"  Improved: {results['improved']}")
        print(f"  Already specified: {results['already_specified']}")
        print(f"  No signatures found: {results['no_signatures']}")
        print(f"  Errors: {results['errors']}")

        # Show sample improvements
        improved_samples = [t for t in results["tasks"] if t["status"] == "improved"][:2]
        for task in improved_samples:
            print(f"\n  Sample: {task['task']}")
            print(f"    Functions found: {task['functions_found']}")
            print(f"    Old: {task['old_preview'][:60]}...")
            print(f"    New: {task['new_preview'][:80]}...")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tasks improved: {total_improved}/{total_tasks}")

    # Per-dataset summary
    print("\nPer-dataset breakdown:")
    for r in all_results:
        print(f"  {r['dataset']}: {r['improved']}/{r['total']} improved")

    return all_results


if __name__ == "__main__":
    main()
