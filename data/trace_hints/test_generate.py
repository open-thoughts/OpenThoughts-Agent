#!/usr/bin/env python3
"""
Test script for trace hints dataset generation (small subset)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.trace_hints.generate import generate_hints_dataset
from data.commons import generate_tasks_from_questions


def main():
    print("Testing trace hints dataset generation with small subset...")

    augmented_instructions = generate_hints_dataset(
        dataset_name="DCAgent/taskmaster2-gpt5mini",
        model="gpt-4o-mini",
        batch_size=10,
        max_requests_per_minute=100,
        max_trials=5
    )

    print(f"\nGenerated {len(augmented_instructions)} augmented instructions")

    dataset_dir = generate_tasks_from_questions(
        augmented_instructions,
        dataset_prefix="taskmaster2-hints-test"
    )
    print(f"Test dataset generated in: {dataset_dir}")

    tasks = list(Path(dataset_dir).iterdir())
    print(f"Generated {len(tasks)} task directories")

    if tasks:
        first_task = tasks[0]
        instruction_file = first_task / "instruction.md"
        if instruction_file.exists():
            print(f"\nContent of {first_task.name}/instruction.md:")
            content = instruction_file.read_text()
            print(content[:1000] + ("..." if len(content) > 1000 else ""))


if __name__ == "__main__":
    main()
