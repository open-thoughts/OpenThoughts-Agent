#!/usr/bin/env python3
"""
Generate trace hints dataset in sandboxes style
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from bespokelabs import curator
    CURATOR_AVAILABLE = True
except ImportError:
    CURATOR_AVAILABLE = False

from datasets import load_dataset
from data.commons import upload_tasks_to_hf, generate_tasks_from_questions
from scripts.sandboxes.run_and_export_traces import run_dataset_to_traces
from data.commons import upload_traces_to_hf
from data.trace_hints.augment_dataset import (
    group_episodes_by_trial,
    extract_hints_from_trials,
    create_augmented_instructions
)


def load_traces_dataset(dataset_name: str = "DCAgent/taskmaster2-gpt5mini"):
    print(f"Loading traces dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print(f"Loaded {len(dataset['train'])} trace episodes")
    return dataset


def generate_hints_dataset(
    dataset_name: str = "DCAgent/taskmaster2-gpt5mini",
    model: str = "gpt-4o-mini",
    batch_size: int = 100,
    max_requests_per_minute: int = 500,
    task_markers: List[str] = None,
    max_trials: Optional[int] = None
) -> List[tuple]:
    if not CURATOR_AVAILABLE:
        raise ImportError("curator required. Install: pip install bespokelabs-curator")

    dataset = load_traces_dataset(dataset_name)

    print("\n[1/3] Grouping episodes by trial...")
    trials = group_episodes_by_trial(dataset)

    if max_trials:
        trials = dict(list(trials.items())[:max_trials])
        print(f"Limited to {len(trials)} trials for testing")

    print("\n[2/3] Extracting hints from episodes...")
    trial_hints = extract_hints_from_trials(
        trials,
        model=model,
        batch_size=batch_size,
        max_requests_per_minute=max_requests_per_minute,
        task_markers=task_markers
    )

    print("\n[3/3] Creating augmented instructions...")
    augmented_instructions = create_augmented_instructions(
        trials,
        trial_hints,
        task_markers=task_markers
    )

    return augmented_instructions


def main() -> None:
    dataset_name = "DCAgent/taskmaster2-gpt5mini"
    repo_name = "mlfoundations-dev/taskmaster2-hints"

    augmented_instructions = generate_hints_dataset(
        dataset_name=dataset_name,
        model="gpt-4o-mini",
        batch_size=100,
        max_requests_per_minute=500
    )

    dataset_prefix = f"{dataset_name.split('/')[-1]}-hints"
    final_dataset_dir = generate_tasks_from_questions(
        augmented_instructions,
        dataset_prefix=dataset_prefix
    )

    upload_tasks_to_hf(final_dataset_dir, repo_name)

    hf_dataset = run_dataset_to_traces(
        final_dataset_dir,
        model_name="gpt-5-nano-2025-08-07",
        agent_name="terminus-2",
        n_concurrent=1024,
        agent_kwargs={"max_episodes": 8}
    )

    upload_traces_to_hf(
        hf_dataset,
        f"{repo_name}-traces-terminus-2",
        "SFT"
    )


if __name__ == "__main__":
    main()
