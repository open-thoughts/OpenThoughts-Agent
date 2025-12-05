#!/usr/bin/env python3
"""BaseDataGenerator integration for neulab/agent-data-collection subsets."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.commons import (
    BaseDataGenerator,
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    finalize_dataset_output,
    generate_tasks_from_questions,
    subsample_tasks_directory,
    upsample_tasks_directory,
    upload_tasks_to_hf,
)
from data.neulab.common import extract_questions_from_subset


NEULAB_TASK_CONFIGS: Dict[str, Dict[str, object]] = {
    "agenttuning-alfworld": {
        "subset": "agenttuning_alfworld",
        "dataset_prefix": "agenttuning-alfworld",
        "default_repo": "DCAgent/neulab-agenttuning-alfworld-sandboxes",
    },
    "agenttuning-db": {
        "subset": "agenttuning_db",
        "dataset_prefix": "agenttuning-db",
        "default_repo": "DCAgent/neulab-agenttuning-db-sandboxes",
    },
    "agenttuning-kg": {
        "subset": "agenttuning_kg",
        "dataset_prefix": "agenttuning-kg",
        "default_repo": "DCAgent/neulab-agenttuning-kg-sandboxes",
    },
    "agenttuning-mind2web": {
        "subset": "agenttuning_mind2web",
        "dataset_prefix": "agenttuning-mind2web",
        "default_repo": "DCAgent/neulab-agenttuning-mind2web-sandboxes",
    },
    "agenttuning-os": {
        "subset": "agenttuning_os",
        "dataset_prefix": "agenttuning-os",
        "default_repo": "DCAgent/neulab-agenttuning-os-sandboxes",
    },
    "agenttuning-webshop": {
        "subset": "agenttuning_webshop",
        "dataset_prefix": "agenttuning-webshop",
        "default_repo": "DCAgent/neulab-agenttuning-webshop-sandboxes",
    },
    "code-feedback": {
        "subset": "code_feedback",
        "dataset_prefix": "code-feedback",
        "default_repo": "DCAgent/neulab-code-feedback-sandboxes",
    },
    "codeactinstruct": {
        "subset": "codeactinstruct",
        "dataset_prefix": "codeactinstruct",
        "default_repo": "DCAgent/neulab-codeactinstruct-sandboxes",
    },
    "go-browse-wa": {
        "subset": "go-browse-wa",
        "dataset_prefix": "go-browse-wa",
        "default_repo": "DCAgent/neulab-go-browse-wa-sandboxes",
    },
    "mind2web": {
        "subset": "mind2web",
        "dataset_prefix": "mind2web",
        "default_repo": "DCAgent/neulab-mind2web-sandboxes",
    },
    "nebius-swe-agent-trajectories": {
        "subset": "nebius_SWE-agent-trajectories",
        "dataset_prefix": "nebius-swe-agent-trajectories",
        "default_repo": "DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes",
    },
    "nnetnav-live": {
        "subset": "nnetnav-live",
        "dataset_prefix": "nnetnav-live",
        "default_repo": "DCAgent/neulab-nnetnav-live-sandboxes",
    },
    "nnetnav-wa": {
        "subset": "nnetnav-wa",
        "dataset_prefix": "nnetnav-wa",
        "default_repo": "DCAgent/neulab-nnetnav-wa-sandboxes",
    },
    "openhands": {
        "subset": "openhands",
        "dataset_prefix": "openhands",
        "default_repo": "DCAgent/neulab-openhands-sandboxes",
    },
    "orca-agentinstruct": {
        "subset": "orca_agentinstruct",
        "dataset_prefix": "orca-agentinstruct",
        "default_repo": "DCAgent/neulab-orca-agentinstruct-sandboxes",
    },
    "swe-gym-openhands-sampled-trajectories": {
        "subset": "SWE-Gym_OpenHands-Sampled-Trajectories",
        "dataset_prefix": "swe-gym-openhands-sampled-trajectories",
        "default_repo": "DCAgent/neulab-swe-gym-openhands-sampled-trajectories-sandboxes",
    },
    "swe-smith-5ktrajectories": {
        "subset": "SWE-smith_5kTrajectories",
        "dataset_prefix": "swe-smith-5ktrajectories",
        "default_repo": "DCAgent/neulab-swe-smith-5ktrajectories-sandboxes",
    },
    "synatra": {
        "subset": "synatra",
        "dataset_prefix": "synatra",
        "default_repo": "DCAgent/neulab-synatra-sandboxes",
    },
}


def _count_task_directories(root: Path) -> int:
    return sum(1 for entry in root.iterdir() if entry.is_dir())


class NeulabGenerator(BaseDataGenerator):
    """BaseDataGenerator-compatible wrapper for neulab subsets."""

    parser_description = "Generate neulab agent-data-collection sandboxes dataset"
    default_target_repo: Optional[str] = None

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--task-type",
            type=str,
            choices=sorted(NEULAB_TASK_CONFIGS.keys()),
            default="code-feedback",
            help="Neulab subset to materialise.",
        )
        parser.add_argument(
            "--target-samples",
            type=int,
            default=10_000,
            help="Target number of tasks after up/down sampling.",
        )

    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        args = context.args
        task_type = getattr(args, "task_type", "code-feedback")
        if task_type not in NEULAB_TASK_CONFIGS:
            raise ValueError(
                f"Unknown task type '{task_type}'. "
                f"Expected one of: {', '.join(sorted(NEULAB_TASK_CONFIGS.keys()))}"
            )

        config = NEULAB_TASK_CONFIGS[task_type]
        subset_name = str(config["subset"])
        dataset_prefix = str(config["dataset_prefix"])

        limit = self._resolve_limit(request)
        target_samples = getattr(args, "target_samples", None)
        if target_samples is None or target_samples <= 0:
            target_samples = 10_000
        print(f"[1/3] Extracting questions from neulab subset '{subset_name}'")
        questions = extract_questions_from_subset(subset_name)
        if limit and limit > 0:
            questions = questions[:limit]
            target_samples = min(target_samples, limit)
        if target_samples <= 0:
            target_samples = max(1, len(questions) or 1)

        print(f"[2/3] Materialising tasks (prefix={dataset_prefix})")
        temp_tasks_dir = Path(generate_tasks_from_questions(questions, dataset_prefix))

        temp_paths = {temp_tasks_dir}
        current_root = temp_tasks_dir

        if _count_task_directories(current_root) < target_samples:
            upsampled = Path(
                upsample_tasks_directory(
                    str(current_root),
                    target_samples,
                    dataset_prefix=dataset_prefix,
                )
            )
            if upsampled != current_root:
                temp_paths.add(upsampled)
                current_root = upsampled

        if _count_task_directories(current_root) > target_samples:
            subsampled = Path(
                subsample_tasks_directory(
                    str(current_root),
                    target_samples,
                    dataset_prefix=dataset_prefix,
                )
            )
            if subsampled != current_root:
                temp_paths.add(subsampled)
                current_root = subsampled

        print("[3/3] Finalising dataset output")
        final_path = finalize_dataset_output(str(current_root), getattr(args, "output_dir", None))

        uploaded_repo: Optional[str] = None
        target_repo = getattr(args, "target_repo", None) or str(config.get("default_repo", ""))
        if target_repo == "":
            target_repo = None
        if not getattr(args, "no_upload", False) and target_repo:
            upload_tasks_to_hf(str(final_path), target_repo)
            uploaded_repo = target_repo

        for candidate in temp_paths:
            if str(candidate) != str(final_path):
                shutil.rmtree(candidate, ignore_errors=True)

        artifacts = {
            "task_type": task_type,
            "subset": subset_name,
            "target_samples": target_samples,
            "questions": len(questions),
            "dataset_prefix": dataset_prefix,
        }

        return GenerationResult(
            dataset_path=str(final_path),
            status=GenerationStatus.SUCCESS,
            uploaded_repo=uploaded_repo,
            artifacts=artifacts,
        )


def main() -> None:
    NeulabGenerator().run()


if __name__ == "__main__":
    main()
