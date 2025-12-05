"""CLI for running batched nl2bash teacher generations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from data.generation.task_generation import (
    DaytonaTaskGenerationExecutor,
    TaskGenerationSpec,
)
from data.nl2bash import generate as nl2bash_generate

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a batch of nl2bash teacher generations.",
    )
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks to generate.")
    parser.add_argument("--first-seed", type=int, default=0, help="Starting seed value.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to sample.")
    parser.add_argument(
        "--dataset-prefix-base",
        type=str,
        default="nl2bash",
        help="Base prefix for generated dataset directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/nl2bash/local_runs/tasks"),
        help="Root directory for generated tasks.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("data/nl2bash/local_runs/jobs"),
        help="Directory for Daytona job artefacts.",
    )
    parser.add_argument(
        "--task-max-concurrent",
        type=int,
        default=1,
        help="Maximum number of concurrent Daytona task generations.",
    )
    parser.add_argument(
        "--teacher-agent",
        type=str,
        default="terminus-2",
        help="Teacher agent name.",
    )
    parser.add_argument(
        "--teacher-agent-model",
        type=str,
        default="gpt-5-mini",
        help="Teacher agent model name.",
    )
    parser.add_argument(
        "--teacher-agent-kwargs",
        type=str,
        default='{"max_episodes": 8}',
        help="JSON string for teacher agent kwargs.",
    )
    parser.add_argument(
        "--teacher-env",
        type=str,
        default="daytona",
        help="Teacher environment type.",
    )
    parser.add_argument(
        "--teacher-no-force-build",
        action="store_true",
        help="Do not force rebuild of the Daytona environment.",
    )
    parser.add_argument(
        "--teacher-no-delete-env",
        action="store_true",
        help="Do not delete the Daytona environment after execution.",
    )
    parser.add_argument(
        "--teacher-n-concurrent",
        type=int,
        default=1,
        help="Number of concurrent trials per teacher job.",
    )
    parser.add_argument(
        "--teacher-timeout-multiplier",
        type=float,
        default=0.2,
        help="Timeout multiplier for the teacher job.",
    )
    return parser.parse_args()


def _build_specs(
    *,
    num_tasks: int,
    first_seed: int,
    output_root: Path,
    jobs_dir: Path,
    dataset_prefix_base: str,
) -> List[TaskGenerationSpec]:
    specs: List[TaskGenerationSpec] = []
    for index in range(num_tasks):
        seed = first_seed + index
        output_dir = output_root / f"task_{index}"
        dataset_prefix = f"{dataset_prefix_base}-{index}"
        specs.append(
            TaskGenerationSpec(
                seed=seed,
                output_dir=output_dir,
                dataset_prefix=dataset_prefix,
                jobs_dir=jobs_dir,
            )
        )
    return specs


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be a positive integer.")
    if args.task_max_concurrent <= 0:
        raise ValueError("--task-max-concurrent must be a positive integer.")
    if args.teacher_n_concurrent <= 0:
        raise ValueError("--teacher-n-concurrent must be a positive integer.")

    output_root = args.output_root.resolve()
    jobs_dir = args.jobs_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    parser = nl2bash_generate.build_arg_parser()
    base_args = parser.parse_args([])
    base_args.split = args.split
    base_args.teacher_agent = args.teacher_agent
    base_args.teacher_agent_model = args.teacher_agent_model
    base_args.teacher_agent_kwargs = args.teacher_agent_kwargs
    base_args.teacher_env = args.teacher_env
    base_args.teacher_no_force_build = args.teacher_no_force_build
    base_args.teacher_no_delete_env = args.teacher_no_delete_env
    base_args.teacher_n_concurrent = args.teacher_n_concurrent
    base_args.teacher_timeout_multiplier = args.teacher_timeout_multiplier

    specs = _build_specs(
        num_tasks=args.num_tasks,
        first_seed=args.first_seed,
        output_root=output_root,
        jobs_dir=jobs_dir,
        dataset_prefix_base=args.dataset_prefix_base,
    )

    executor = DaytonaTaskGenerationExecutor(
        base_args=base_args,
        max_concurrent=args.task_max_concurrent,
        generate_fn=nl2bash_generate.generate_tasks,
    )
    results = executor.run(specs)

    successes = sum(1 for result in results if result.success)
    LOGGER.info(
        "Completed %s/%s generations successfully",
        successes,
        len(results),
    )

    summary_path = output_root / "summary.json"
    serialisable = []
    for result in results:
        entry = {
            "seed": result.spec.seed,
            "output_dir": str(result.spec.output_dir),
            "dataset_prefix": result.spec.dataset_prefix,
            "jobs_dir": str(result.spec.jobs_dir),
            "success": result.success,
            "dataset_dir": str(result.dataset_dir) if result.dataset_dir else None,
            "exception_path": str(result.exception_path) if result.exception_path else None,
        }
        serialisable.append(entry)

    summary_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
