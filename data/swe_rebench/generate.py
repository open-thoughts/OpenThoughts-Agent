"""
Generate SWE-rebench dataset in sandboxes style.

Creates Harbor-compatible task directories from nebius/SWE-rebench instances.
Each task gets a pre-built Docker image, the issue's problem statement as
instruction, a test harness that applies the test_patch and grades via
swebench, and the gold patch as the reference solution.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from datasets import load_dataset

from data.swe_rebench.task_templates import (
    TASK_TOML,
    render_instruction_md,
    render_repo_dockerfile,
    render_test_sh,
    render_test_state_py,
    render_solution_script,
    serialize_config,
)
from data.swe_rebench.utils import get_test_cmd


def create_sandboxed_task(datum: dict, out_root: Path, idx: int) -> None:
    """Create one sandbox directory for a SWE-rebench instance."""
    d = out_root / f"swe_rebench-{idx:05d}"

    (d / "environment").mkdir(parents=True, exist_ok=True)
    (d / "solution").mkdir(parents=True, exist_ok=True)
    (d / "tests").mkdir(parents=True, exist_ok=True)

    # task.toml
    (d / "task.toml").write_text(TASK_TOML, encoding="utf-8")

    # instruction.md
    instruction = render_instruction_md(datum["problem_statement"])
    (d / "instruction.md").write_text(instruction, encoding="utf-8")

    # environment/Dockerfile
    dockerfile = render_repo_dockerfile(docker_image=datum["docker_image"])
    (d / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")

    # solution/solve.sh (gold patch)
    patch = datum.get("patch", "")
    if patch and patch.strip():
        solve_script = render_solution_script(patch)
        (d / "solution" / "solve.sh").write_text(solve_script, encoding="utf-8")

    # tests/config.json (full instance data for the grading harness)
    (d / "tests" / "config.json").write_text(
        serialize_config(datum),
        encoding="utf-8",
    )

    # tests/test_patch.diff (applied at verification time, not visible to agent)
    test_patch = datum.get("test_patch", "")
    if test_patch and test_patch.strip():
        (d / "tests" / "test_patch.diff").write_text(test_patch, encoding="utf-8")

    # tests/test.sh
    test_cmd = get_test_cmd(datum)
    test_sh = render_test_sh(test_cmd)
    (d / "tests" / "test.sh").write_text(test_sh, encoding="utf-8")

    # tests/test_state.py (grading logic, uses swebench)
    test_state = render_test_state_py()
    (d / "tests" / "test_state.py").write_text(test_state, encoding="utf-8")


def create_sandboxed_tasks(
    limit: int = 10_000,
    offset: int = 0,
    split: str = "filtered",
) -> Path:
    """Load SWE-rebench and emit sandbox task directories.

    Only instances with a ``docker_image`` are included.

    Returns:
        Path to temporary directory containing the generated tasks.
    """
    out_root = Path(tempfile.mkdtemp())

    ds = load_dataset("nebius/SWE-rebench", split=split)
    num_produced = 0

    for i in range(offset, len(ds)):
        if num_produced >= limit:
            break
        row = ds[i]

        # Skip instances without required fields
        if not row.get("docker_image"):
            continue
        if not row.get("problem_statement"):
            continue
        if not row.get("patch"):
            continue

        create_sandboxed_task(row, out_root, num_produced)
        num_produced += 1

    print(f"Generated {num_produced} SWE-rebench tasks in {out_root}")
    return out_root


def add_swe_rebench_args(parser: argparse.ArgumentParser) -> None:
    """Add SWE-rebench-specific CLI arguments."""
    parser.add_argument(
        "--limit", type=int, default=10_000,
        help="Max number of tasks to generate",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Dataset row offset to start from",
    )
    parser.add_argument(
        "--split", type=str, default="filtered",
        choices=["filtered", "test"],
        help="Dataset split: 'filtered' (6.5k) or 'test' (21k)",
    )
    parser.add_argument(
        "--subsample-count", type=int, default=10_000,
        help="Number of tasks to retain after subsampling",
    )
    parser.add_argument(
        "--repo-id", type=str,
        default="mlfoundations-dev/swe_rebench-sandboxes",
        help="HuggingFace repo id for upload",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip the upload step",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate SWE-rebench sandboxes and upload to HF",
    )
    add_swe_rebench_args(p)
    return p.parse_args()


def main() -> None:
    # Lazy import — data.commons has heavy transitive deps (google-cloud, harbor, …)
    from data.commons import upload_tasks_to_hf, subsample_tasks_directory

    args = parse_args()

    print(
        f"[1/2] Generating SWE-rebench sandboxes "
        f"(split={args.split}, limit={args.limit}, offset={args.offset})"
    )
    tasks_dir = create_sandboxed_tasks(
        limit=args.limit,
        offset=args.offset,
        split=args.split,
    )

    final_dir = subsample_tasks_directory(
        source_dir=tasks_dir,
        num_samples=args.subsample_count,
    )

    if not args.no_upload:
        upload_tasks_to_hf(
            dataset_path=final_dir,
            repo_id=args.repo_id,
        )
    else:
        print(f"Skipping upload. Dataset at: {final_dir}")


if __name__ == "__main__":
    main()
