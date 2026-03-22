#!/usr/bin/env python3
"""
Patch logsmith-easy-500 tasks to:
1. Move Dockerfile into environment/Dockerfile (expected by validate_and_upload_from_hf.py)
2. Move test.sh into tests/test.sh (expected by harbor)
3. Remove COPY seeds/ from the Dockerfile (reduces 500 unique snapshots to 1)
4. Copy seeds/ into setup_files/ so Harbor uploads data to /workspace/ before agent runs

Original flat task structure:
  task_000001/
    Dockerfile
    instruction.md
    task.json
    task.toml
    test.sh
    seeds/
      data/
        logs/
        decoys/

Patched structure:
  task_000001/
    instruction.md
    task.json
    task.toml
    environment/
      Dockerfile        <- moved here, COPY seeds/ removed
    tests/
      test.sh           <- moved here
    setup_files/
      data/             <- copied from seeds/data/ for Harbor upload
        logs/
        decoys/
    seeds/              <- left in place (untouched)

Usage:
    python patch_logsmith_tasks.py /path/to/tasks

    # Write to a separate output directory (leaves originals untouched)
    python patch_logsmith_tasks.py /path/to/tasks --output-dir /path/to/patched

    # Dry run
    python patch_logsmith_tasks.py /path/to/tasks --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

DOCKERFILE_TEMPLATE = """\
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8 LANG=C.UTF-8 TZ=UTC LOG_DIR=/logs/verifier
WORKDIR /workspace

RUN apt-get update \\
    && apt-get install -y --no-install-recommends tmux=3.2a-4ubuntu0.2 \\
    && rm -rf /var/lib/apt/lists/*

# Seed data is uploaded to /workspace/ by Harbor before the agent runs
# (via setup_files/ mechanism) — no COPY needed here.

# Non-root user (safer by default).
RUN useradd -m -u 1000 agent \\
    && mkdir -p /logs/verifier /output \\
    && chown -R agent:agent /workspace /logs /output
USER agent
"""


# ---------------------------------------------------------------------------
# Patching logic
# ---------------------------------------------------------------------------

def patch_task(
    task_dir: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, bool | str]:
    """Patch a single task directory. Returns dict of what was changed."""
    changes: dict[str, bool | str] = {}

    # Validate expected flat structure
    dockerfile_src = task_dir / "Dockerfile"
    test_sh_src = task_dir / "test.sh"
    seeds_dir = task_dir / "seeds"

    if not dockerfile_src.exists():
        return {"error": True, "reason": "no Dockerfile"}
    if not seeds_dir.exists():
        return {"error": True, "reason": "no seeds/ dir"}

    # Determine target directory
    if output_dir:
        target = output_dir / task_dir.name
        if not dry_run:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(task_dir, target)
    else:
        target = task_dir

    # --- 1. environment/Dockerfile: move + replace with generic version ---
    env_dir = target / "environment"
    target_dockerfile = env_dir / "Dockerfile"

    if dry_run:
        changes["environment/Dockerfile"] = True
    else:
        env_dir.mkdir(parents=True, exist_ok=True)
        target_dockerfile.write_text(DOCKERFILE_TEMPLATE)
        # Remove the old flat Dockerfile
        old_dockerfile = target / "Dockerfile"
        if old_dockerfile.exists():
            old_dockerfile.unlink()
        # Remove any seeds/ that got copied under environment/ by shutil.copytree
        env_seeds = env_dir / "seeds"
        if env_seeds.exists():
            shutil.rmtree(env_seeds)
        changes["environment/Dockerfile"] = True

    # --- 2. tests/test.sh: move test.sh into tests/ subdir ---
    tests_dir = target / "tests"
    target_test_sh = tests_dir / "test.sh"
    old_test_sh = target / "test.sh"

    if target_test_sh.exists():
        changes["tests/test.sh"] = False  # already moved
    elif dry_run:
        changes["tests/test.sh"] = bool(test_sh_src.exists())
    else:
        if old_test_sh.exists():
            tests_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_test_sh), str(target_test_sh))
            changes["tests/test.sh"] = True
        else:
            changes["tests/test.sh"] = False

    # --- 3. setup_files/: copy seeds/data/ so Harbor uploads it ---
    # Harbor uploads setup_files/ contents to /workspace/ before the agent
    # runs. We copy seeds/data/ -> setup_files/data/ so the workspace layout
    # the agent sees is identical to the original (data/ at /workspace/data/).
    target_seeds_data = target / "seeds" / "data"
    target_setup_files = target / "setup_files"
    already_patched = target_setup_files.exists() and any(target_setup_files.iterdir())

    if already_patched:
        changes["setup_files"] = False
    elif dry_run:
        changes["setup_files"] = True
    else:
        target_setup_files.mkdir(parents=True, exist_ok=True)
        if target_seeds_data.exists():
            shutil.copytree(target_seeds_data, target_setup_files / "data")
        changes["setup_files"] = True

    return changes


def main():
    parser = argparse.ArgumentParser(
        description="Patch logsmith tasks: restructure to environment/Dockerfile + tests/test.sh, remove COPY seeds/",
    )
    parser.add_argument(
        "tasks_dir", help="Root directory containing task folders")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write patched tasks to this directory (default: patch in-place)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_dir)
    if not tasks_root.is_dir():
        raise SystemExit(f"Not a directory: {tasks_root}")

    task_dirs = sorted(
        d for d in tasks_root.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    )
    print(f"Found {len(task_dirs)} tasks in {tasks_root}")

    if args.output_dir and not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {args.output_dir}")

    totals: dict[str, int] = {}
    errors = 0
    for td in task_dirs:
        changes = patch_task(
            td, output_dir=args.output_dir, dry_run=args.dry_run)
        if changes.get("error"):
            errors += 1
            print(f"  ERROR {td.name}: {changes.get('reason')}")
            continue
        for k, v in changes.items():
            if v:
                totals[k] = totals.get(k, 0) + 1

    action = "Would patch" if args.dry_run else "Patched"
    print(f"\n{action}:")
    for filename, count in sorted(totals.items()):
        print(f"  {filename}: {count}/{len(task_dirs)}")
    if errors:
        print(f"  Errors: {errors}")

    # Report unique Dockerfiles after patching
    if not args.dry_run:
        out_root = args.output_dir or tasks_root
        dockerfiles: set[str] = set()
        for td in sorted(out_root.iterdir()):
            df = td / "environment" / "Dockerfile"
            if df.exists():
                dockerfiles.add(df.read_text())
        print(f"\nUnique Dockerfiles: {len(dockerfiles)}")


if __name__ == "__main__":
    main()
