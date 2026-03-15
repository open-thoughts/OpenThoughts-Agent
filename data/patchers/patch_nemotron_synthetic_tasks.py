#!/usr/bin/env python3
"""
Patch nvidia/Nemotron-Terminal-Synthetic-Tasks to use shared base images
instead of per-task unique images, making it compatible with Daytona's
snapshot system for RL training.

The original dataset has two sources of unique images per task:
  1. task.toml specifies `docker_image` pointing to a private Nvidia Gitlab
     registry (inaccessible outside Nvidia), causing failures on Daytona.
  2. Dockerfile contains `COPY files/ /app/` which embeds task-specific data
     files into the image, making every image unique.

After patching, all tasks within the same category share one Dockerfile,
resulting in ~10 unique images total (one per category: data_science,
security, debugging, etc.) — well within Daytona's snapshot limit.

Changes made per task:
  1. environment/Dockerfile  — Remove `COPY files/ /app/` line; fix WORKDIR to /app
  2. task.toml               — Remove `docker_image` (use Dockerfile build instead)
  3. environment/files/      — Moved to setup_files/ (Harbor uploads these to
                               /setup_files/ in the container before agent runs)
  4. instruction.md          — Prepend setup preamble for tasks that have data files

Usage:
    # Patch a local directory of extracted tasks:
    python patch_nemotron_synthetic_tasks.py /path/to/tasks

    # Write to a separate output directory (leaves originals untouched):
    python patch_nemotron_synthetic_tasks.py /path/to/tasks --output-dir /path/to/patched

    # Download, extract, and patch directly from HuggingFace:
    python patch_nemotron_synthetic_tasks.py --hf-dataset nvidia/Nemotron-Terminal-Synthetic-Tasks --output-dir /path/to/patched

    # Dry run (show what would change without writing):
    python patch_nemotron_synthetic_tasks.py /path/to/tasks --dry-run
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import tarfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# Preamble added to instruction.md when task has data files in setup_files/.
# Harbor uploads setup_files/* to /setup_files/ in the container before
# the agent runs. This preamble tells the agent to copy them to /app/.
SETUP_PREAMBLE = """\
## Setup

Data files for this task have been pre-loaded into `/setup_files/`.
Before starting, copy them to your working directory:

```bash
cp -r /setup_files/. /app/
```

---

"""

ALREADY_PATCHED_MARKER = "Data files for this task have been pre-loaded into"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patch_dockerfile(content: str) -> tuple[str, bool]:
    """Remove COPY files/ /app/ line and fix WORKDIR to /app.

    Returns (patched_content, was_changed).
    """
    lines = content.splitlines(keepends=True)
    new_lines = []
    changed = False

    for line in lines:
        stripped = line.strip()
        # Remove any COPY that copies into /app (per-task files baked in at build time)
        if re.match(r"COPY\s+files/\s+/app/", stripped):
            changed = True
            continue
        # Fix WORKDIR / → WORKDIR /app (Nvidia's Dockerfiles set WORKDIR / which is wrong)
        if re.match(r"WORKDIR\s+/$", stripped):
            new_lines.append("WORKDIR /app\n")
            changed = True
            continue
        new_lines.append(line)

    return "".join(new_lines), changed


def patch_task_toml(content: str) -> tuple[str, bool]:
    """Remove the docker_image line from task.toml.

    The original tasks point to a private Nvidia Gitlab registry. Removing this
    line causes Harbor to build from the Dockerfile instead.

    Returns (patched_content, was_changed).
    """
    lines = content.splitlines(keepends=True)
    new_lines = []
    changed = False

    for line in lines:
        if re.match(r"\s*docker_image\s*=", line):
            changed = True
            continue
        new_lines.append(line)

    return "".join(new_lines), changed


# ---------------------------------------------------------------------------
# Per-task patching
# ---------------------------------------------------------------------------

def patch_task(
    task_dir: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, bool | str]:
    """Patch a single task directory.

    Returns a dict describing what was changed. Special keys:
      "error" / "reason" — present when the task was skipped.
    """
    changes: dict[str, bool | str] = {}

    # Validate basic structure
    if not (task_dir / "instruction.md").exists():
        return {"error": True, "reason": "no instruction.md"}

    # Determine target (copy-then-patch vs in-place)
    if output_dir is not None:
        target = output_dir / task_dir.name
        if not dry_run:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(task_dir, target)
    else:
        target = task_dir

    # ------------------------------------------------------------------
    # 1. Dockerfile — remove COPY files/ /app/, fix WORKDIR
    # ------------------------------------------------------------------
    dockerfile_path = target / "environment" / "Dockerfile"
    if dockerfile_path.exists():
        original = dockerfile_path.read_text()
        patched, changed = patch_dockerfile(original)
        changes["Dockerfile"] = changed
        if changed and not dry_run:
            dockerfile_path.write_text(patched)
    else:
        changes["Dockerfile"] = False

    # ------------------------------------------------------------------
    # 2. task.toml — remove docker_image line
    # ------------------------------------------------------------------
    toml_path = target / "task.toml"
    if toml_path.exists():
        original = toml_path.read_text()
        patched, changed = patch_task_toml(original)
        changes["task.toml"] = changed
        if changed and not dry_run:
            toml_path.write_text(patched)
    else:
        changes["task.toml"] = False

    # ------------------------------------------------------------------
    # 3. environment/files/ → setup_files/
    # ------------------------------------------------------------------
    env_files_dir = target / "environment" / "files"
    has_data_files = env_files_dir.is_dir() and any(env_files_dir.iterdir())
    changes["setup_files"] = has_data_files

    if has_data_files and not dry_run:
        setup_files_dir = target / "setup_files"
        setup_files_dir.mkdir(exist_ok=True)
        for item in env_files_dir.iterdir():
            dest = setup_files_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        # Remove original files dir (it's now in setup_files/)
        shutil.rmtree(env_files_dir)

    # ------------------------------------------------------------------
    # 4. instruction.md — prepend setup preamble if task has data files
    # ------------------------------------------------------------------
    instruction_path = target / "instruction.md"
    if has_data_files and instruction_path.exists():
        original = instruction_path.read_text()
        if ALREADY_PATCHED_MARKER in original:
            changes["instruction.md"] = False  # already patched
        elif dry_run:
            changes["instruction.md"] = True
        else:
            instruction_path.write_text(SETUP_PREAMBLE + original)
            changes["instruction.md"] = True
    else:
        changes["instruction.md"] = False

    return changes


# ---------------------------------------------------------------------------
# HuggingFace download + extraction helpers
# ---------------------------------------------------------------------------

def _iter_tasks_from_tar(tar_path: Path, out_dir: Path) -> list[Path]:
    """Extract tasks from a .tar.gz file, flattening the category subdirectory.

    The HF tarballs have structure:
        ./easy_5000/data_science/data_science_task_1766/...

    We extract each leaf task directory directly into out_dir as:
        out_dir/data_science_task_1766/...
    """
    task_dirs: list[Path] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        # Identify leaf task dirs (contain instruction.md)
        instruction_paths = {
            m.name for m in members if m.name.endswith("/instruction.md")
        }

        for instr_path in instruction_paths:
            task_rel = instr_path[: -len("/instruction.md")]  # e.g. ./easy_5000/data_science/task_1766
            task_name = Path(task_rel).name  # e.g. task_1766

            task_out = out_dir / task_name
            task_out.mkdir(parents=True, exist_ok=True)

            # Extract all members belonging to this task
            prefix = task_rel + "/"
            for member in members:
                if not (member.name == task_rel or member.name.startswith(prefix)):
                    continue
                # Compute relative path within the task
                rel = member.name[len(task_rel):].lstrip("/")
                if not rel:
                    continue

                dest = task_out / rel
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        dest.write_bytes(file_obj.read())

            task_dirs.append(task_out)

    return task_dirs


def download_and_extract_from_hf(dataset_name: str, out_dir: Path) -> list[Path]:
    """Download all tar.gz shards from a HuggingFace dataset and extract tasks.

    Returns list of extracted task directories.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("huggingface_hub is required. Run: pip install huggingface_hub")

    api = HfApi()
    all_files = list(api.list_repo_files(dataset_name, repo_type="dataset"))
    tar_files = [f for f in all_files if f.endswith(".tar.gz")]

    if not tar_files:
        raise SystemExit(f"No .tar.gz files found in {dataset_name}")

    print(f"Found {len(tar_files)} tar.gz shards in {dataset_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_task_dirs: list[Path] = []

    for i, hf_path in enumerate(tar_files, 1):
        print(f"  [{i}/{len(tar_files)}] Downloading {hf_path} ...")
        local_path = api.hf_hub_download(
            repo_id=dataset_name,
            filename=hf_path,
            repo_type="dataset",
        )
        shard_out = out_dir / f"_extracted_{i}"
        shard_out.mkdir(exist_ok=True)
        task_dirs = _iter_tasks_from_tar(Path(local_path), shard_out)
        all_task_dirs.extend(task_dirs)
        print(f"    Extracted {len(task_dirs)} tasks")

    return all_task_dirs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch Nemotron-Terminal-Synthetic-Tasks for shared Docker images",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "tasks_dir",
        nargs="?",
        help="Local directory containing extracted task folders",
    )
    source.add_argument(
        "--hf-dataset",
        metavar="REPO_ID",
        help="Download and extract from HuggingFace (e.g. nvidia/Nemotron-Terminal-Synthetic-Tasks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write patched tasks here (default: patch in-place)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing",
    )
    args = parser.parse_args()

    # ---- Resolve task directories ----
    if args.hf_dataset:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required when using --hf-dataset")
        extract_dir = args.output_dir / "_raw"
        print(f"Downloading from HuggingFace: {args.hf_dataset}")
        task_dirs = download_and_extract_from_hf(args.hf_dataset, extract_dir)
        tasks_root = extract_dir
    else:
        tasks_root = Path(args.tasks_dir)
        if not tasks_root.is_dir():
            raise SystemExit(f"Not a directory: {tasks_root}")
        task_dirs = sorted(
            d for d in tasks_root.iterdir()
            if d.is_dir() and (d / "instruction.md").exists()
        )

    if not task_dirs:
        raise SystemExit(f"No tasks found in {tasks_root}")

    print(f"Found {len(task_dirs)} tasks")

    if args.output_dir and not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Patch each task ----
    totals: dict[str, int] = {}
    errors = 0

    for td in task_dirs:
        result = patch_task(
            td,
            output_dir=args.output_dir if not args.hf_dataset else args.output_dir,
            dry_run=args.dry_run,
        )
        if result.get("error"):
            errors += 1
            continue
        for k, v in result.items():
            if v:
                totals[k] = totals.get(k, 0) + 1

    # ---- Report ----
    action = "Would patch" if args.dry_run else "Patched"
    print(f"\n{action}:")
    for filename, count in sorted(totals.items()):
        print(f"  {filename}: {count}/{len(task_dirs)}")
    if errors:
        print(f"  Errors (skipped): {errors}")

    # Count unique Dockerfiles in output
    if not args.dry_run:
        out_root = args.output_dir or tasks_root
        dockerfiles: set[str] = set()
        for td in out_root.rglob("environment/Dockerfile"):
            dockerfiles.add(td.read_text())
        print(f"\nUnique Dockerfiles after patching: {len(dockerfiles)}")
        if len(dockerfiles) <= 10:
            print("✓ Within Daytona's snapshot limit (≤10)")
        else:
            print(f"⚠ Exceeds recommended limit of 10 — consider consolidating Dockerfiles")


if __name__ == "__main__":
    main()
