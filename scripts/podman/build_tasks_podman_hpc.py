#!/usr/bin/env python3
"""
Build a set of Harbor tasks with Dockerfiles using podman-hpc, save the images,
and emit a task directory that references the prebuilt images instead of Dockerfiles.

Usage:
  python build_tasks_podman_hpc.py \
    --tasks-dir /path/to/tasks \
    --output-images-dir /path/to/output/images \
    --output-tasks-dir /path/to/output/tasks \
    [--tag-prefix harbor-task] \
    [--skip-migrate] \
    [--verbose]

What it does:
  - For each subdirectory under --tasks-dir that contains a Dockerfile:
      * podman-hpc build -t <tag> .
      * podman-hpc migrate <tag>          (unless --skip-migrate)
      * podman-hpc save -o <task>.tar <tag>
  - Copies the task directory to --output-tasks-dir/<task> and writes a file
    DOCKER_IMAGE containing the tag, so Harbor can use the prebuilt image and
    skip Dockerfile builds. The original Dockerfile is removed in the output copy
    to prevent accidental rebuilds.

Notes:
  - Requires podman-hpc in PATH (NERSC Perlmutter module: `module load podman-hpc`).
  - Image tags are local names; if you want to push to a registry, do that separately.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib
import toml
import re


def run(cmd: list[str], cwd: Path, verbose: bool):
    if verbose:
        print(f"+ {' '.join(cmd)} (cwd={cwd})")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def sanitize_component(text: str) -> str:
    # Lowercase, replace disallowed chars with '-', collapse repeats, trim hyphens.
    text = text.lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "task"


def build_task(task_dir: Path, tag: str, output_images_dir: Path, skip_migrate: bool, verbose: bool):
    # podman-hpc build
    run(["podman-hpc", "build", "-t", tag, "-f", "Dockerfile", "."], cwd=task_dir, verbose=verbose)
    if not skip_migrate:
        run(["podman-hpc", "migrate", tag], cwd=task_dir, verbose=verbose)
    # save image
    output_images_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_images_dir / f"{tag.replace(':', '_')}.tar"
    run(["podman-hpc", "save", "-o", str(tar_path), tag], cwd=task_dir, verbose=verbose)
    return tar_path


def copy_task(task_dir: Path, dest_dir: Path, tag: str):
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(task_dir, dest_dir)
    dockerfile = dest_dir / "Dockerfile"
    if dockerfile.exists():
        dockerfile.unlink()
    (dest_dir / "DOCKER_IMAGE").write_text(tag + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build Harbor tasks with podman-hpc and emit image-based tasks.")
    ap.add_argument("--tasks-dir", required=True, type=Path, help="Input tasks directory (each subdir is a task with Dockerfile).")
    ap.add_argument("--output-images-dir", required=True, type=Path, help="Where to store saved images (.tar).")
    ap.add_argument("--output-tasks-dir", required=True, type=Path, help="Where to write task copies that reference prebuilt images.")
    ap.add_argument("--tag-prefix", default="harbor-task", help="Prefix for image tags (tag will be <prefix>-<taskname>:latest).")
    ap.add_argument("--skip-migrate", action="store_true", help="Skip podman-hpc migrate step.")
    ap.add_argument("--verbose", action="store_true", help="Print commands.")
    args = ap.parse_args()

    tasks_dir: Path = args.tasks_dir
    if not tasks_dir.is_dir():
        sys.exit(f"Tasks dir not found: {tasks_dir}")

    # Find all Dockerfiles (recursive)
    dockerfiles = sorted(p for p in tasks_dir.rglob("Dockerfile") if p.is_file())

    if not dockerfiles:
        print("No tasks with Dockerfile found.")
        return

    for dockerfile_path in dockerfiles:
        # task root = first path component under tasks_dir
        rel = dockerfile_path.relative_to(tasks_dir)
        task_root_rel = rel.parts[0]
        task_root = tasks_dir / task_root_rel
        env_dir = dockerfile_path.parent
        env_rel = env_dir.relative_to(task_root)  # path from task root to env dir

        task_slug = sanitize_component(task_root_rel)
        env_slug = sanitize_component("-".join(env_rel.parts)) if env_rel.parts else "env"
        tag = f"{args.tag_prefix}-{task_slug}-{env_slug}:latest"

        print(f"[task] {task_root_rel} (env {env_rel}) -> tag {tag}")

        # Build from original env dir
        tar_path = build_task(env_dir, tag, args.output_images_dir, args.skip_migrate, args.verbose)

        # Copy full task tree, then mutate env dir
        dest_task_dir = args.output_tasks_dir / task_root_rel
        if dest_task_dir.exists():
            shutil.rmtree(dest_task_dir)
        shutil.copytree(task_root, dest_task_dir)

        dest_env_dir = dest_task_dir / env_rel
        dest_dockerfile = dest_env_dir / "Dockerfile"
        if dest_dockerfile.exists():
            dest_dockerfile.unlink()
        (dest_env_dir / "DOCKER_IMAGE").write_text(str(tar_path) + "\n")

        # Update task.toml to point to the prebuilt image
        task_toml = dest_task_dir / "task.toml"
        if task_toml.exists():
            try:
                data = tomllib.loads(task_toml.read_text())
                env_cfg = data.get("environment", {})
                env_cfg["docker_image"] = str(tar_path)
                data["environment"] = env_cfg
                task_toml.write_text(toml.dumps(data))
                print(f"  updated task.toml docker_image={tar_path}")
            except Exception as e:
                print(f"  warning: failed to update task.toml: {e}")
        else:
            print("  warning: task.toml not found; docker_image not recorded")

        print(f"  saved: {tar_path}")
        print(f"  task copy: {dest_env_dir} (Dockerfile removed, DOCKER_IMAGE written)")

    print("Done.")


if __name__ == "__main__":
    main()
