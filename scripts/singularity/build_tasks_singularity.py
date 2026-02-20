#!/usr/bin/env python3
"""
Build Harbor tasks into docker-archive tarballs (via podman-hpc) for use with
the Apptainer/Singularity backend. Copies tasks and rewrites docker_image to the
tar path so Harbor can load it with oci-archive://<tar>.

Usage:
  python build_tasks_singularity.py \
    --tasks-dir /path/to/tasks \
    --output-images-dir /path/to/output/images \
    --output-tasks-dir /path/to/output/tasks \
    [--tag-prefix harbor-task] \
    [--skip-migrate] \
    [--verbose]

Notes:
  - Requires podman-hpc in PATH.
  - The Apptainer backend has been updated to treat docker_image pointing to an
    existing file as an oci-archive tar and will load it automatically.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib
import toml


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
    text = text.lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "task"


def build_to_tar(task_dir: Path, tag: str, output_images_dir: Path, skip_migrate: bool, verbose: bool) -> Path:
    run(["podman-hpc", "build", "-t", tag, "-f", "Dockerfile", "."], cwd=task_dir, verbose=verbose)
    if not skip_migrate:
        run(["podman-hpc", "migrate", tag], cwd=task_dir, verbose=verbose)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_images_dir / f"{tag.replace(':', '_')}.tar"
    run(["podman-hpc", "save", "-o", str(tar_path), tag], cwd=task_dir, verbose=verbose)
    return tar_path


def main():
    ap = argparse.ArgumentParser(description="Prebuild Harbor tasks into tar archives for Apptainer backend.")
    ap.add_argument("--tasks-dir", required=True, type=Path)
    ap.add_argument("--output-images-dir", required=True, type=Path)
    ap.add_argument("--output-tasks-dir", required=True, type=Path)
    ap.add_argument("--tag-prefix", default="harbor-task")
    ap.add_argument("--skip-migrate", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tasks_dir: Path = args.tasks_dir
    if not tasks_dir.is_dir():
        sys.exit(f"Tasks dir not found: {tasks_dir}")

    dockerfiles = sorted(p for p in tasks_dir.rglob("Dockerfile") if p.is_file())
    if not dockerfiles:
        print("No tasks with Dockerfile found.")
        return

    for dockerfile_path in dockerfiles:
        rel = dockerfile_path.relative_to(tasks_dir)
        task_root_rel = rel.parts[0]
        task_root = tasks_dir / task_root_rel
        env_dir = dockerfile_path.parent
        env_rel = env_dir.relative_to(task_root)

        task_slug = sanitize_component(task_root_rel)
        env_slug = sanitize_component("-".join(env_rel.parts)) if env_rel.parts else "env"
        tag = f"{args.tag_prefix}-{task_slug}-{env_slug}:latest"

        print(f"[task] {task_root_rel} (env {env_rel}) -> tag {tag}")

        tar_path = build_to_tar(env_dir, tag, args.output_images_dir, args.skip_migrate, args.verbose)

        dest_task_dir = args.output_tasks_dir / task_root_rel
        if dest_task_dir.exists():
            shutil.rmtree(dest_task_dir)
        shutil.copytree(task_root, dest_task_dir)

        dest_env_dir = dest_task_dir / env_rel
        dest_dockerfile = dest_env_dir / "Dockerfile"
        if dest_dockerfile.exists():
            dest_dockerfile.unlink()
        (dest_env_dir / "DOCKER_IMAGE").write_text(str(tar_path) + "\n")

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
