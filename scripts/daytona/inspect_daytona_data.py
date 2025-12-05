#!/usr/bin/env python3
"""
Utility script to build a Daytona sandbox from a local task, upload the same
assets the sandboxes orchestrator would stage, and report on key directories.

Example usage (run from repo root):

    python scripts/daytona/inspect_daytona_data.py \
        --dockerfile dev_set_71_tasks/3dd9a49b-a9a4-44ac-932f-79899b92/environment/Dockerfile
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Iterable, Sequence

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    FileUpload,
    Image,
    Resources,
)
from daytona.common.errors import DaytonaError

# Remote directories the sandboxes orchestrator normally prepares.
LOG_DIRS_TO_CREATE: Sequence[str] = (
    "/logs/agent",
    "/logs/verifier",
    "/output",
)

DEFAULT_REMOTE_DIRS: Sequence[str] = (
    "/workdir/data",
    "/tests",
    "/solution",
    "/logs/agent",
    "/logs/verifier",
    "/output",
)


async def _list_remote_dir(sandbox, remote_dir: str) -> list[tuple[str, int, bool]]:
    """Return a list of (path, size, is_dir) entries under remote_dir."""
    try:
        search_result = await sandbox.fs.search_files(remote_dir, "*")
    except DaytonaError as exc:
        print(f"[warn] Failed to list {remote_dir}: {exc}")
        return []

    entries: list[tuple[str, int, bool]] = []
    files = getattr(search_result, "files", None) or []
    for remote_path in sorted(files):
        try:
            info = await sandbox.fs.get_file_info(remote_path)
        except DaytonaError as exc:
            print(f"[warn] Could not stat {remote_path}: {exc}")
            continue
        size = int(getattr(info, "size", 0) or 0)
        is_dir = bool(getattr(info, "is_dir", False))
        entries.append((remote_path, size, is_dir))
    return entries


async def _ensure_remote_dirs(sandbox, dirs: Sequence[str]) -> None:
    for directory in dirs:
        try:
            await sandbox.process.exec(command=f"mkdir -p {directory}")
        except DaytonaError as exc:
            print(f"[warn] Failed to ensure remote directory {directory}: {exc}")


def _gather_uploads(src: Path, dest_root: str) -> list[FileUpload]:
    uploads: list[FileUpload] = []
    for path in src.rglob("*"):
        if path.is_file():
            relative = path.relative_to(src)
            uploads.append(
                FileUpload(
                    source=str(path),
                    destination=str(Path(dest_root) / relative),
                )
            )
    return uploads


async def _upload_dir_if_present(sandbox, src: Path, dest_root: str) -> None:
    if not src.exists():
        print(f"[info] Skipping upload for {src} (missing).")
        return

    uploads = _gather_uploads(src, dest_root)
    if not uploads:
        print(f"[info] Skipping upload for {src} (no files).")
        return

    try:
        await sandbox.fs.upload_files(files=uploads)
        print(f"[info] Uploaded {len(uploads)} files from {src} to {dest_root}")
    except DaytonaError as exc:
        print(f"[warn] Failed to upload {src} to {dest_root}: {exc}")


async def inspect_dockerfile(
    dockerfile: Path,
    remote_dirs: Sequence[str],
    *,
    cpu: int,
    memory_gb: int,
    disk_gb: int,
    gpu: int,
    timeout: int,
) -> None:
    """Build the given Dockerfile, upload auxiliary assets, and inspect directories."""
    dockerfile = dockerfile.resolve()
    if not dockerfile.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")

    print(f"[info] Building sandbox image from {dockerfile}")

    resources = Resources(
        cpu=max(cpu, 1),
        memory=max(memory_gb, 1),
        disk=max(disk_gb, 1),
        gpu=max(gpu, 0),
    )

    params = CreateSandboxFromImageParams(
        image=Image.from_dockerfile(dockerfile),
        auto_delete_interval=0,
        resources=resources,
    )

    daytona = AsyncDaytona()
    sandbox = None
    try:
        sandbox = await daytona.create(params=params, timeout=timeout)
        print(f"[info] Sandbox {sandbox.id} started.")

        # Prepare remote paths similar to the sandboxes runtime.
        await _ensure_remote_dirs(sandbox, LOG_DIRS_TO_CREATE)

        environment_dir = dockerfile.parent
        task_dir = environment_dir.parent

        await _upload_dir_if_present(sandbox, task_dir / "tests", "/tests")
        await _upload_dir_if_present(sandbox, task_dir / "solution", "/solution")

        for remote_dir in remote_dirs:
            print(f"[info] Inspecting {remote_dir!r} …")
            entries = await _list_remote_dir(sandbox, remote_dir)
            if not entries:
                print(f"[warn] No files found under {remote_dir}")
            else:
                print(f"[info] Contents of {remote_dir}:")
                for path, size, is_dir in entries:
                    kind = "<DIR>" if is_dir else f"{size} bytes"
                    print(f"  {path}  {kind}")
    finally:
        if sandbox is not None:
            print(f"[info] Cleaning up sandbox {sandbox.id}")
            await sandbox.delete()
        await daytona.close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Daytona sandbox directories after build."
    )
    parser.add_argument(
        "--dockerfile",
        required=True,
        type=Path,
        help="Path to the Dockerfile to build.",
    )
    parser.add_argument(
        "--remote-dir",
        action="append",
        help=(
            "Remote directory to inspect inside the sandbox. "
            "May be provided multiple times. Defaults to a set of key directories."
        ),
    )
    parser.add_argument("--cpu", type=int, default=2, help="Number of CPUs to request.")
    parser.add_argument(
        "--memory-gb", type=int, default=4, help="Memory to request (GB)."
    )
    parser.add_argument(
        "--disk-gb", type=int, default=10, help="Disk space to request (GB)."
    )
    parser.add_argument("--gpu", type=int, default=0, help="Number of GPUs to request.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Build timeout in seconds (default: 600).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(
            inspect_dockerfile(
                dockerfile=args.dockerfile,
                remote_dirs=tuple(args.remote_dir or DEFAULT_REMOTE_DIRS),
                cpu=args.cpu,
                memory_gb=args.memory_gb,
                disk_gb=args.disk_gb,
                gpu=args.gpu,
                timeout=args.timeout,
            )
        )
    except KeyboardInterrupt:
        print("[warn] Aborted by user.")


if __name__ == "__main__":
    main()
