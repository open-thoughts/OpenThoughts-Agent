"""Lifecycle and reader helpers for image-backed Harbor trial artifacts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal


ARTIFACT_STORE_IMAGE = "artifact_store.img"
ARTIFACT_STORE_MOUNT = "artifact_mnt"
ARTIFACT_STORE_LOCK_SUFFIX = ".lock"
ARTIFACT_STORE_MOUNT_LEASE_SUFFIX = ".mount-lease"
ARTIFACT_STORE_AUTHORITY = "artifact_authority.json"
TRACE_JOBS_SUBDIR = "trace_jobs"
DEFAULT_IMAGE_SIZE = "1T"
DEFAULT_INODE_COUNT = 50_000_000
DEFAULT_MOUNT_ROOT = Path("/tmp/otagent-artifact-stores")


class ArtifactStoreBusyError(RuntimeError):
    """The image has an active writer and cannot be mounted on this host."""


@dataclass(frozen=True)
class ArtifactStorePaths:
    """Stable files and directories for one image-backed run."""

    image: Path
    mount: Path

    @property
    def lock(self) -> Path:
        return Path(f"{self.image}{ARTIFACT_STORE_LOCK_SUFFIX}")

    @property
    def trials(self) -> Path:
        return self.mount / TRACE_JOBS_SUBDIR


def mount_path_for_image(
    image_path: Path, *, mount_root: Path = DEFAULT_MOUNT_ROOT
) -> Path:
    """Return a stable node-local mount path for an artifact image."""
    image_path = Path(image_path).absolute()
    identity = hashlib.sha256(str(image_path).encode()).hexdigest()[:12]
    return Path(mount_root) / f"{image_path.parent.name}-{identity}"


def mount_lease_path(image_path: Path) -> Path:
    """Return the cross-host lease directory for an image mount."""
    return Path(f"{Path(image_path)}{ARTIFACT_STORE_MOUNT_LEASE_SUFFIX}")


def paths_for_run(run_dir: Path) -> ArtifactStorePaths:
    """Return the durable image and its deterministic node-local mount path."""
    run_dir = Path(run_dir)
    image = run_dir / ARTIFACT_STORE_IMAGE
    return ArtifactStorePaths(image=image, mount=mount_path_for_image(image))


def ensure_image(
    image_path: Path,
    *,
    size: str = DEFAULT_IMAGE_SIZE,
    inode_count: int = DEFAULT_INODE_COUNT,
) -> None:
    """Create a sparse ext4 image atomically, or retain the existing image."""
    image_path = Path(image_path)
    if image_path.exists():
        return
    if not size or inode_count <= 0:
        raise ValueError("artifact-store size and inode_count must be positive")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{image_path}{ARTIFACT_STORE_LOCK_SUFFIX}")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if image_path.exists():
            return
        temporary = image_path.with_name(f".{image_path.name}.creating-{os.getpid()}")
        try:
            subprocess.run(["truncate", "-s", size, str(temporary)], check=True)
            subprocess.run(
                ["mkfs.ext4", "-F", "-N", str(inode_count), str(temporary)],
                check=True,
            )
            os.replace(temporary, image_path)
        finally:
            temporary.unlink(missing_ok=True)


def write_authority_record(
    image_path: Path,
    *,
    job_id: str,
    node: str,
    node_ip: str,
    mount_path: Path,
) -> Path:
    """Publish the live writer location for node-routed readers."""
    image_path = Path(image_path)
    record_path = image_path.parent / ARTIFACT_STORE_AUTHORITY
    temporary = record_path.with_name(f".{record_path.name}.tmp-{os.getpid()}")
    payload = {
        "job_id": job_id,
        "node": node,
        "node_ip": node_ip,
        "image": str(image_path),
        "mount": str(mount_path),
        "trials": str(Path(mount_path) / TRACE_JOBS_SUBDIR),
    }
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, record_path)
    return record_path


def resolve_trials_root(run_dir: Path) -> tuple[Literal["bare", "image"], Path]:
    """Resolve a legacy trial tree or the image that contains the new tree."""
    run_dir = Path(run_dir)
    image_path = run_dir / ARTIFACT_STORE_IMAGE
    if image_path.is_file():
        return "image", image_path
    return "bare", run_dir / TRACE_JOBS_SUBDIR


@contextmanager
def mounted(
    image_path: Path,
    mode: Literal["ro", "rw"] = "ro",
    *,
    mount_path: Path | None = None,
) -> Iterator[Path]:
    """Mount an inactive ext4 image with fuse2fs and unmount it on exit.

    A shared sidecar lock protects read-only mounts and an exclusive lock protects
    read-write mounts. Both fail while a batch link holds the writer lock.
    """
    if mode not in {"ro", "rw"}:
        raise ValueError(f"unsupported artifact-store mount mode: {mode}")

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    lease_path = mount_lease_path(image_path)
    try:
        lease_path.mkdir()
    except FileExistsError as error:
        raise ArtifactStoreBusyError(
            f"artifact store has an active mount lease: {image_path}"
        ) from error
    lease_owner = lease_path / "owner"
    try:
        lease_owner.write_text(
            f"kind=reader\nhost={socket.gethostname()}\npid={os.getpid()}\n"
        )
        lock_path = Path(f"{image_path}{ARTIFACT_STORE_LOCK_SUFFIX}")
        requested_lock = fcntl.LOCK_SH if mode == "ro" else fcntl.LOCK_EX
        with lock_path.open("a+b") as lock_file:
            try:
                fcntl.flock(lock_file, requested_lock | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ArtifactStoreBusyError(
                    f"artifact store is mounted by an active writer: {image_path}"
                ) from error

            temporary_mount = None
            if mount_path is None:
                temporary_mount = tempfile.TemporaryDirectory(
                    prefix="otagent-artifacts-"
                )
                resolved_mount = Path(temporary_mount.name)
            else:
                resolved_mount = Path(mount_path)
                resolved_mount.mkdir(parents=True, exist_ok=True)

            mounted_ok = False
            try:
                if mode == "rw":
                    check = subprocess.run(
                        ["e2fsck", "-p", str(image_path)], check=False
                    )
                    if check.returncode not in {0, 1}:
                        raise subprocess.CalledProcessError(
                            check.returncode, check.args
                        )
                subprocess.run(
                    ["fuse2fs", "-o", mode, str(image_path), str(resolved_mount)],
                    check=True,
                )
                mount_check = subprocess.run(
                    ["mountpoint", "-q", str(resolved_mount)], check=False
                )
                if mount_check.returncode != 0:
                    subprocess.run(
                        ["fusermount3", "-u", str(resolved_mount)], check=False
                    )
                    raise subprocess.CalledProcessError(
                        mount_check.returncode, mount_check.args
                    )
                mounted_ok = True
                yield resolved_mount
            finally:
                if mounted_ok:
                    subprocess.run(
                        ["fusermount3", "-u", str(resolved_mount)], check=True
                    )
                if temporary_mount is not None:
                    temporary_mount.cleanup()
    finally:
        lease_owner.unlink(missing_ok=True)
        lease_path.rmdir()


@contextmanager
def open_trials_root(run_dir: Path) -> Iterator[Path]:
    """Yield the readable trial tree for legacy and image-backed runs."""
    kind, path = resolve_trials_root(run_dir)
    if kind == "bare":
        yield path
        return
    with mounted(path, mode="ro") as root:
        yield root / TRACE_JOBS_SUBDIR


@contextmanager
def open_trials_path(path: Path) -> Iterator[Path]:
    """Yield a requested trial path, mounting its parent run image if needed."""
    path = Path(path)
    if path.is_dir():
        yield path
        return
    if path.name != TRACE_JOBS_SUBDIR:
        yield path
        return
    kind, source = resolve_trials_root(path.parent)
    if kind == "bare":
        yield source
        return
    with mounted(source, mode="ro") as root:
        yield root / TRACE_JOBS_SUBDIR


def main() -> None:
    """Mount an inactive image read-only for an interactive inspection."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--mount-path", type=Path)
    args = parser.parse_args()

    with mounted(args.image, mode="ro", mount_path=args.mount_path) as root:
        print(root, flush=True)
        input("Press Enter to unmount: ")


if __name__ == "__main__":
    main()
