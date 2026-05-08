#!/usr/bin/env python3
"""Pre-create Daytona snapshots for eval datasets.

Computes unique environment hashes per dataset, then creates snapshots
on the Daytona org specified by the secrets file. Handles RL region routing
(builds via "us" but registers for "RL").

Usage:
    python eval/MBZ/precreate_snapshots.py \
        --dataset DCAgent/dev_set_v2 \
        --dataset DCAgent2/terminal_bench_2 \
        --secrets-file ~/secrets.env

    python eval/MBZ/precreate_snapshots.py \
        --dataset DCAgent/dev_set_v2 \
        --dataset DCAgent2/terminal_bench_2 \
        --secrets-file ~/secrets_rl_org.env
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure harbor is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
HARBOR_SRC = REPO_ROOT / "harbor" / "src"
if str(HARBOR_SRC) not in sys.path:
    sys.path.insert(0, str(HARBOR_SRC))


def load_secrets(secrets_file: str) -> None:
    """Parse 'export KEY=value' lines from a secrets file into os.environ."""
    with open(os.path.expanduser(secrets_file)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def resolve_dataset_path(repo_id: str) -> Path:
    """Resolve a HF repo ID to the local cached snapshot path."""
    hf_cache = os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    dataset_dir = Path(hf_cache) / f"datasets--{repo_id.replace('/', '--')}"
    snapshots_dir = dataset_dir / "snapshots"

    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"Dataset {repo_id} not found in HF cache at {snapshots_dir}. "
            "Download it first with snapshot_download.py."
        )

    snapshots = sorted(
        [d for d in snapshots_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")

    return snapshots[-1]


def get_task_dirs(dataset_path: Path) -> list[Path]:
    """Find task directories (those containing instruction.md)."""
    return sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and (d / "instruction.md").exists()
    ])


def compute_unique_snapshots(
    datasets: list[tuple[str, Path]],
    target: str | None,
) -> dict[str, str]:
    """Compute unique snapshot names and a representative Dockerfile for each.

    Returns:
        Dict of snapshot_name -> representative Dockerfile path (as string).
    """
    from harbor.utils.container_cache import get_task_environment_hash

    snapshots: dict[str, str] = {}  # snapshot_name -> dockerfile_path

    for repo_id, dataset_path in datasets:
        task_dirs = get_task_dirs(dataset_path)
        print(f"\n=== {repo_id} ({len(task_dirs)} tasks) ===")

        hash_counts: dict[str, int] = {}
        hash_example: dict[str, Path] = {}

        for task_dir in task_dirs:
            env_hash = get_task_environment_hash(task_dir)
            if env_hash is None:
                continue
            hash_counts[env_hash] = hash_counts.get(env_hash, 0) + 1
            if env_hash not in hash_example:
                hash_example[env_hash] = task_dir

        print(f"Unique environment hashes: {len(hash_counts)}")

        for env_hash, count in sorted(hash_counts.items(), key=lambda x: -x[1]):
            if target:
                snap_name = f"harbor__{env_hash}__{target}__snapshot"
            else:
                snap_name = f"harbor__{env_hash}__snapshot"

            example_task = hash_example[env_hash]
            dockerfile = example_task / "environment" / "Dockerfile"

            print(f"  {snap_name}  ({count} tasks)  e.g. {example_task.name}")
            snapshots[snap_name] = str(dockerfile)

    return snapshots


async def create_snapshot(
    client,
    name: str,
    dockerfile_path: str,
    target: str | None,
    label: str,
) -> bool:
    """Create a single snapshot, handling already-exists gracefully."""
    from daytona import AsyncDaytona, DaytonaConfig, CreateSnapshotParams, Image, Resources
    from daytona._async.snapshot import SnapshotState

    # Check existing state
    try:
        snap = await client.snapshot.get(name)
        if snap.state == SnapshotState.ACTIVE:
            print(f"  [{label}] {name}: already ACTIVE, skipping")
            return True
        elif snap.state == SnapshotState.ERROR:
            print(f"  [{label}] {name}: ERROR state, deleting and recreating...")
            try:
                await client.snapshot.delete(snap)
            except Exception as e:
                print(f"  [{label}] {name}: failed to delete ERROR snapshot: {e}")
    except Exception:
        pass  # Doesn't exist yet

    # RL region has no build runners — build via "us", register for RL
    build_client = client
    if target and target.upper() == "RL":
        api_key = os.environ.get("DAYTONA_API_KEY", "")
        build_client = AsyncDaytona(DaytonaConfig(api_key=api_key, target="us"))

    print(f"  [{label}] Creating {name} from {dockerfile_path}...")
    try:
        await build_client.snapshot.create(
            CreateSnapshotParams(
                name=name,
                image=Image.from_dockerfile(dockerfile_path),
                resources=Resources(cpu=1, memory=1, disk=3),
                region_id=target if target else None,
            )
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "conflict" in error_msg:
            print(f"  [{label}] {name}: already exists (global), OK")
            return True
        print(f"  [{label}] {name}: create FAILED: {e}")
        return False
    finally:
        if build_client is not client:
            await build_client.close()

    # Poll for ACTIVE state (up to 10 minutes, 5s intervals)
    for i in range(120):
        await asyncio.sleep(5)
        try:
            snap = await client.snapshot.get(name)
            if snap.state == SnapshotState.ACTIVE:
                print(f"  [{label}] {name}: ACTIVE (took ~{(i+1)*5}s)")
                return True
            elif snap.state == SnapshotState.ERROR:
                reason = getattr(snap, "error_reason", "unknown")
                print(f"  [{label}] {name}: entered ERROR state: {reason}")
                return False
        except Exception:
            pass

    print(f"  [{label}] {name}: TIMEOUT waiting for ACTIVE (600s)")
    return False


async def main():
    parser = argparse.ArgumentParser(
        description="Pre-create Daytona snapshots for eval datasets"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="HF repo ID (e.g. DCAgent/dev_set_v2). Can be repeated.",
    )
    parser.add_argument(
        "--secrets-file",
        required=True,
        help="Path to secrets env file (must contain DAYTONA_API_KEY and DAYTONA_TARGET)",
    )
    args = parser.parse_args()

    # Load secrets into environment
    load_secrets(args.secrets_file)

    api_key = os.environ.get("DAYTONA_API_KEY")
    target = os.environ.get("DAYTONA_TARGET")

    if not api_key:
        print("ERROR: DAYTONA_API_KEY not found in secrets file", file=sys.stderr)
        sys.exit(1)

    label = target or "default"
    print(f"Daytona target: {label}")
    print(f"API key: {api_key[:12]}...")

    # Resolve dataset paths
    datasets: list[tuple[str, Path]] = []
    for repo_id in args.dataset:
        path = resolve_dataset_path(repo_id)
        print(f"Dataset {repo_id} -> {path}")
        datasets.append((repo_id, path))

    # Compute unique snapshots
    snapshots = compute_unique_snapshots(datasets, target)

    if not snapshots:
        print("\nNo snapshots to create (no tasks with Dockerfiles found).")
        return

    print(f"\n--- Creating {len(snapshots)} unique snapshots on '{label}' ---\n")

    # Create snapshots
    from daytona import AsyncDaytona, DaytonaConfig

    client = AsyncDaytona(DaytonaConfig(api_key=api_key, target=target or "us"))
    try:
        results = {}
        for snap_name, dockerfile in snapshots.items():
            ok = await create_snapshot(client, snap_name, dockerfile, target, label)
            results[snap_name] = ok
    finally:
        await client.close()

    # Summary
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    print(f"\n=== Done: {succeeded} succeeded, {failed} failed ===")

    if failed:
        print("\nFailed snapshots:")
        for name, ok in results.items():
            if not ok:
                print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
