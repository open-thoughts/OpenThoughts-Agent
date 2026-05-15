#!/usr/bin/env python3
"""Pre-create Daytona snapshots for all unique dockerfiles across GLM-4.7 datasets.

Reads dataset_analysis_report.jsonl, downloads each non-skipped dataset,
discovers unique environment hashes, and registers them as Daytona snapshots
on both orgs.

Modes:
    register  - Create snapshots for needed hashes (default)
    list      - List all existing snapshots on both orgs
    cleanup   - Delete snapshots NOT needed by GLM-4.7 datasets
    status    - Check status of needed snapshots

Run from login node (needs internet for Daytona API + HF downloads).

Usage:
    source ~/data_gen_secrets.env
    export PYTHONPATH="/e/scratch/jureap59/feuer1/OpenThoughts-Agent:/e/scratch/jureap59/feuer1/OpenThoughts-Agent/scripts/harbor:/e/scratch/jureap59/etash/harbor/src:/e/scratch/jureap59/feuer1/OpenThoughts-Agent/data:$PYTHONPATH"
    /lib/ld-linux-aarch64.so.1 /e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python3.12 register_snapshots.py [list|cleanup|status|register]
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Redirect temp files and HF cache to playground (scratch has quota limits)
PLAYGROUND_TMP = "/e/data1/datasets/playground/mmlaion/shared/guha1/tmp_glm47"
os.environ["TMPDIR"] = PLAYGROUND_TMP
os.environ["HF_HOME"] = os.path.join(PLAYGROUND_TMP, "hf_cache")

# --- Harbor imports (use harbor's own code as much as possible) ---
from harbor.utils.container_cache import environment_dir_hash_truncated
from harbor.models.task.paths import TaskPaths
from harbor.environments.daytona_utils import (
    is_transient_daytona_error,
    create_snapshot_retry_callback,
    create_snapshot_wait_callback,
    get_snapshot_retry_callback,
    get_snapshot_wait_callback,
)

# --- Daytona SDK ---
from daytona import AsyncDaytona, DaytonaConfig, CreateSnapshotParams, Image, Resources
from daytona._async.snapshot import SnapshotState

# --- HF dataset helpers ---
from tasks_parquet_converter import from_hf_dataset, find_tasks

# --- tenacity for retry ---
from tenacity import retry

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_PATH = SCRIPT_DIR / "dataset_analysis_report.jsonl"
REGISTRY_PATH = SCRIPT_DIR / "snapshot_registry.jsonl"
DOCKERFILE_CACHE_DIR = os.path.join(PLAYGROUND_TMP, "dockerfile_cache")

DAYTONA_KEYS = {
    "org1": "dtn_17868a1955b56a52cb367af6dd3c6e93ee531b2073df801784273435c0e0fc6c",
    "org2": "dtn_ecfb759d2abcbf0f044413182677b3cc7624af495cb48409e8873b5283bda313",
}

# Match harbor's default snapshot resources
DEFAULT_RESOURCES = Resources(cpu=1, memory=1, disk=3)


def get_snapshot_name(env_hash: str) -> str:
    """Compute snapshot name matching harbor's DaytonaEnvironment._get_auto_snapshot_name().

    When DAYTONA_TARGET is set, includes the region in the name.
    The sbatch script does NOT set DAYTONA_TARGET, so we use the simple format.
    """
    target = os.environ.get("DAYTONA_TARGET")
    if target:
        return f"harbor__{env_hash}__{target}__snapshot"
    return f"harbor__{env_hash}__snapshot"


# ---- Snapshot operations (modeled after harbor's daytona.py) ----

@retry(
    retry=get_snapshot_retry_callback,
    wait=get_snapshot_wait_callback,
    reraise=True,
)
async def get_snapshot_with_retry(client: AsyncDaytona, name: str):
    """Get snapshot by name with retries for transient errors.

    Mirrors DaytonaEnvironment._get_snapshot_with_retry().
    """
    return await client.snapshot.get(name)


async def wait_for_snapshot(client: AsyncDaytona, name: str, timeout: int = 600) -> str:
    """Wait for snapshot to become ACTIVE.

    Mirrors DaytonaEnvironment._wait_for_snapshot().
    """
    for _ in range(timeout // 5):
        await asyncio.sleep(5)
        try:
            snapshot = await get_snapshot_with_retry(client, name)
            if snapshot.state == SnapshotState.ACTIVE:
                return "ACTIVE"
            if snapshot.state == SnapshotState.ERROR:
                return "ERROR"
        except Exception as e:
            print(f"    Warning checking snapshot: {e}")
    return "TIMEOUT"


@retry(
    retry=create_snapshot_retry_callback,
    wait=create_snapshot_wait_callback,
    reraise=True,
)
async def create_snapshot_with_retry(
    client: AsyncDaytona, snapshot_name: str, env_dir_path: str, org_name: str
) -> str:
    """Create a snapshot and wait for ACTIVE.

    Mirrors DaytonaEnvironment._create_snapshot_with_retry():
    - Cleans up ERROR-state snapshots before retry
    - Handles "already exists" / "conflict" as success
    - Uses Image.from_dockerfile() with the environment dir's Dockerfile
    """
    # Clean up any ERROR-state snapshot left by a previous attempt
    try:
        existing = await client.snapshot.get(snapshot_name)
        if existing.state == SnapshotState.ERROR:
            print(f"  [{org_name}] Removing stale ERROR snapshot {snapshot_name}")
            await client.snapshot.delete(existing)
    except Exception:
        pass  # Snapshot doesn't exist — normal case

    print(f"  [{org_name}] Creating {snapshot_name}...")
    dockerfile_path = str(Path(env_dir_path) / "Dockerfile")
    target = os.environ.get("DAYTONA_TARGET")
    try:
        await client.snapshot.create(
            CreateSnapshotParams(
                name=snapshot_name,
                image=Image.from_dockerfile(dockerfile_path),
                resources=DEFAULT_RESOURCES,
                region_id=target if target else None,
            )
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "conflict" in error_msg:
            print(f"  [{org_name}] {snapshot_name}: already exists (global), OK")
            return "ACTIVE"
        raise

    return await wait_for_snapshot(client, snapshot_name)


async def ensure_snapshot(
    client: AsyncDaytona, snapshot_name: str, env_dir_path: str, org_name: str
) -> str:
    """Ensure snapshot exists, creating if needed.

    Mirrors DaytonaEnvironment._ensure_auto_snapshot() fast/slow path logic.
    """
    # === FAST PATH: check if already exists ===
    try:
        snapshot = await get_snapshot_with_retry(client, snapshot_name)
        if snapshot.state == SnapshotState.ACTIVE:
            print(f"  [{org_name}] {snapshot_name}: already ACTIVE")
            return "ACTIVE"
        elif snapshot.state == SnapshotState.PENDING:
            print(f"  [{org_name}] {snapshot_name}: PENDING, waiting...")
            return await wait_for_snapshot(client, snapshot_name)
        elif snapshot.state == SnapshotState.ERROR:
            print(f"  [{org_name}] {snapshot_name}: ERROR state, will recreate")
            try:
                await client.snapshot.delete(snapshot)
            except Exception as e:
                print(f"    Warning: failed to delete ERROR snapshot: {e}")
    except Exception:
        # snapshot.get() failed — may be global or doesn't exist yet
        pass

    # === SLOW PATH: create ===
    try:
        status = await create_snapshot_with_retry(client, snapshot_name, env_dir_path, org_name)
        print(f"  [{org_name}] {snapshot_name}: {status}")
        return status
    except Exception as e:
        print(f"  [{org_name}] {snapshot_name}: FAILED: {e}")
        return f"ERROR: {e}"


# ---- Discovery ----

def discover_unique_envs(records: list[dict]) -> dict[str, str]:
    """Download datasets one at a time, save env dirs to cache, clean up.

    Returns dict mapping environment hash to cached environment dir path.
    Uses environment_dir_hash_truncated() from harbor.utils.container_cache —
    the same function harbor uses internally for snapshot naming.
    """
    import shutil

    os.makedirs(DOCKERFILE_CACHE_DIR, exist_ok=True)
    hash_to_env_dir: dict[str, str] = {}

    # Check cached env dirs from previous runs
    cache_path = Path(DOCKERFILE_CACHE_DIR)
    if cache_path.exists():
        for cached in cache_path.iterdir():
            if cached.is_dir() and (cached / "Dockerfile").exists():
                hash_to_env_dir[cached.name] = str(cached)
        if hash_to_env_dir:
            print(f"Found {len(hash_to_env_dir)} cached env dirs from previous run")

    actionable = [r for r in records if not r["skip_reason"] and not r["error"]]

    # Collect all hashes we need
    all_needed = set()
    for r in actionable:
        all_needed.update(r["hash_counts"].keys())

    # Only need to discover hashes not already cached
    missing = all_needed - set(hash_to_env_dir.keys())
    if not missing:
        print(f"All {len(all_needed)} env dirs already cached!")
        # Filter to only needed hashes
        return {h: hash_to_env_dir[h] for h in all_needed if h in hash_to_env_dir}

    print(f"\nNeed env dirs for {len(missing)} hashes ({len(hash_to_env_dir)} cached)")

    # Sort: fewest unique envs first
    actionable.sort(key=lambda r: r["unique_envs"])

    for i, rec in enumerate(actionable, 1):
        needed_from_this = set(rec["hash_counts"].keys()) & missing - set(hash_to_env_dir.keys())
        if not needed_from_this:
            continue

        repo_id = rec["repo_id"]
        remaining = len(missing) - (len(hash_to_env_dir) - (len(all_needed) - len(missing)))
        print(f"  [{i}/{len(actionable)}] {repo_id} (need {len(needed_from_this)} hashes) ...")

        task_base = None
        try:
            task_base = from_hf_dataset(repo_id)
            task_dirs = find_tasks(task_base, recursive=True)

            for task_dir in task_dirs:
                env_dir = TaskPaths(task_dir).environment_dir
                if not env_dir.exists():
                    continue
                h = environment_dir_hash_truncated(env_dir, truncate=12)
                if h in needed_from_this and h not in hash_to_env_dir:
                    cached_env = Path(DOCKERFILE_CACHE_DIR) / h
                    if not cached_env.exists():
                        shutil.copytree(str(env_dir), str(cached_env))
                    hash_to_env_dir[h] = str(cached_env)
                    needed_from_this.discard(h)
                    print(f"    Cached {h}")
                    if not needed_from_this:
                        break

        except Exception as e:
            print(f"    ERROR: {e}")
        finally:
            if task_base and Path(task_base).exists():
                shutil.rmtree(task_base, ignore_errors=True)
            for d in [os.path.join(PLAYGROUND_TMP, "hf_datasets"),
                      os.path.join(PLAYGROUND_TMP, "hf_cache")]:
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
                    os.makedirs(d, exist_ok=True)

        if all(h in hash_to_env_dir for h in all_needed):
            print(f"  All {len(all_needed)} hashes collected!")
            break

    still_missing = all_needed - set(hash_to_env_dir.keys())
    if still_missing:
        print(f"  WARNING: missing env dirs for {len(still_missing)} hashes: {still_missing}")

    return {h: hash_to_env_dir[h] for h in all_needed if h in hash_to_env_dir}


def load_needed_hashes() -> set[str]:
    """Load all needed env hashes from the analysis report."""
    hashes = set()
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if not rec["skip_reason"] and not rec["error"]:
                        hashes.update(rec["hash_counts"].keys())
    return hashes


# ---- CLI modes ----

async def list_all_snapshots(client: AsyncDaytona) -> list:
    """List all snapshots with pagination and rate-limit retries."""
    from daytona.common.errors import DaytonaRateLimitError

    all_snaps = []
    page = 1
    for attempt in range(5):
        try:
            while True:
                result = await client.snapshot.list(page=page, limit=100)
                all_snaps.extend(result.items)
                if page >= result.total_pages:
                    return all_snaps
                page += 1
                await asyncio.sleep(2)
        except DaytonaRateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            await asyncio.sleep(wait)
    return all_snaps


async def cmd_list():
    """List all existing snapshots on both orgs."""
    for org_name, api_key in DAYTONA_KEYS.items():
        print(f"\n{'=' * 60}")
        print(f"=== {org_name} ===")
        print(f"{'=' * 60}")

        client = AsyncDaytona(DaytonaConfig(api_key=api_key, target="us"))
        try:
            snapshots = await list_all_snapshots(client)
            print(f"Total snapshots: {len(snapshots)}")
            for snap in sorted(snapshots, key=lambda s: s.name):
                print(f"  {snap.state.value:8s} {snap.name}")
        except Exception as e:
            print(f"  ERROR listing: {e}")
        finally:
            await client.close()
        await asyncio.sleep(5)


async def cmd_status():
    """Check status of all needed snapshots."""
    needed = load_needed_hashes()
    print(f"Need {len(needed)} unique snapshots")

    for org_name, api_key in DAYTONA_KEYS.items():
        print(f"\n--- {org_name} ---")
        client = AsyncDaytona(DaytonaConfig(api_key=api_key, target="us"))
        active = missing = error = pending = 0
        try:
            for h in sorted(needed):
                name = get_snapshot_name(h)
                try:
                    snap = await client.snapshot.get(name)
                    if snap.state == SnapshotState.ACTIVE:
                        active += 1
                    elif snap.state == SnapshotState.PENDING:
                        pending += 1
                        print(f"  PENDING: {name}")
                    elif snap.state == SnapshotState.ERROR:
                        error += 1
                        print(f"  ERROR:   {name}")
                except Exception:
                    missing += 1
        finally:
            await client.close()
        print(f"  ACTIVE={active}, PENDING={pending}, ERROR={error}, MISSING={missing}")


async def cmd_cleanup():
    """Delete snapshots NOT needed by GLM-4.7 datasets to free up quota."""
    from daytona.common.errors import DaytonaRateLimitError

    needed = load_needed_hashes()
    needed_names = {get_snapshot_name(h) for h in needed}
    print(f"Needed snapshots: {len(needed_names)}")

    for org_name, api_key in DAYTONA_KEYS.items():
        print(f"\n{'=' * 60}")
        print(f"=== {org_name} ===")
        print(f"{'=' * 60}")

        client = AsyncDaytona(DaytonaConfig(api_key=api_key, target="us"))
        try:
            snapshots = await list_all_snapshots(client)
            print(f"Total snapshots: {len(snapshots)}")

            to_delete = [s for s in snapshots if s.name not in needed_names]
            to_keep = [s for s in snapshots if s.name in needed_names]

            print(f"Keeping: {len(to_keep)}")
            print(f"Deleting: {len(to_delete)}")

            deleted = 0
            for snap in to_delete:
                for attempt in range(5):
                    try:
                        await client.snapshot.delete(snap)
                        print(f"  Deleted {snap.name}")
                        deleted += 1
                        await asyncio.sleep(1)
                        break
                    except DaytonaRateLimitError:
                        wait = 30 * (attempt + 1)
                        print(f"  Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    except Exception as e:
                        print(f"  FAILED {snap.name}: {e}")
                        break

            print(f"Deleted {deleted}/{len(to_delete)}, kept {len(to_keep)}")
        finally:
            await client.close()
        await asyncio.sleep(5)


async def cmd_register():
    """Register (create) all needed snapshots."""
    if not REPORT_PATH.exists():
        print(f"ERROR: {REPORT_PATH} not found. Run analyze_datasets.py first.")
        sys.exit(1)

    records = []
    with open(REPORT_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Discover all unique environments
    hash_to_env_dir = discover_unique_envs(records)
    print(f"\nTotal unique environment hashes: {len(hash_to_env_dir)}")

    # Load already-registered snapshots (latest status per hash+org)
    registered: dict[str, dict] = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = f"{rec['hash']}_{rec['org']}"
                    registered[key] = rec

    # Register on both orgs
    for org_name, api_key in DAYTONA_KEYS.items():
        print(f"\n{'=' * 60}")
        print(f"=== {org_name} ({api_key[:12]}...) ===")
        print(f"{'=' * 60}")

        client = AsyncDaytona(DaytonaConfig(api_key=api_key, target="us"))
        try:
            for env_hash, env_dir_path in sorted(hash_to_env_dir.items()):
                snap_name = get_snapshot_name(env_hash)

                # Check if already done on this org
                reg_key = f"{env_hash}_{org_name}"
                if reg_key in registered and registered[reg_key].get("status") == "ACTIVE":
                    print(f"  [{org_name}] {snap_name}: already registered ACTIVE, skipping")
                    continue

                status = await ensure_snapshot(client, snap_name, env_dir_path, org_name)

                # Save to registry
                reg_record = {
                    "hash": env_hash,
                    "snapshot_name": snap_name,
                    "org": org_name,
                    "status": status,
                    "env_dir_path": env_dir_path,
                }
                with open(REGISTRY_PATH, "a") as f:
                    f.write(json.dumps(reg_record) + "\n")
                registered[reg_key] = reg_record

        finally:
            await client.close()

    # Print summary
    print(f"\n{'=' * 60}")
    print("SNAPSHOT REGISTRATION SUMMARY")
    print(f"{'=' * 60}")

    # Read latest status per hash+org
    final_status: dict[str, dict] = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = f"{rec['hash']}_{rec['org']}"
                    final_status[key] = rec

    for org_name in DAYTONA_KEYS:
        org_regs = [r for k, r in final_status.items() if r["org"] == org_name]
        active = sum(1 for r in org_regs if r["status"] == "ACTIVE")
        other = len(org_regs) - active
        print(f"  {org_name}: {active} ACTIVE, {other} other")

    print(f"\nRegistry written to {REGISTRY_PATH}")


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "register"

    if mode == "list":
        await cmd_list()
    elif mode == "status":
        await cmd_status()
    elif mode == "cleanup":
        await cmd_cleanup()
    elif mode == "register":
        await cmd_register()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: register_snapshots.py [list|cleanup|status|register]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
