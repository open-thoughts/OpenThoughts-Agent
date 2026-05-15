#!/usr/bin/env python3
"""Daytona sandbox health check and latency benchmarking.

Bypasses Harbor to directly measure Daytona infrastructure performance:
  - Snapshot lookup / sandbox creation latency
  - Command execution latency
  - Multi-turn agent simulation (oracle commands)
  - Sandbox teardown latency
  - Concurrent sandbox stress testing

Designed to run on Jupiter compute nodes mid-job. On Jupiter, all outbound
traffic goes through a proxychains SOCKS5 proxy. This script must be invoked
through proxychains so that Daytona API calls can reach the internet:

    # SSH into a compute node from the login node
    ssh -i $SSH_KEY <compute-node>

    # Source the job's environment
    source ~/secrets.env
    export PROXYCHAINS_CONF_FILE=/tmp/proxychains_<JOBID>.conf
    PCBIN=/e/scratch/jureap59/feuer1/proxychains-ng-aarch64/bin/proxychains4

    # Run through proxychains
    $PCBIN -f $PROXYCHAINS_CONF_FILE python scripts/daytona/health_check.py \\
        --snapshot "harbor__abc123__snapshot"

    # Stress test with 8 concurrent sandboxes
    $PCBIN -f $PROXYCHAINS_CONF_FILE python scripts/daytona/health_check.py \\
        --snapshot "harbor__abc123__snapshot" --concurrency 8

Environment variables:
    DAYTONA_API_KEY   - Daytona API key (required)
    DAYTONA_API_URL   - Daytona API URL (default: https://app.daytona.io/api)
    DAYTONA_TARGET    - Daytona target region (optional)
"""

import argparse
import hashlib
import json
import os
import socket
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Oracle commands: simple, fast commands that exercise the sandbox without
# needing a real LLM.  Each is a (description, command) pair.
# ---------------------------------------------------------------------------
ORACLE_TURNS = [
    ("list root", "ls -lha /"),
    ("create directory", "mkdir -p /tmp/healthcheck/subdir"),
    ("write file", "echo 'hello from health check' > /tmp/healthcheck/test.txt"),
    ("read file", "cat /tmp/healthcheck/test.txt"),
    ("system info", "uname -a && cat /etc/os-release 2>/dev/null || true"),
]


def _hash_dockerfile(dockerfile_path: Path) -> str:
    """SHA-256 hash of Dockerfile content (matches Harbor's auto_snapshot naming)."""
    content = dockerfile_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:12]


def _find_dockerfile(task_path: Path) -> Path | None:
    """Find Dockerfile in a task directory."""
    for c in [task_path / "environment" / "Dockerfile", task_path / "Dockerfile"]:
        if c.exists():
            return c
    return None


def _find_snapshot_name(dockerfile_path: Path) -> str:
    """Generate the Harbor auto_snapshot name from a Dockerfile."""
    return f"harbor__{_hash_dockerfile(dockerfile_path)}__snapshot"


def _read_job_config(job_dir: Path) -> dict[str, Any]:
    """Read the RL job config to extract environment settings."""
    config_files = list(job_dir.glob("configs/*rl_config.json"))
    if not config_files:
        return {}
    with open(config_files[0]) as f:
        return json.load(f)


def _check_proxy_environment() -> None:
    """Warn if proxychains doesn't appear to be active on JSC compute nodes."""
    ld_preload = os.environ.get("LD_PRELOAD", "")
    proxychains_conf = os.environ.get("PROXYCHAINS_CONF_FILE", "")

    if "proxychains" in ld_preload or proxychains_conf:
        print(f"[proxy] Proxychains detected: {proxychains_conf or '(via LD_PRELOAD)'}")
    else:
        hostname = socket.gethostname()
        if hostname.startswith(("jpbo-", "jwb", "jrc")):
            print(
                "[proxy] WARNING: Running on a JSC compute node without proxychains!\n"
                "[proxy] Daytona API calls will likely fail. Run through proxychains:\n"
                "[proxy]   proxychains4 -f $PROXYCHAINS_CONF_FILE python scripts/daytona/health_check.py ...\n"
            )


def create_sandbox(
    daytona: Any,
    snapshot_name: str | None = None,
    dockerfile_path: Path | None = None,
    cpus: int = 1,
    memory_mb: int = 2048,
    storage_mb: int = 2048,
    timeout: float = 300,
) -> tuple[Any, float]:
    """Create a sandbox and return (sandbox, creation_time_sec)."""
    from daytona import (
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        Image,
        Resources,
    )

    resources = Resources(
        cpu=cpus,
        memory=max(1, memory_mb // 1024),
        disk=max(1, storage_mb // 1024),
        gpu=0,
    )

    t0 = time.monotonic()

    if snapshot_name:
        try:
            params = CreateSandboxFromSnapshotParams(
                snapshot=snapshot_name,
                resources=resources,
                auto_stop_interval=0,
                auto_archive_interval=0,
                auto_delete_interval=30,
                ephemeral=True,
            )
            sandbox = daytona.create(params, timeout=timeout)
            return sandbox, time.monotonic() - t0
        except Exception as e:
            print(f"  Snapshot '{snapshot_name}' failed: {e}")
            if dockerfile_path is None:
                raise
            print(f"  Falling back to Dockerfile build...")

    if dockerfile_path:
        image = Image.from_dockerfile(str(dockerfile_path))
        params = CreateSandboxFromImageParams(
            image=image,
            resources=resources,
            auto_stop_interval=0,
            auto_archive_interval=0,
            auto_delete_interval=30,
            ephemeral=True,
        )
        sandbox = daytona.create(params, timeout=timeout)
        return sandbox, time.monotonic() - t0

    raise ValueError("Either snapshot_name or dockerfile_path must be provided")


def exec_command(sandbox: Any, command: str, timeout: float = 30) -> tuple[int, str, float]:
    """Execute a command and return (exit_code, output, elapsed_sec)."""
    t0 = time.monotonic()
    result = sandbox.process.exec(command, timeout=timeout)
    return result.exit_code, result.result, time.monotonic() - t0


def delete_sandbox(sandbox: Any, timeout: float = 30) -> float:
    """Delete a sandbox and return elapsed time."""
    t0 = time.monotonic()
    sandbox.delete(timeout=timeout)
    return time.monotonic() - t0


def run_single_health_check(
    daytona: Any,
    snapshot_name: str | None,
    dockerfile_path: Path | None,
    cpus: int,
    memory_mb: int,
    storage_mb: int,
    build_timeout: float,
    worker_id: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a full health check cycle on one sandbox."""
    prefix = f"[worker-{worker_id}]" if worker_id > 0 else ""
    results: dict[str, Any] = {"worker_id": worker_id, "success": False}

    # 1. Create sandbox
    if verbose:
        src = f"snapshot={snapshot_name}" if snapshot_name else f"Dockerfile={dockerfile_path}"
        print(f"{prefix} Creating sandbox ({src})...")
    try:
        sandbox, create_time = create_sandbox(
            daytona,
            snapshot_name=snapshot_name,
            dockerfile_path=dockerfile_path,
            cpus=cpus,
            memory_mb=memory_mb,
            storage_mb=storage_mb,
            timeout=build_timeout,
        )
        results["create_sec"] = create_time
        if verbose:
            print(f"{prefix} Created in {create_time:.1f}s (id={sandbox.id})")
    except Exception as e:
        results["create_error"] = str(e)
        if verbose:
            print(f"{prefix} FAILED to create sandbox: {e}")
        return results

    try:
        # 2. Execute oracle turns
        turn_times = []
        for i, (desc, cmd) in enumerate(ORACLE_TURNS):
            try:
                exit_code, output, elapsed = exec_command(sandbox, cmd)
                turn_times.append(elapsed)
                if verbose:
                    status = "OK" if exit_code == 0 else f"exit={exit_code}"
                    print(f"{prefix} Turn {i+1}/{len(ORACLE_TURNS)} ({desc}): {elapsed:.2f}s [{status}]")
            except Exception as e:
                turn_times.append(None)
                if verbose:
                    print(f"{prefix} Turn {i+1}/{len(ORACLE_TURNS)} ({desc}): FAILED ({e})")

        valid_times = [t for t in turn_times if t is not None]
        results["turn_times_sec"] = turn_times
        if valid_times:
            results["turn_min_sec"] = min(valid_times)
            results["turn_avg_sec"] = statistics.mean(valid_times)
            results["turn_max_sec"] = max(valid_times)
            results["turns_succeeded"] = len(valid_times)
            results["turns_total"] = len(ORACLE_TURNS)

        # 3. Teardown
        if verbose:
            print(f"{prefix} Deleting sandbox...")
        dt = delete_sandbox(sandbox)
        results["delete_sec"] = dt
        if verbose:
            print(f"{prefix} Deleted in {dt:.1f}s")

        results["success"] = True
        results["total_sec"] = results["create_sec"] + sum(valid_times) + results["delete_sec"]

    except Exception as e:
        results["teardown_error"] = str(e)
        if verbose:
            print(f"{prefix} Error during execution/teardown: {e}")
        try:
            sandbox.delete(timeout=10)
        except Exception:
            pass

    return results


def run_concurrent_stress_test(
    daytona: Any,
    concurrency: int,
    snapshot_name: str | None,
    dockerfile_path: Path | None,
    cpus: int,
    memory_mb: int,
    storage_mb: int,
    build_timeout: float,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Run N health checks concurrently using threads."""
    print(f"\n{'='*60}")
    print(f"STRESS TEST: {concurrency} concurrent sandboxes")
    print(f"{'='*60}\n")

    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                run_single_health_check,
                daytona=daytona,
                snapshot_name=snapshot_name,
                dockerfile_path=dockerfile_path,
                cpus=cpus,
                memory_mb=memory_mb,
                storage_mb=storage_mb,
                build_timeout=build_timeout,
                worker_id=i + 1,
                verbose=verbose,
            ): i
            for i in range(concurrency)
        }
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"success": False, "error": str(e)})

    wall_time = time.monotonic() - t0

    # Summary
    succeeded = sum(1 for r in results if r.get("success"))
    create_times = [r["create_sec"] for r in results if "create_sec" in r]
    delete_times = [r["delete_sec"] for r in results if "delete_sec" in r]
    turn_avgs = [r["turn_avg_sec"] for r in results if "turn_avg_sec" in r]

    print(f"\n{'='*60}")
    print(f"STRESS TEST RESULTS ({concurrency} sandboxes)")
    print(f"{'='*60}")
    print(f"  Succeeded:   {succeeded}/{concurrency}")
    print(f"  Wall time:   {wall_time:.1f}s")
    if create_times:
        print(f"  Create time: min={min(create_times):.1f}s  avg={statistics.mean(create_times):.1f}s  max={max(create_times):.1f}s")
    if turn_avgs:
        print(f"  Turn latency: min={min(turn_avgs):.2f}s  avg={statistics.mean(turn_avgs):.2f}s  max={max(turn_avgs):.2f}s")
    if delete_times:
        print(f"  Delete time: min={min(delete_times):.1f}s  avg={statistics.mean(delete_times):.1f}s  max={max(delete_times):.1f}s")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-path", type=str, help="Path to a task directory with environment/Dockerfile")
    parser.add_argument("--job-dir", type=str, help="Path to RL experiment directory (auto-detects task)")
    parser.add_argument("--snapshot", type=str, help="Explicit Daytona snapshot name to use")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent sandboxes (stress test)")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per sandbox (default: 1)")
    parser.add_argument("--memory-mb", type=int, default=2048, help="Memory per sandbox in MB (default: 2048)")
    parser.add_argument("--storage-mb", type=int, default=2048, help="Storage per sandbox in MB (default: 2048)")
    parser.add_argument("--build-timeout", type=float, default=300, help="Sandbox build timeout in seconds (default: 300)")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Verbose output (default: true)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    _check_proxy_environment()

    from daytona import Daytona, DaytonaConfig

    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        print("Error: DAYTONA_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    config = DaytonaConfig(
        api_key=api_key,
        api_url=os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api"),
        target=os.environ.get("DAYTONA_TARGET"),
    )
    daytona = Daytona(config)

    # Resolve snapshot/dockerfile
    snapshot_name = args.snapshot
    dockerfile_path = None

    if args.task_path:
        task_path = Path(args.task_path)
        df = _find_dockerfile(task_path)
        if df:
            dockerfile_path = df
            if not snapshot_name:
                snapshot_name = _find_snapshot_name(df)
                print(f"Auto-detected snapshot name: {snapshot_name}")
        else:
            print(f"Warning: No Dockerfile found in {task_path}")

    if args.job_dir:
        job_dir = Path(args.job_dir)
        config_data = _read_job_config(job_dir)
        train_data = config_data.get("train_data", [])
        if train_data and isinstance(train_data, list):
            for td in train_data:
                td_path = Path(td)
                if td_path.is_dir():
                    for task_dir in sorted(td_path.iterdir()):
                        if task_dir.is_dir():
                            df = _find_dockerfile(task_dir)
                            if df:
                                dockerfile_path = df
                                if not snapshot_name:
                                    snapshot_name = _find_snapshot_name(df)
                                print(f"Auto-detected task: {task_dir.name}")
                                print(f"Auto-detected snapshot: {snapshot_name}")
                                break
                    if dockerfile_path:
                        break

    if not snapshot_name and not dockerfile_path:
        print("Error: Must provide --snapshot, --task-path, or --job-dir", file=sys.stderr)
        sys.exit(1)

    # Resource settings
    cpus = args.cpus
    memory_mb = args.memory_mb
    storage_mb = args.storage_mb
    build_timeout = args.build_timeout

    if args.concurrency > 1:
        run_concurrent_stress_test(
            daytona=daytona,
            concurrency=args.concurrency,
            snapshot_name=snapshot_name,
            dockerfile_path=dockerfile_path,
            cpus=cpus,
            memory_mb=memory_mb,
            storage_mb=storage_mb,
            build_timeout=build_timeout,
            verbose=args.verbose,
        )
    else:
        print(f"\n{'='*60}")
        print(f"DAYTONA HEALTH CHECK")
        print(f"{'='*60}\n")

        result = run_single_health_check(
            daytona=daytona,
            snapshot_name=snapshot_name,
            dockerfile_path=dockerfile_path,
            cpus=cpus,
            memory_mb=memory_mb,
            storage_mb=storage_mb,
            build_timeout=build_timeout,
            verbose=True,
        )

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        if result.get("success"):
            print(f"  Status:      OK")
            print(f"  Create:      {result['create_sec']:.1f}s")
            print(f"  Turns:       {result.get('turns_succeeded', 0)}/{result.get('turns_total', 0)}")
            print(f"  Turn latency: min={result.get('turn_min_sec', 0):.2f}s  avg={result.get('turn_avg_sec', 0):.2f}s  max={result.get('turn_max_sec', 0):.2f}s")
            print(f"  Delete:      {result['delete_sec']:.1f}s")
            print(f"  Total:       {result['total_sec']:.1f}s")
        else:
            print(f"  Status:      FAILED")
            for k, v in result.items():
                if "error" in k:
                    print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
