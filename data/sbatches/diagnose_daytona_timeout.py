#!/usr/bin/env python3
"""
Diagnostic script to identify Daytona sandbox creation timeout issues.

Usage:
    # Activate conda env first
    source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
    conda activate /scratch/10000/eguha3/vllm_sandboxes_backup
    source /scratch/10000/eguha3/old-dc-agent/secret.env

    python diagnose_daytona_timeout.py
"""

import asyncio
import time
import sys
import os
from pathlib import Path

# Add harbor to path
sys.path.insert(0, "/scratch/08134/negin/dc-agent-shared/harbor/src")

async def test_daytona_connection():
    """Test if we can connect to Daytona service."""
    print("\n" + "="*60)
    print("TEST 1: Daytona Client Connection")
    print("="*60)

    try:
        from daytona import AsyncDaytona

        start = time.time()
        client = AsyncDaytona()
        elapsed = time.time() - start
        print(f"[OK] Client created in {elapsed:.2f}s")

        # Test if client can make basic calls
        print("Testing client connectivity...")
        start = time.time()
        # This will test if auth/connection works
        try:
            # Try to list snapshots as a basic connectivity test
            snapshots = await client.snapshot.list()
            elapsed = time.time() - start
            print(f"[OK] Listed {len(snapshots)} snapshots in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start
            print(f"[WARN] Could not list snapshots after {elapsed:.2f}s: {e}")

        await client.close()
        return True

    except Exception as e:
        print(f"[FAIL] Could not connect to Daytona: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_sandbox_creation():
    """Test creating a simple sandbox with a minimal Dockerfile."""
    print("\n" + "="*60)
    print("TEST 2: Simple Sandbox Creation")
    print("="*60)

    try:
        from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources

        client = AsyncDaytona()

        # Create a minimal Dockerfile for testing
        test_dockerfile = Path("/tmp/test_daytona_dockerfile/Dockerfile")
        test_dockerfile.parent.mkdir(parents=True, exist_ok=True)
        test_dockerfile.write_text("FROM python:3.11-slim\n")

        image = Image.from_dockerfile(test_dockerfile)
        params = CreateSandboxFromImageParams(
            image=image,
            auto_delete_interval=1,  # Auto-delete quickly
            resources=Resources(cpu=1, memory=1, disk=1),
        )

        print(f"Creating sandbox with minimal Dockerfile...")
        print(f"  - Using: {test_dockerfile}")
        print(f"  - Resources: 1 CPU, 1GB RAM, 1GB disk")

        start = time.time()
        try:
            sandbox = await asyncio.wait_for(
                client.create(params=params, timeout=300),  # 5 min timeout for simple test
                timeout=330  # Slightly longer asyncio timeout
            )
            elapsed = time.time() - start
            print(f"[OK] Sandbox created in {elapsed:.2f}s")
            print(f"  - Sandbox ID: {sandbox.id}")

            # Test running a command
            print("Testing command execution (mkdir)...")
            cmd_start = time.time()
            response = await sandbox.process.start_process("mkdir -p /workspace/test_dir")
            cmd_elapsed = time.time() - cmd_start
            print(f"[OK] mkdir completed in {cmd_elapsed:.2f}s")

            # Cleanup
            print("Deleting sandbox...")
            await sandbox.delete()
            print("[OK] Sandbox deleted")

            await client.close()
            return True

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"[FAIL] Sandbox creation TIMED OUT after {elapsed:.2f}s")
            await client.close()
            return False

    except Exception as e:
        print(f"[FAIL] Error during sandbox creation: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_concurrent_sandbox_creation(n_concurrent=5):
    """Test creating multiple sandboxes concurrently."""
    print("\n" + "="*60)
    print(f"TEST 3: Concurrent Sandbox Creation (n={n_concurrent})")
    print("="*60)

    try:
        from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources

        client = AsyncDaytona()

        # Create a minimal Dockerfile
        test_dockerfile = Path("/tmp/test_daytona_dockerfile/Dockerfile")
        test_dockerfile.parent.mkdir(parents=True, exist_ok=True)
        test_dockerfile.write_text("FROM python:3.11-slim\n")

        async def create_one_sandbox(idx):
            try:
                image = Image.from_dockerfile(test_dockerfile)
                params = CreateSandboxFromImageParams(
                    image=image,
                    auto_delete_interval=1,
                    resources=Resources(cpu=1, memory=1, disk=1),
                )

                start = time.time()
                sandbox = await asyncio.wait_for(
                    client.create(params=params, timeout=600),
                    timeout=660
                )
                elapsed = time.time() - start

                # Cleanup immediately
                await sandbox.delete()

                return {"idx": idx, "success": True, "elapsed": elapsed, "error": None}
            except asyncio.TimeoutError:
                return {"idx": idx, "success": False, "elapsed": time.time() - start, "error": "timeout"}
            except Exception as e:
                return {"idx": idx, "success": False, "elapsed": 0, "error": str(e)}

        print(f"Launching {n_concurrent} concurrent sandbox creations...")
        start = time.time()

        tasks = [create_one_sandbox(i) for i in range(n_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_elapsed = time.time() - start

        successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failures = n_concurrent - successes

        print(f"\nResults (total time: {total_elapsed:.2f}s):")
        print(f"  - Successes: {successes}/{n_concurrent}")
        print(f"  - Failures: {failures}/{n_concurrent}")

        for r in results:
            if isinstance(r, dict):
                status = "OK" if r["success"] else "FAIL"
                print(f"  - Sandbox {r['idx']}: [{status}] {r['elapsed']:.2f}s - {r.get('error', 'success')}")
            else:
                print(f"  - Exception: {r}")

        await client.close()
        return successes == n_concurrent

    except Exception as e:
        print(f"[FAIL] Error during concurrent test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tezos_dockerfile():
    """Test with actual tezos Dockerfile from the failing jobs."""
    print("\n" + "="*60)
    print("TEST 4: Actual Tezos Dockerfile Build")
    print("="*60)

    # Find a tezos dockerfile from the shards
    shard_base = Path("/scratch/10000/eguha3/data_cache/hf_datasets_shards/ec7cdb0cb381_8shards/shard_7")

    if not shard_base.exists():
        print(f"[SKIP] Shard directory not found: {shard_base}")
        return None

    # Find first available task with a Dockerfile
    dockerfile_path = None
    for task_dir in sorted(shard_base.iterdir())[:1]:  # Just first one
        candidate = task_dir / "environment" / "Dockerfile"
        if candidate.exists():
            dockerfile_path = candidate
            break

    if not dockerfile_path:
        print("[SKIP] No Dockerfile found in shard")
        return None

    print(f"Using Dockerfile: {dockerfile_path}")

    # Show Dockerfile content (first 20 lines)
    content = dockerfile_path.read_text()
    lines = content.split('\n')[:20]
    print("Dockerfile preview (first 20 lines):")
    for line in lines:
        print(f"  {line}")
    if len(content.split('\n')) > 20:
        print("  ...")

    try:
        from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources

        client = AsyncDaytona()

        image = Image.from_dockerfile(dockerfile_path)
        params = CreateSandboxFromImageParams(
            image=image,
            auto_delete_interval=1,
            resources=Resources(cpu=2, memory=4, disk=8),  # More resources for tezos
        )

        print(f"\nCreating sandbox with Tezos Dockerfile...")
        print(f"  - Resources: 2 CPU, 4GB RAM, 8GB disk")
        print(f"  - Build timeout: 1200s (as in production)")

        start = time.time()
        try:
            sandbox = await asyncio.wait_for(
                client.create(params=params, timeout=1200),
                timeout=1260
            )
            elapsed = time.time() - start
            print(f"[OK] Sandbox created in {elapsed:.2f}s")
            print(f"  - Sandbox ID: {sandbox.id}")

            # Test mkdir (the second operation in start())
            print("\nTesting mkdir (as in trial.py start())...")
            cmd_start = time.time()
            await sandbox.process.start_process("mkdir -p /workspace/agent /workspace/verifier")
            cmd_elapsed = time.time() - cmd_start
            print(f"[OK] mkdir completed in {cmd_elapsed:.2f}s")

            # Cleanup
            print("Deleting sandbox...")
            await sandbox.delete()
            print("[OK] Sandbox deleted")

            await client.close()
            return True

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"[FAIL] Sandbox creation TIMED OUT after {elapsed:.2f}s")
            print("\nThis confirms the issue - Daytona cannot build the Tezos image within timeout.")
            await client.close()
            return False

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_daytona_rate_limits():
    """Check current rate limit status."""
    print("\n" + "="*60)
    print("TEST 5: Rate Limit Check")
    print("="*60)

    try:
        from daytona import AsyncDaytona

        client = AsyncDaytona()

        # Try to get some info about current usage
        print("Checking Daytona service status...")

        # List any existing sandboxes (might show capacity issues)
        try:
            # There's no direct list method, but we can check via API
            print("  - Note: Daytona SDK doesn't expose rate limit info directly")
            print("  - If you see DaytonaRateLimitError, that indicates capacity issues")
        except Exception as e:
            print(f"  - Error checking: {e}")

        await client.close()

    except Exception as e:
        print(f"[FAIL] Error: {e}")


async def main():
    print("="*60)
    print("DAYTONA TIMEOUT DIAGNOSTICS")
    print("="*60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")

    # Check environment
    print("\nEnvironment check:")
    print(f"  - DAYTONA_API_KEY set: {'Yes' if os.environ.get('DAYTONA_API_KEY') else 'No'}")
    print(f"  - DAYTONA_API_URL set: {'Yes' if os.environ.get('DAYTONA_API_URL') else 'No'}")

    results = {}

    # Run tests
    results["connection"] = await test_daytona_connection()

    if results["connection"]:
        results["simple_sandbox"] = await test_simple_sandbox_creation()

        if results.get("simple_sandbox"):
            results["concurrent"] = await test_concurrent_sandbox_creation(n_concurrent=3)

    # Always try the actual tezos test
    results["tezos"] = await test_tezos_dockerfile()

    await check_daytona_rate_limits()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {test}: {status}")

    # Diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if not results.get("connection"):
        print("ISSUE: Cannot connect to Daytona service")
        print("  - Check DAYTONA_API_KEY and DAYTONA_API_URL environment variables")
        print("  - Check network connectivity")
    elif not results.get("simple_sandbox"):
        print("ISSUE: Daytona service cannot create even simple sandboxes")
        print("  - Possible rate limiting")
        print("  - Daytona service capacity issues")
        print("  - Try again later or contact Daytona support")
    elif not results.get("tezos"):
        print("ISSUE: Tezos Dockerfile takes too long to build")
        print("  - Consider pre-building the Docker image and using docker_image config")
        print("  - Consider creating Daytona snapshots for the environments")
        print("  - The 1200s timeout may not be enough for complex Dockerfiles")
    else:
        print("All tests passed - issue may be intermittent or load-related")
        print("  - Run this diagnostic during peak job times")
        print("  - Check if n_concurrent=64 is overwhelming Daytona")


if __name__ == "__main__":
    asyncio.run(main())
