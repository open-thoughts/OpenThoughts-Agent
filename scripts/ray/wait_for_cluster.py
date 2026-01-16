#!/usr/bin/env python3
"""Wait for a Ray cluster to report the desired resources."""

from __future__ import annotations

import argparse
import sys
import time

import ray


def log(msg: str) -> None:
    """Print with immediate flush to avoid buffering issues."""
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for Ray cluster resources to become available.")
    parser.add_argument("--address", required=True, help="Ray head address (e.g., 10.0.0.1:6379)")
    parser.add_argument(
        "--expected-gpus",
        type=float,
        default=0.0,
        help="Total number of GPU resources expected across the cluster (default: 0)",
    )
    parser.add_argument(
        "--expected-nodes",
        type=int,
        default=1,
        help="Number of nodes expected in the cluster (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds before giving up (default: 600)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between cluster inspections (default: 5)",
    )
    return parser.parse_args()


def wait_for_cluster(address: str, expected_gpus: float, expected_nodes: int, timeout: int, poll_interval: int) -> None:
    """Poll the Ray head until the desired resources are visible."""
    deadline = time.time() + timeout

    log(f"Connecting to Ray at {address} (expecting {expected_nodes} nodes, {expected_gpus} GPUs)")

    # Retry the initial connection - ray.init() has its own 30s timeout
    # and the Ray head may not be ready immediately after ray start
    connected = False
    while time.time() < deadline:
        try:
            ray.init(address=address, ignore_reinit_error=True)
            connected = True
            log("Ray connection established, polling for resources...")
            break
        except (ConnectionError, RuntimeError) as e:
            log(f"[Ray wait] Connection attempt failed: {e}")
            log(f"[Ray wait] Retrying in {poll_interval}s...")
            try:
                ray.shutdown()
            except Exception:
                pass
            time.sleep(poll_interval)

    if not connected:
        raise TimeoutError(
            f"Could not connect to Ray cluster at {address} within {timeout} seconds"
        )

    try:
        while True:
            resources = ray.cluster_resources()
            num_nodes = len(ray.nodes())
            available_gpus = resources.get("GPU", 0.0)

            log(
                f"[Ray wait] nodes={num_nodes}/{expected_nodes} "
                f"GPUs={available_gpus}/{expected_gpus} "
                f"resources={resources}"
            )

            if num_nodes >= expected_nodes and available_gpus >= expected_gpus:
                log("✓ Ray cluster ready")
                return

            if time.time() > deadline:
                raise TimeoutError(
                    f"Ray cluster did not reach desired resources within {timeout} seconds "
                    f"(nodes={num_nodes}, gpus={available_gpus})"
                )

            time.sleep(poll_interval)
    finally:
        ray.shutdown()


def main() -> None:
    args = parse_args()
    wait_for_cluster(
        address=args.address,
        expected_gpus=args.expected_gpus,
        expected_nodes=args.expected_nodes,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
