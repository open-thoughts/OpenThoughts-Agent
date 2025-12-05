"""Wait for a vLLM endpoint to become healthy."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def wait_for_endpoint(
    *,
    endpoint_json: Path,
    max_attempts: int,
    retry_delay: int,
    health_path: str,
) -> None:
    if not endpoint_json.exists():
        raise FileNotFoundError(f"Endpoint JSON not found: {endpoint_json}")

    data = json.loads(endpoint_json.read_text())
    base_url = (data.get("endpoint_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("endpoint_url missing from endpoint JSON")

    path_suffix = health_path.lstrip("/")
    health_url = f"{base_url}/{path_suffix}" if path_suffix else base_url

    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(health_url, timeout=10) as resp:
                if resp.status == 200:
                    print(f"✓ Trace endpoint healthy (GET {health_url})")
                    return
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"Healthcheck attempt {attempt}/{max_attempts} failed: {exc}",
                file=sys.stderr,
            )
        if attempt < max_attempts:
            time.sleep(retry_delay)

    raise TimeoutError(
        f"Endpoint {health_url} did not become healthy after {max_attempts} attempts"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a vLLM endpoint JSON to become healthy",
    )
    parser.add_argument(
        "--endpoint-json",
        type=Path,
        required=True,
        help="Path to the endpoint JSON (e.g., vllm_endpoint.json)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Maximum number of healthcheck attempts (default: 20)",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=30,
        help="Seconds to wait between attempts (default: 30)",
    )
    parser.add_argument(
        "--health-path",
        type=str,
        default="v1/models",
        help="Relative path appended to endpoint URL for healthchecking (default: v1/models)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wait_for_endpoint(
        endpoint_json=args.endpoint_json,
        max_attempts=args.max_attempts,
        retry_delay=args.retry_delay,
        health_path=args.health_path,
    )


if __name__ == "__main__":
    main()
