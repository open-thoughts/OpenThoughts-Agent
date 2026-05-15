#!/usr/bin/env python3
"""Parse vLLM inference engine throughput from Ray worker logs.

Recursively scans a ray_logs directory for vLLM worker logs, extracts
prompt and generation throughput metrics, and reports per-engine statistics.

Usage:
    python scripts/analysis/parse_vllm_throughput.py <ray_logs_dir>

Example:
    python scripts/analysis/parse_vllm_throughput.py \
        traces/rl_traces/.../ray_logs/ray_338019_workers
"""

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Match lines like:
# Engine 000: Avg prompt throughput: 191.4 tokens/s, Avg generation throughput: 9.7 tokens/s, ...
THROUGHPUT_RE = re.compile(
    r"Engine\s+(\d+):\s+"
    r"Avg prompt throughput:\s+([\d.]+)\s+tokens/s,\s+"
    r"Avg generation throughput:\s+([\d.]+)\s+tokens/s"
)

# Match the actor name header that Ray prepends to worker logs
ACTOR_RE = re.compile(r":actor_name:AsyncVLLMInferenceEngine")


def find_vllm_worker_logs(root: Path) -> list[Path]:
    """Find worker-*.out files that contain vLLM inference engine logs."""
    candidates = []
    for f in root.rglob("worker-*.out"):
        # Quick check: read first 5 lines for the actor name tag
        try:
            with open(f) as fh:
                head = "".join(fh.readline() for _ in range(5))
            if "AsyncVLLMInferenceEngine" in head:
                candidates.append(f)
        except (OSError, UnicodeDecodeError):
            continue
    return sorted(candidates)


def parse_throughput(log_path: Path) -> dict[str, list[float]]:
    """Extract per-engine throughput samples from a single log file.

    Returns:
        Dict mapping engine_id -> {"prompt": [...], "generation": [...]}
    """
    engines: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"prompt": [], "generation": []}
    )
    with open(log_path) as f:
        for line in f:
            m = THROUGHPUT_RE.search(line)
            if m:
                engine_id = m.group(1)
                prompt_tput = float(m.group(2))
                gen_tput = float(m.group(3))
                engines[engine_id]["prompt"].append(prompt_tput)
                engines[engine_id]["generation"].append(gen_tput)
    return engines


def fmt_stats(values: list[float], total_samples: int | None = None) -> str:
    """Format min/avg/max for a list of values, with idle percentage."""
    if not values:
        return "no data"
    nonzero = [v for v in values if v > 0]
    zero_count = len(values) - len(nonzero)
    idle_pct = zero_count / len(values) * 100 if values else 0
    # Stats on non-zero values only (active throughput)
    if nonzero:
        stats = (
            f"min={min(nonzero):8.1f}  "
            f"avg={statistics.mean(nonzero):8.1f}  "
            f"max={max(nonzero):8.1f}"
        )
    else:
        stats = "    all idle"
    return f"{stats}  (n={len(values)}, idle={idle_pct:.0f}%)"


def extract_node_name(path: Path) -> str:
    """Try to extract the node name (e.g. jpbo-027-18) from the path."""
    for part in path.parts:
        if part.startswith("jpbo-") or part.startswith("lrdn") or part.startswith("nid"):
            return part
    return path.parent.parent.parent.parent.name  # fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ray_logs_dir", type=Path, help="Root directory to scan")
    args = parser.parse_args()

    if not args.ray_logs_dir.is_dir():
        print(f"Error: {args.ray_logs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    logs = find_vllm_worker_logs(args.ray_logs_dir)
    if not logs:
        print("No vLLM worker logs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(logs)} vLLM inference engine log(s)\n")

    all_prompt = []
    all_gen = []

    for log_path in logs:
        node = extract_node_name(log_path)
        engines = parse_throughput(log_path)

        for engine_id in sorted(engines):
            data = engines[engine_id]
            prompt = data["prompt"]
            gen = data["generation"]

            all_prompt.extend(prompt)
            all_gen.extend(gen)

            print(f"  {node} / Engine {engine_id}:")
            print(f"    Prompt throughput:     {fmt_stats(prompt)}")
            print(f"    Generation throughput: {fmt_stats(gen)}")
            print()

    print("=" * 72)
    print("GLOBAL SUMMARY (all engines)")
    print(f"  Prompt throughput:     {fmt_stats(all_prompt)}")
    print(f"  Generation throughput: {fmt_stats(all_gen)}")


if __name__ == "__main__":
    main()
