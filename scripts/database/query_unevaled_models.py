#!/usr/bin/env python3
"""
Query Supabase for models that have NOT been evaluated on a given benchmark family.

Uses the `duplicate_of` field in the benchmarks table to resolve benchmark families:
a parent benchmark and all its children (duplicates) form one family. A model is
considered "evaluated" if it has ANY sandbox_job (Finished, Started, or Pending)
against ANY benchmark in that specific family. Each family is independent — a
dev_set_v2 result does NOT count as evaluated for terminal_bench_2.

Size matching walks the base_model_id chain to the root ancestor and classifies
by the root's model_size_b. For example, any model whose root is Qwen/Qwen3-8B
(8.2B) will match --size 8. Falls back to the model's own model_size_b or name
heuristic if the chain is missing.

Usage:
    # List 8B models not yet evaluated on dev_set_v2 (and its duplicates)
    python scripts/database/query_unevaled_models.py --benchmark dev_set_v2 --size 8

    # List 32B models not yet evaluated on terminal_bench_2
    python scripts/database/query_unevaled_models.py --benchmark terminal_bench_2 --size 32

    # Write output to a file (for use as eval priority list)
    python scripts/database/query_unevaled_models.py --benchmark dev_set_v2 --size 8 -o eval/lists/models_8b_dsv2_remaining.txt

    # Show all models (no size filter)
    python scripts/database/query_unevaled_models.py --benchmark dev_set_v2

    # Exclude models matching patterns (e.g. test models, paths)
    python scripts/database/query_unevaled_models.py --benchmark dev_set_v2 --size 8 --exclude "test_" --exclude "NO_EVAL"

Requires: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Set, Tuple


def get_client():
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set", file=sys.stderr)
        sys.exit(1)
    return create_client(url, key)


def resolve_benchmark_family(client, benchmark_name: str) -> Tuple[str, Set[str]]:
    """Resolve a benchmark name to its full family (parent + all children).

    Returns:
        Tuple of (parent_id, set_of_all_family_ids)
    """
    benchmarks = client.table("benchmarks").select("id,name,duplicate_of").execute()

    # Find the target benchmark
    name_to_id = {b["name"]: b["id"] for b in benchmarks.data}
    id_to_name = {b["id"]: b["name"] for b in benchmarks.data}

    if benchmark_name not in name_to_id:
        print(f"Error: benchmark '{benchmark_name}' not found.", file=sys.stderr)
        print(f"Available benchmarks: {sorted(name_to_id.keys())}", file=sys.stderr)
        sys.exit(1)

    target_id = name_to_id[benchmark_name]

    # Find the root parent: follow duplicate_of chain up
    parent_id = target_id
    for b in benchmarks.data:
        if b["id"] == target_id and b["duplicate_of"]:
            parent_id = b["duplicate_of"]
            break

    # Collect all family members: parent + all children pointing to parent
    family_ids = {parent_id}
    for b in benchmarks.data:
        if b["duplicate_of"] == parent_id:
            family_ids.add(b["id"])

    family_names = [id_to_name.get(fid, fid) for fid in family_ids]
    return parent_id, family_ids, family_names


def get_evaled_model_ids(client, benchmark_family_ids: Set[str]) -> Set[str]:
    """Get all model IDs that have any sandbox_job against the benchmark family."""
    evaled = set()
    for bid in benchmark_family_ids:
        jobs = client.table("sandbox_jobs").select("model_id,job_status").eq("benchmark_id", bid).execute()
        for j in jobs.data:
            # Any status counts — Finished, Started, Pending
            if j.get("model_id"):
                evaled.add(j["model_id"])
    return evaled


def _size_in_bucket(model_size_b: Optional[float], target_size: int) -> bool:
    """Check if a model size falls in the target bucket.

    e.g. 8.2B matches --size 8, 32.8B matches --size 32.
    Uses a range: [target, target * 1.5).
    """
    if model_size_b is None:
        return False
    return target_size <= model_size_b < target_size * 1.5


def _build_root_size_map(models_data: list) -> Dict[str, Optional[float]]:
    """Walk base_model_id chains to find each model's root ancestor size.

    Returns {model_id: root_model_size_b}.
    """
    by_id = {m["id"]: m for m in models_data}
    cache: Dict[str, Optional[float]] = {}

    def _get_root_size(model_id: str, visited: Optional[Set[str]] = None) -> Optional[float]:
        if model_id in cache:
            return cache[model_id]
        if visited is None:
            visited = set()
        if model_id in visited:
            # Cycle — shouldn't happen, but guard against it
            return by_id.get(model_id, {}).get("model_size_b")
        visited.add(model_id)

        m = by_id.get(model_id)
        if m is None:
            return None

        parent_id = m.get("base_model_id")
        if parent_id and parent_id in by_id:
            root_size = _get_root_size(parent_id, visited)
        else:
            # This is a root model — use its own size
            root_size = m.get("model_size_b")

        cache[model_id] = root_size
        return root_size

    for m in models_data:
        _get_root_size(m["id"])

    return cache


def get_models(client, size: Optional[int] = None) -> Dict[str, str]:
    """Get all models, optionally filtered by size. Returns {id: name}.

    Size classification walks each model's base_model_id chain to the root
    ancestor and uses the root's model_size_b. Falls back to the model's
    own size or name heuristic.
    """
    models = client.table("models").select("id,name,model_size_b,base_model_id").execute()

    if size is not None:
        root_sizes = _build_root_size_map(models.data)

    result = {}
    for m in models.data:
        name = m.get("name", "")
        if not name or name.startswith("/"):
            continue

        if size is not None:
            # Primary: classify by root ancestor size
            root_size = root_sizes.get(m["id"])
            if _size_in_bucket(root_size, size):
                result[m["id"]] = name
                continue
            # Fallback: model's own size (for orphans with no chain)
            own_size = m.get("model_size_b")
            if own_size and _size_in_bucket(own_size, size):
                result[m["id"]] = name
                continue
            # Last resort: name heuristic (for models with no size at all)
            if own_size is None and root_size is None:
                size_str = f"{size}B"
                size_str_lower = f"{size}b"
                if size_str in name or size_str_lower in name:
                    result[m["id"]] = name
        else:
            result[m["id"]] = name

    return result


def main():
    parser = argparse.ArgumentParser(description="Query unevaled models for a benchmark family")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g. dev_set_v2, terminal_bench_2)")
    parser.add_argument("--size", type=int, default=None, help="Model size in B (e.g. 8, 32). Omit for all sizes.")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude models containing this substring (repeatable)")
    parser.add_argument("-o", "--output", default=None, help="Write model names to file (one per line)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show benchmark family and stats")
    args = parser.parse_args()

    client = get_client()

    # Resolve benchmark family
    parent_id, family_ids, family_names = resolve_benchmark_family(client, args.benchmark)
    if args.verbose:
        print(f"Benchmark family for '{args.benchmark}':", file=sys.stderr)
        for fn in sorted(family_names):
            print(f"  - {fn}", file=sys.stderr)

    # Get models
    models = get_models(client, args.size)
    if args.verbose:
        print(f"\nTotal {args.size or 'all'}B models in DB: {len(models)}", file=sys.stderr)

    # Get evaled model IDs
    evaled_ids = get_evaled_model_ids(client, family_ids)
    if args.verbose:
        evaled_in_scope = len(evaled_ids & set(models.keys()))
        print(f"Already evaluated (any status): {evaled_in_scope}", file=sys.stderr)

    # Filter to remaining
    remaining = {mid: name for mid, name in models.items() if mid not in evaled_ids}

    # Apply exclusion patterns
    for pattern in args.exclude:
        remaining = {mid: name for mid, name in remaining.items() if pattern not in name}

    remaining_names = sorted(remaining.values())

    if args.verbose:
        print(f"Remaining (not evaluated): {len(remaining_names)}", file=sys.stderr)
        print(file=sys.stderr)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            for name in remaining_names:
                f.write(name + "\n")
        print(f"Wrote {len(remaining_names)} models to {args.output}", file=sys.stderr)
    else:
        for name in remaining_names:
            print(name)


if __name__ == "__main__":
    main()
