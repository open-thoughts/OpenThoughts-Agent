#!/usr/bin/env python3
"""
Check eval status of recently-added models across multiple benchmark families.

For the firing agent: scope to models added to the DB in the last N days,
show per-benchmark status, and write per-benchmark "still needs firing"
priority files ready for unified_eval_listener.py.

Default scope: models with creation_time within the last 7 days, sized 32B,
checked against terminal_bench_2 + swebench-verified-random-100-folders.

Examples:
  # Default: 32B, last 7 days, tb2 + swebench
  python scripts/database/check_firing_candidates.py

  # 8B last 3 days, write priority files
  python scripts/database/check_firing_candidates.py --size 8 --max-age-days 3 \
      --output-dir eval/.local_notes/uneval_lists

  # Add an include filter on model name
  python scripts/database/check_firing_candidates.py --include EtashGuha

  # Different benchmarks
  python scripts/database/check_firing_candidates.py \
      --benchmarks terminal_bench_2,dev_set_v2

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (sourced from
hpc/dotenv/jupiter_eval.env).
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Reuse helpers from the unevaled-models query.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from query_unevaled_models import (  # noqa: E402
    _build_root_size_map,
    _size_in_bucket,
    get_client,
    resolve_benchmark_family,
)


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def get_recent_models(
    client,
    cutoff: datetime,
    size: Optional[int],
    includes: List[str],
    excludes: List[str],
) -> List[dict]:
    """Return models with creation_time >= cutoff, optionally size-filtered."""
    # Pull once, classify in-process. Models table is small enough.
    models = client.table("models").select(
        "id,name,model_size_b,base_model_id,creation_time"
    ).execute().data
    root_sizes = _build_root_size_map(models) if size is not None else {}

    out = []
    for m in models:
        name = m.get("name") or ""
        if not name or name.startswith("/"):
            continue
        ts = m.get("creation_time")
        if not ts:
            continue
        if parse_iso(ts) < cutoff:
            continue
        if includes and not any(s in name for s in includes):
            continue
        if any(s in name for s in excludes):
            continue
        if size is not None:
            root_size = root_sizes.get(m["id"])
            own_size = m.get("model_size_b")
            if not (
                _size_in_bucket(root_size, size)
                or _size_in_bucket(own_size, size)
                or (
                    root_size is None
                    and own_size is None
                    and (f"{size}B" in name or f"{size}b" in name)
                )
            ):
                continue
        out.append(m)
    return out


def get_evaled_for_family(client, family_ids: Set[str]) -> Set[str]:
    """All model_ids with any sandbox_job (any status) against the family."""
    evaled = set()
    for bid in family_ids:
        # Filter to non-null model_id at query time to keep the response small.
        rows = client.table("sandbox_jobs").select("model_id").eq(
            "benchmark_id", bid
        ).execute().data
        for r in rows:
            mid = r.get("model_id")
            if mid:
                evaled.add(mid)
    return evaled


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main():
    p = argparse.ArgumentParser(
        description="Check firing candidates: recent models × benchmarks."
    )
    p.add_argument(
        "--benchmarks",
        default="terminal_bench_2,swebench-verified-random-100-folders",
        help="Comma-separated benchmark names (resolved to families via duplicate_of).",
    )
    p.add_argument("--size", type=int, default=32, help="Size bucket in B (default 32).")
    p.add_argument("--max-age-days", type=int, default=7, help="Max model age in days (default 7).")
    p.add_argument("--include", action="append", default=[], help="Substring filter (repeatable).")
    p.add_argument("--exclude", action="append", default=[], help="Substring exclusion (repeatable).")
    p.add_argument(
        "--output-dir",
        default=None,
        help="If set, write per-benchmark priority files of unevaled models.",
    )
    p.add_argument(
        "--blacklist-output-dir",
        default=None,
        help="If set, write per-benchmark blacklist files (all evaled models in DB) here.",
    )
    args = p.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if not benchmarks:
        print("Error: at least one benchmark required.", file=sys.stderr)
        sys.exit(1)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.max_age_days)
    client = get_client()

    # Resolve every benchmark family.
    families: Dict[str, Tuple[str, Set[str]]] = {}
    for b in benchmarks:
        _parent, family_ids, family_names = resolve_benchmark_family(client, b)
        families[b] = (family_ids, family_names)

    # Pull recent candidates.
    recent = get_recent_models(client, cutoff, args.size, args.include, args.exclude)
    if not recent:
        print(
            f"No models found with creation_time >= {cutoff.isoformat()} "
            f"(size={args.size}, includes={args.include}, excludes={args.exclude}).",
            file=sys.stderr,
        )
        return

    # Per-benchmark evaled sets.
    evaled_by_bench: Dict[str, Set[str]] = {}
    for b, (family_ids, _names) in families.items():
        evaled_by_bench[b] = get_evaled_for_family(client, family_ids)

    # Build status matrix: rows = models, cols = benchmarks.
    rows = []
    for m in sorted(recent, key=lambda x: x.get("creation_time") or ""):
        statuses = {b: (m["id"] in evaled_by_bench[b]) for b in benchmarks}
        rows.append((m, statuses))

    # Print summary.
    print(
        f"Recent models (creation_time >= {cutoff.date().isoformat()} UTC, "
        f"size={args.size}B, n={len(rows)}):",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print(f"  {'created':<10} {'name':<70} " + " ".join(f"{b[:18]:<18}" for b in benchmarks))
    print(f"  {'-' * 10} {'-' * 70} " + " ".join(f"{'-' * 18:<18}" for b in benchmarks))
    for m, statuses in rows:
        created = (m.get("creation_time") or "")[:10]
        cells = " ".join(
            f"{('✓ evaled' if statuses[b] else '✗ MISSING'):<18}" for b in benchmarks
        )
        print(f"  {created:<10} {m['name']:<70} {cells}")

    # Per-benchmark unevaled priority lists.
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in benchmarks:
            missing = sorted(m["name"] for m, st in rows if not st[b])
            path = out_dir / f"uneval_{sanitize(b)}_{args.size}b_{args.max_age_days}d.txt"
            path.write_text("\n".join(missing) + ("\n" if missing else ""))
            print(f"\nWrote {len(missing)} models to {path}", file=sys.stderr)

    # Combined "needs any benchmark" list.
    if args.output_dir:
        any_missing = sorted(
            {m["name"] for m, st in rows if not all(st.values())}
        )
        path = Path(args.output_dir) / f"uneval_any_{args.size}b_{args.max_age_days}d.txt"
        path.write_text("\n".join(any_missing) + ("\n" if any_missing else ""))
        print(f"Wrote {len(any_missing)} models to {path}", file=sys.stderr)

    # Optional: dump full blacklist (all evaled models per benchmark, regardless of age).
    if args.blacklist_output_dir:
        bl_dir = Path(args.blacklist_output_dir)
        bl_dir.mkdir(parents=True, exist_ok=True)
        # We need names for evaled ids, so map ids back via the models table.
        all_models = {
            m["id"]: m["name"]
            for m in client.table("models").select("id,name").execute().data
            if m.get("name")
        }
        for b in benchmarks:
            evaled_names = sorted(
                all_models[mid] for mid in evaled_by_bench[b] if mid in all_models
            )
            path = bl_dir / f"blacklist_{sanitize(b)}.txt"
            path.write_text("\n".join(evaled_names) + ("\n" if evaled_names else ""))
            print(
                f"Wrote {len(evaled_names)} evaled models to {path} (blacklist)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
