#!/usr/bin/env python3
"""Query sandbox_jobs for entries containing mean_drop_ei_tasks_dropped in stats JSON."""

import csv
import json
import os
import sys

from supabase import create_client


def find_field(obj, target_key):
    """Recursively search a nested dict/list for target_key, yield (path, value) tuples."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target_key:
                yield (k, v)
            else:
                yield from find_field(v, target_key)
    elif isinstance(obj, list):
        for item in obj:
            yield from find_field(item, target_key)


def main():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set", file=sys.stderr)
        sys.exit(1)

    client = create_client(url, key)

    # Paginate through all sandbox_jobs rows (Supabase caps at 1000 per request)
    rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            client.table("sandbox_jobs")
            .select("id,job_name,stats,metrics")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not resp.data:
            break
        rows.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size

    print(f"Fetched {len(rows)} sandbox_jobs rows")

    results = []
    for row in rows:
        stats = row.get("stats")
        if not stats:
            continue
        for _, value in find_field(stats, "mean_drop_ei_tasks_dropped"):
            # Extract accuracy from metrics list
            metrics = row.get("metrics") or []
            accuracy = next((m["value"] for m in metrics if m.get("name") == "accuracy"), None)

            # Extract mean_drop_ei_reward from stats
            reward_val = None
            for _, rv in find_field(stats, "mean_drop_ei_reward"):
                reward_val = rv
                break

            results.append({
                "id": row["id"],
                "job_name": row["job_name"],
                "accuracy": accuracy,
                "mean_drop_ei_reward": reward_val,
                "mean_drop_ei_tasks_dropped": value,
            })

    print(f"Found {len(results)} entries with mean_drop_ei_tasks_dropped")

    out_path = "/Users/benjaminfeuer/Documents/mean_drop_ei_tasks_dropped.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "job_name", "accuracy", "mean_drop_ei_reward", "mean_drop_ei_tasks_dropped"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
