#!/usr/bin/env python3
"""
Manual backup of the entire Supabase database to local JSON files.

Downloads all rows from every table and saves them as timestamped JSON files
in a backup directory. This is a logical backup (data only, no schema DDL).

Usage:
    # Source secrets first
    source ~/secrets.env  # or source /path/to/secrets.env

    # Full backup (all tables)
    python scripts/database/backup_db.py

    # Backup to custom directory
    python scripts/database/backup_db.py --output-dir /path/to/backups

    # Backup specific tables only
    python scripts/database/backup_db.py --tables models sandbox_jobs

    # Dry run (show what would be backed up)
    python scripts/database/backup_db.py --dry-run

Required environment variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY: Supabase service role key (needs read access to all tables)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Known tables in our schema (ordered by dependency for potential restore)
ALL_TABLES = [
    "datasets",
    "benchmarks",
    "agents",
    "models",
    "sandbox_jobs",
    "sandbox_trials",
    "sandbox_trial_model_usage",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backup Supabase database tables to local JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for backup files. Default: scripts/database/backups/<timestamp>/",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help=f"Tables to backup (default: all). Available: {', '.join(ALL_TABLES)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backed up without downloading",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Number of rows per API request (default: 1000)",
    )
    return parser.parse_args()


def backup_table(client, table_name: str, out_path: Path, page_size: int = 1000) -> int:
    """Download all rows from a table to a JSON file, streaming in chunks.

    Writes rows incrementally to avoid holding millions of rows in memory.
    Retries individual pages on transient errors (connection reset, timeout).

    Returns the number of rows written.
    """
    import time

    total_rows = 0
    offset = 0
    max_retries = 3

    with open(out_path, "w") as f:
        f.write("[\n")
        first = True

        while True:
            # Retry logic for transient connection errors
            for attempt in range(max_retries):
                try:
                    response = (
                        client.table(table_name)
                        .select("*")
                        .range(offset, offset + page_size - 1)
                        .execute()
                    )
                    batch = response.data
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"    Retry {attempt + 1}/{max_retries} for {table_name} "
                              f"at offset {offset} ({e}), waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise

            for row in batch:
                if not first:
                    f.write(",\n")
                json.dump(row, f, default=str)
                first = False

            total_rows += len(batch)

            # Progress for large tables
            if total_rows % 10000 == 0 and total_rows > 0:
                print(f"    {table_name}: {total_rows} rows...", flush=True)

            if len(batch) < page_size:
                break  # Last page
            offset += page_size

        f.write("\n]")

    return total_rows


def main() -> None:
    args = _parse_args()

    # Validate environment
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
        print("  source ~/secrets.env")
        sys.exit(1)

    tables = args.tables or ALL_TABLES
    for t in tables:
        if t not in ALL_TABLES:
            print(f"Warning: '{t}' is not a known table. Will attempt anyway.")

    # Create output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(__file__).parent / "backups" / timestamp
    output_dir = Path(output_dir)

    if args.dry_run:
        print(f"DRY RUN — would backup {len(tables)} tables to {output_dir}/")
        for t in tables:
            print(f"  {t}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect
    from supabase import create_client

    client = create_client(url, key)
    print(f"Backing up {len(tables)} tables to {output_dir}/")
    print()

    manifest = {
        "timestamp": timestamp,
        "supabase_url": url.split("//")[1].split(".")[0] if "//" in url else "unknown",
        "tables": {},
    }

    total_rows = 0
    for table_name in tables:
        try:
            out_path = output_dir / f"{table_name}.json"
            n_rows = backup_table(client, table_name, out_path, page_size=args.page_size)
            manifest["tables"][table_name] = {
                "rows": n_rows,
                "file": f"{table_name}.json",
            }
            total_rows += n_rows
            print(f"  {table_name}: {n_rows} rows -> {out_path.name}")
        except Exception as e:
            print(f"  {table_name}: ERROR - {e}")
            manifest["tables"][table_name] = {"error": str(e)}

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    manifest["total_rows"] = total_rows
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print(f"Backup complete: {total_rows} total rows across {len(tables)} tables")
    print(f"Output: {output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
