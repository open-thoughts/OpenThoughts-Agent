#!/usr/bin/env python3
"""
Delete Daytona sandboxes in the RL org that have not had an event in over an hour.

Uses the Daytona REST API directly (paginated endpoint) to list all active
sandboxes, then deletes any whose `updatedAt` timestamp is older than the
configured threshold.

Usage:
    # Dry run (default) — shows what would be deleted
    python cleanup_stale_sandboxes.py

    # Actually delete
    python cleanup_stale_sandboxes.py --delete

    # Custom threshold (minutes)
    python cleanup_stale_sandboxes.py --delete --threshold 120

Environment:
    DAYTONA_API_KEY     — API key for the target organization (default)
    DAYTONA_RL_API_KEY  — Fallback API key (or set in secrets.env)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "https://app.daytona.io/api"
PAGE_LIMIT = 200  # max allowed by the paginated endpoint
DEFAULT_THRESHOLD_MINUTES = 60

# Try to load secrets.env from a few common locations
SECRET_ENV_PATH = os.environ.get("DC_AGENT_SECRET_ENV")
if SECRET_ENV_PATH and os.path.isfile(SECRET_ENV_PATH):
    load_dotenv(SECRET_ENV_PATH)
else:
    # Fallback: look next to this script or in ~/Documents
    for candidate in [
        os.path.join(os.path.dirname(__file__), "..", "..", "secrets.env"),
        os.path.expanduser("~/Documents/secrets.env"),
    ]:
        if os.path.isfile(candidate):
            load_dotenv(candidate)
            break


def get_api_key(env_var: str = "DAYTONA_API_KEY") -> str:
    key = os.environ.get(env_var)
    if not key:
        # Fallback chain: try common key names
        for fallback in ("DAYTONA_API_KEY", "DAYTONA_RL_API_KEY"):
            if fallback != env_var:
                key = os.environ.get(fallback)
                if key:
                    break
    if not key:
        sys.exit(f"ERROR: {env_var} (and fallbacks) not set in environment or secrets.env")
    return key


def headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def list_started_sandboxes(api_key: str) -> list[dict]:
    """Fetch all sandboxes in 'started' state, sorted by updatedAt desc."""
    sandboxes: list[dict] = []
    page = 1

    while True:
        resp = requests.get(
            f"{API_BASE}/sandbox/paginated",
            headers=headers(api_key),
            params={
                "states": "started",
                "sort": "updatedAt",
                "order": "desc",
                "limit": PAGE_LIMIT,
                "page": page,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        sandboxes.extend(items)
        total = data.get("total", 0)
        if len(sandboxes) >= total:
            break
        page += 1

    return sandboxes


def delete_sandbox(api_key: str, sandbox_id: str) -> bool:
    """Delete a single sandbox. Returns True on success."""
    resp = requests.delete(
        f"{API_BASE}/sandbox/{sandbox_id}",
        headers=headers(api_key),
        timeout=30,
    )
    return resp.status_code in (200, 204, 202)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_stale_sandboxes(
    sandboxes: list[dict], threshold_minutes: int
) -> list[dict]:
    """Return sandboxes whose updatedAt is older than threshold_minutes ago."""
    now = datetime.now(timezone.utc)
    stale = []

    for sb in sandboxes:
        updated_str = sb.get("updatedAt")
        if not updated_str:
            continue
        # Parse ISO-8601 timestamp (with or without trailing Z)
        updated_str = updated_str.replace("Z", "+00:00")
        updated_at = datetime.fromisoformat(updated_str)
        age_minutes = (now - updated_at).total_seconds() / 60.0
        if age_minutes > threshold_minutes:
            sb["_age_minutes"] = round(age_minutes, 1)
            stale.append(sb)

    return stale


def main():
    parser = argparse.ArgumentParser(
        description="Clean up stale Daytona RL sandboxes"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete stale sandboxes (default is dry-run)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD_MINUTES,
        help=f"Minutes of inactivity before a sandbox is considered stale (default: {DEFAULT_THRESHOLD_MINUTES})",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="DAYTONA_API_KEY",
        help="Environment variable name containing the API key (default: DAYTONA_API_KEY)",
    )
    args = parser.parse_args()

    api_key = get_api_key(args.api_key_env)

    # 1. List all active sandboxes
    print("Fetching all started sandboxes …")
    sandboxes = list_started_sandboxes(api_key)
    print(f"  Found {len(sandboxes)} started sandboxes.")

    # 2. Find stale ones
    stale = find_stale_sandboxes(sandboxes, args.threshold)
    print(f"  {len(stale)} are stale (no event in >{args.threshold} min).\n")

    if not stale:
        print("Nothing to clean up.")
        return

    # 3. Print summary
    print(f"{'ID':<40} {'Age (min)':>10}  {'Created':<26} {'Updated':<26}")
    print("-" * 110)
    for sb in stale:
        print(
            f"{sb['id']:<40} {sb['_age_minutes']:>10.1f}  "
            f"{sb['createdAt']:<26} {sb['updatedAt']:<26}"
        )

    if not args.delete:
        print(f"\nDry run — pass --delete to actually remove these {len(stale)} sandboxes.")
        return

    # 4. Delete
    print(f"\nDeleting {len(stale)} stale sandboxes …")
    success = 0
    failed = 0

    for i, sb in enumerate(stale, 1):
        sid = sb["id"]
        ok = delete_sandbox(api_key, sid)
        if ok:
            success += 1
        else:
            failed += 1
            print(f"  FAILED to delete {sid}")

        # Progress every 50
        if i % 50 == 0 or i == len(stale):
            print(f"  Progress: {i}/{len(stale)}  (ok={success}, fail={failed})")

        # Small delay to avoid rate-limiting
        time.sleep(0.05)

    print(f"\nDone. Deleted {success}/{len(stale)} sandboxes.")
    if failed:
        print(f"  {failed} deletions failed.")


if __name__ == "__main__":
    main()
