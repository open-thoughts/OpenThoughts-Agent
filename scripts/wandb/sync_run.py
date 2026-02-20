#!/usr/bin/env python3
"""
Sync W&B run data to a local directory.

Usage:
    python sync_run.py <run_id> [--output-dir DIR] [--entity ENTITY] [--project PROJECT]

Example:
    python sync_run.py xkl0jcj9 --output-dir ~/notes/rl_runs_wandb
    python sync_run.py xkl0jcj9 --entity dogml --project OpenThoughts-Agent
"""

import argparse
import json
import os
from pathlib import Path

import wandb


def sync_run(
    run_id: str,
    output_dir: Path,
    entity: str = "dogml",
    project: str = "OpenThoughts-Agent",
) -> None:
    """Download and save W&B run data to local directory."""
    api = wandb.Api()

    run_path = f"{entity}/{project}/{run_id}"
    print(f"Fetching run: {run_path}")

    run = api.run(run_path)

    # Create output directory
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    metadata = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": run.created_at,
        "heartbeat_at": run.heartbeat_at,
        "tags": run.tags,
        "notes": run.notes,
        "config": dict(run.config),
        "summary": dict(run.summary),
    }

    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {metadata_path}")

    # Save config separately for easy access
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(dict(run.config), f, indent=2, default=str)
    print(f"Saved config to {config_path}")

    # Save summary separately
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(dict(run.summary), f, indent=2, default=str)
    print(f"Saved summary to {summary_path}")

    # Download history (metrics over time)
    print("Fetching history...")
    history = list(run.scan_history())
    if history:
        history_path = run_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        print(f"Saved {len(history)} history rows to {history_path}")

    # Download files
    files_dir = run_dir / "files"
    files_dir.mkdir(exist_ok=True)

    print("Downloading files...")
    for file in run.files():
        try:
            file_path = files_dir / file.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file.download(root=str(files_dir), replace=True)
            print(f"  Downloaded: {file.name}")
        except Exception as e:
            print(f"  Failed to download {file.name}: {e}")

    print(f"\nSync complete: {run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync W&B run data to a local directory"
    )
    parser.add_argument("run_id", help="W&B run ID (e.g., xkl0jcj9)")
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path.home() / "notes" / "rl_runs_wandb",
        help="Output directory (default: ~/notes/rl_runs_wandb)",
    )
    parser.add_argument(
        "--entity", "-e",
        default="dogml",
        help="W&B entity (default: dogml)",
    )
    parser.add_argument(
        "--project", "-p",
        default="OpenThoughts-Agent",
        help="W&B project (default: OpenThoughts-Agent)",
    )

    args = parser.parse_args()

    sync_run(
        run_id=args.run_id,
        output_dir=args.output_dir,
        entity=args.entity,
        project=args.project,
    )


if __name__ == "__main__":
    main()
