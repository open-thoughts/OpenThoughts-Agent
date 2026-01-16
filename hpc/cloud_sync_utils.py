"""
Sync utilities for SkyPilot-based cloud launches.

This module provides rsync-based file synchronization between
remote cloud clusters and local machines. It uses SkyPilot's
SSH configuration (cluster name as SSH host) for authentication.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


def sync_outputs(
    cluster_name: Optional[str],
    remote_path: str,
    local_dir: str,
    verbose: bool = True,
) -> bool:
    """Sync outputs from remote cluster to local machine using rsync.

    Args:
        cluster_name: SkyPilot cluster name (used as SSH host)
        remote_path: Path on the remote cluster
        local_dir: Local destination directory
        verbose: Print progress information

    Returns:
        True if sync succeeded, False otherwise
    """
    if not cluster_name:
        print("[cloud-sync] No cluster name provided, skipping sync.", file=sys.stderr)
        return False

    target = Path(local_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    # SkyPilot clusters can be accessed via SSH using the cluster name as host
    # rsync -Pavz cluster:/remote/path/ /local/path/
    remote_spec = f"{cluster_name}:{remote_path}/"
    local_spec = str(target) + "/"

    if verbose:
        print(f"[cloud-sync] Syncing outputs from cluster...")
        print(f"[cloud-sync]   Remote: {remote_spec}")
        print(f"[cloud-sync]   Local:  {local_spec}")

    # Build rsync command
    rsync_cmd = [
        "rsync",
        "-avz",  # archive, verbose, compress
        "--progress",
        remote_spec,
        local_spec,
    ]

    try:
        result = subprocess.run(
            rsync_cmd,
            check=False,
            capture_output=not verbose,
            text=True,
        )

        if result.returncode == 0:
            if verbose:
                print(f"[cloud-sync] Successfully synced outputs to {target}")
            return True
        elif result.returncode == 23:
            # rsync code 23: partial transfer due to error (often "file/directory not found")
            # This is expected when remote directory doesn't exist yet
            stderr_lower = (result.stderr or "").lower()
            if "no such file" in stderr_lower or "does not exist" in stderr_lower or "change_dir" in stderr_lower:
                if verbose:
                    print(f"[cloud-sync] Remote directory doesn't exist yet (will sync later): {remote_path}")
                return True  # Treat as success - directory will be created later
            else:
                # Other partial transfer errors should still be reported
                print(f"[cloud-sync] rsync partial transfer (code 23): {result.stderr}", file=sys.stderr)
                return False
        else:
            print(f"[cloud-sync] rsync failed with code {result.returncode}", file=sys.stderr)
            if not verbose and result.stderr:
                print(f"[cloud-sync] Error: {result.stderr}", file=sys.stderr)
            _print_manual_instructions(cluster_name, remote_path, target)
            return False

    except FileNotFoundError:
        print("[cloud-sync] rsync not found. Install it or sync manually:", file=sys.stderr)
        _print_manual_instructions(cluster_name, remote_path, target)
        return False
    except Exception as e:
        print(f"[cloud-sync] Sync failed: {e}", file=sys.stderr)
        _print_manual_instructions(cluster_name, remote_path, target)
        return False


def _print_manual_instructions(cluster_name: str, remote_path: str, local_path: Path) -> None:
    """Print manual sync instructions as fallback."""
    print(
        f"\n[cloud-sync] Manual sync instructions:\n"
        f"  rsync -avz --progress {cluster_name}:{remote_path}/ {local_path}/\n"
        f"  # or\n"
        f"  scp -r {cluster_name}:{remote_path}/* {local_path}/\n"
    )


# Backwards compatibility alias
sync_eval_outputs = sync_outputs
