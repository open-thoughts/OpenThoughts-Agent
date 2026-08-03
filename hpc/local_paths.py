"""Single source of truth for OT-Agent's local persistent storage.

Every place the launcher (or any helper it calls) writes durable data on
the user's laptop registers its path constant here, under the
``~/.ot-agent/`` root. This makes inventory, migration, and cleanup
mechanical instead of archeological.

Layout::

    ~/.ot-agent/
        state/      SQLite registries (e.g. iris job → GCS prefix map)
        runs/       Fetched job outputs, keyed by job_name
        cache/      Reusable downloads (hf datasets, tiktoken, ...)
        logs/       Daemon + launcher logs

Legacy locations (pre-2026-05-22) are left in place by convention; the
``inventory`` CLI surfaces them so they can be migrated or wiped
explicitly. See ``notes/marin/flows/iris-outputs-redesign.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


# Root for every persistent local write the launcher controls. A single
# directory means "wipe ot-agent state" is `rm -rf ~/.ot-agent` and
# "back up ot-agent state" is `tar -C ~ -cf ot-agent-state.tar .ot-agent`.
# Override with OT_AGENT_HOME for tests or alternate users on the same
# machine.
OT_AGENT_HOME = Path(os.environ.get("OT_AGENT_HOME", str(Path.home() / ".ot-agent")))


@dataclass(frozen=True)
class _Layout:
    """Canonical subdirectory roots under :data:`OT_AGENT_HOME`."""

    home: Path
    state: Path
    runs: Path
    cache: Path
    logs: Path


def _build_layout(root: Path) -> _Layout:
    return _Layout(
        home=root,
        state=root / "state",
        runs=root / "runs",
        cache=root / "cache",
        logs=root / "logs",
    )


PATHS: _Layout = _build_layout(OT_AGENT_HOME)


# Pre-2026-05-22 locations the launcher and its helpers used to write
# without coordination. Inventory-only — nothing migrates them
# automatically. Keep the tuple sorted by path string so `inventory`
# output is stable across runs.
LEGACY_PATHS: tuple[tuple[str, Path], ...] = (
    ("cloud_runs (old rsync dest)", Path.home() / "cloud_runs"),
    ("hf cache (sky launcher)",
     Path.home() / "Documents" / "OpenThoughts-Agent" / ".hf_cloud_cache"),
    ("tiktoken cache", Path.home() / ".cache" / "tiktoken_encodings"),
)


def ensure(*subdirs: Path) -> None:
    """Create the requested OT-Agent subdirectories (and their parents).

    Call this at the start of any helper that's about to write into
    ``PATHS.*``. Idempotent. Raises if any path is outside
    :data:`OT_AGENT_HOME` (guards against caller typos that would leak
    state outside the managed tree).
    """
    for p in subdirs:
        p = Path(p)
        if OT_AGENT_HOME not in p.parents and p != OT_AGENT_HOME:
            raise ValueError(
                f"Refusing to create {p}: not under managed root {OT_AGENT_HOME}. "
                "Add a new field to local_paths._Layout if this is intentional."
            )
        p.mkdir(parents=True, exist_ok=True)


def _dir_size_bytes(path: Path) -> int:
    """Sum of file sizes under ``path``, or 0 if missing.

    Accepts either a directory (walks recursively) or a single file
    (returns its size). Tolerates broken symlinks and permission errors
    so an inventory call never raises mid-walk.
    """
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                continue
    return total


@dataclass(frozen=True)
class InventoryEntry:
    name: str
    path: Path
    kind: str  # "managed" | "legacy"
    exists: bool
    size_bytes: int
    mtime: Optional[float]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "kind": self.kind,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
        }


def inventory() -> List[InventoryEntry]:
    """Return one row per persistent location, managed and legacy.

    Used by ``python -m hpc.local_paths inventory`` and (later) the
    daemon's ``status`` view.
    """
    managed: list[tuple[str, Path]] = [
        ("state",       PATHS.state),
        ("runs",        PATHS.runs),
        ("cache",       PATHS.cache),
        ("logs",        PATHS.logs),
        ("cache/hf",    PATHS.cache / "hf"),
        ("cache/tiktoken", PATHS.cache / "tiktoken"),
        # Individual SQLite catalogs under state/. Listing them explicitly
        # makes "did the registry initialize?" answerable from `inventory`.
        ("state/iris_jobs.db",     PATHS.state / "iris_jobs.db"),
        ("state/model_mirrors.db", PATHS.state / "model_mirrors.db"),
    ]

    entries: list[InventoryEntry] = []
    for name, path in managed:
        exists = path.exists()
        entries.append(InventoryEntry(
            name=name, path=path, kind="managed",
            exists=exists,
            size_bytes=_dir_size_bytes(path) if exists else 0,
            mtime=path.stat().st_mtime if exists else None,
        ))
    for name, path in LEGACY_PATHS:
        exists = path.exists()
        entries.append(InventoryEntry(
            name=name, path=path, kind="legacy",
            exists=exists,
            size_bytes=_dir_size_bytes(path) if exists else 0,
            mtime=path.stat().st_mtime if exists else None,
        ))
    return entries


def _humanize_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:7.1f} {unit}"
        n //= 1024
    return f"{n} ?"


def _format_inventory(entries: Iterable[InventoryEntry]) -> str:
    rows = [
        f"{'KIND':<8} {'NAME':<20} {'EXISTS':<7} {'SIZE':>12}  {'MTIME':<20}  PATH",
        "-" * 110,
    ]
    for e in entries:
        mtime_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.mtime)) if e.mtime else "-"
        rows.append(
            f"{e.kind:<8} {e.name:<20} {str(e.exists):<7} "
            f"{_humanize_bytes(e.size_bytes):>12}  {mtime_s:<20}  {e.path}"
        )
    return "\n".join(rows)


def _cli_inventory(args: argparse.Namespace) -> int:
    entries = inventory()
    if args.json:
        print(json.dumps([e.to_dict() for e in entries], indent=2, default=str))
    else:
        print(_format_inventory(entries))
    return 0


def _cli_clean(args: argparse.Namespace) -> int:
    """Delete fetched runs and rotated logs older than ``--older-than`` days.

    Never touches ``state`` or ``cache`` — those are the registry +
    reusable downloads, not run output. Dry-run by default.
    """
    threshold = time.time() - args.older_than * 86400
    to_delete: list[Path] = []
    for parent in (PATHS.runs, PATHS.logs):
        if not parent.exists():
            continue
        for child in sorted(parent.iterdir()):
            try:
                if child.stat().st_mtime < threshold:
                    to_delete.append(child)
            except OSError:
                continue

    if not to_delete:
        print(f"Nothing older than {args.older_than}d under {PATHS.runs} or {PATHS.logs}.")
        return 0

    print(f"{'Would delete' if args.dry_run else 'Deleting'} {len(to_delete)} entries "
          f"older than {args.older_than}d:")
    for p in to_delete:
        size = _humanize_bytes(_dir_size_bytes(p) if p.is_dir() else os.path.getsize(p))
        print(f"  {size}  {p}")
        if not args.dry_run:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    return 0


def _cli_migrate(args: argparse.Namespace) -> int:
    """Move legacy locations into the managed tree.

    No-op by default; pass --apply to actually move. We keep this
    explicit per the design decision ("leave legacy in place"); the
    command is here only so the inventory output is actionable.
    """
    moves: list[tuple[Path, Path]] = []
    for name, src in LEGACY_PATHS:
        if not src.exists():
            continue
        if "tiktoken" in name:
            dst = PATHS.cache / "tiktoken"
        elif "hf cache" in name:
            dst = PATHS.cache / "hf"
        elif "cloud_runs" in name:
            dst = PATHS.runs
        else:
            continue
        moves.append((src, dst))

    if not moves:
        print("No legacy paths to migrate.")
        return 0

    print(f"{'Migrating' if args.apply else 'Would migrate'}:")
    for src, dst in moves:
        print(f"  {src}  -->  {dst}")
        if args.apply:
            ensure(dst.parent if dst.parent != OT_AGENT_HOME else OT_AGENT_HOME)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                # Merge children rather than overwrite.
                for child in src.iterdir():
                    target = dst / child.name
                    if target.exists():
                        print(f"    skip (collision): {target}")
                        continue
                    shutil.move(str(child), str(target))
                src.rmdir()
            else:
                shutil.move(str(src), str(dst))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m hpc.local_paths",
        description="Inspect, migrate, and clean OT-Agent's local persistent state.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inventory", help="List every persistent path the launcher writes to.")
    pi.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    pi.set_defaults(func=_cli_inventory)

    pc = sub.add_parser("clean",
                        help="Delete runs/ and logs/ entries older than --older-than days.")
    pc.add_argument("--older-than", type=int, default=30,
                    help="Age threshold in days (default 30).")
    pc.add_argument("--dry-run", action="store_true", default=True,
                    help="Default; pass --apply to delete.")
    pc.add_argument("--apply", action="store_false", dest="dry_run",
                    help="Actually delete the listed entries.")
    pc.set_defaults(func=_cli_clean)

    pm = sub.add_parser("migrate",
                        help="Move legacy paths into the managed tree (~/.ot-agent/).")
    pm.add_argument("--apply", action="store_true",
                    help="Actually move; default is dry-run.")
    pm.set_defaults(func=_cli_migrate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
