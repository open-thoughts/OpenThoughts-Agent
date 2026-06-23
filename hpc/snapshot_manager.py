"""Unified Daytona snapshot management for hpc.launch (RL, datagen, eval).

This module is the single source of truth for pre-building, listing, and
cleaning up the Daytona snapshots that Harbor uses at trial time via
``auto_snapshot=true``. It replaces the ad-hoc primitive that previously
lived only in ``hpc/rl_launch_utils.py`` and consolidates additional
operational logic ported (re-implemented, not imported) from
``data/sbatches/register_snapshots.py``:

  - Multi-org support via an explicit ``orgs: list[OrgConfig]`` arg.
  - PENDING-state wait (poll every 5s, cap at ``pending_wait_s``).
  - Persistent JSONL registry as a fast-path cache (always reconfirmed
    via ``snapshot.get`` before trusting).
  - Rate-limit backoff around every SDK call.
  - Separate, manual ``cleanup_unused_snapshots`` primitive (NEVER auto-
    invoked from ``ensure_snapshots``).
  - Hard-fail on cap overrun (preserves the previous RL behavior).

The contract with Harbor is the snapshot-name function — both this module
and Harbor's ``DaytonaEnvironment._get_auto_snapshot_name`` compute the
same ``harbor__{hash}__snapshot`` string from the same env-dir hash.

Reused from elsewhere in the repo (NOT in data/sbatches/):
  - ``scripts.harbor.count_snapshots_from_tasks.discover_task_dirs``
  - ``scripts.harbor.count_snapshots_from_tasks.get_snapshot_env_dirs``
  - ``harbor.utils.container_cache.analyze_task_dockerfiles``
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


# -----------------------------------------------------------------------------
# Public dataclasses + exceptions
# -----------------------------------------------------------------------------

@dataclass
class OrgConfig:
    """A single Daytona org/key the manager will register snapshots on."""
    name: str
    api_key: str
    api_url: Optional[str] = None
    target: str = "us"


@dataclass
class SnapshotInfo:
    name: str
    hash: str
    org: str
    state: str  # ACTIVE | PENDING | ERROR | MISSING | OVER_CAP


@dataclass
class StatusCounts:
    active: int = 0
    pending: int = 0
    error: int = 0
    missing: int = 0


@dataclass
class SnapshotPlanResult:
    per_org: Dict[str, StatusCounts] = field(default_factory=dict)
    built: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class CleanupResult:
    per_org: Dict[str, int] = field(default_factory=dict)


class SnapshotCapExceeded(Exception):
    """Raised when ``total_existing + new_needed > max_org_snapshots`` for any org."""


class SnapshotBuildFailed(Exception):
    """Raised when a build never reaches ACTIVE within ``pending_wait_s``."""


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_REGISTRY_ENV_VAR = "OT_AGENT_SNAPSHOT_REGISTRY"
_DEFAULT_REGISTRY_PATH = Path.home() / ".cache" / "ot-agent" / "daytona_snapshot_registry.jsonl"


def _resolve_registry_path(registry_path: Optional[Path]) -> Path:
    if registry_path is not None:
        return Path(registry_path)
    env_override = os.environ.get(_REGISTRY_ENV_VAR)
    if env_override:
        return Path(env_override)
    return _DEFAULT_REGISTRY_PATH


def _snapshot_name(env_hash: str, target_region: str) -> str:
    """Compute the canonical snapshot name. Must match Harbor's naming."""
    if target_region:
        return f"harbor__{env_hash}__{target_region}__snapshot"
    return f"harbor__{env_hash}__snapshot"


def _is_active(state: Any) -> bool:
    return state is not None and str(state).upper() in ("ACTIVE", "SNAPSHOTSTATE.ACTIVE")


def _is_pending(state: Any) -> bool:
    return state is not None and str(state).upper() in ("PENDING", "BUILDING", "SNAPSHOTSTATE.PENDING", "SNAPSHOTSTATE.BUILDING")


def _is_error(state: Any) -> bool:
    return state is not None and str(state).upper() in (
        "ERROR", "BUILD_FAILED", "FAILED",
        "SNAPSHOTSTATE.ERROR", "SNAPSHOTSTATE.BUILD_FAILED", "SNAPSHOTSTATE.FAILED",
    )


def _normalized_state(state: Any) -> str:
    if _is_active(state):
        return "ACTIVE"
    if _is_pending(state):
        return "PENDING"
    if _is_error(state):
        return "ERROR"
    return "MISSING"


def _with_rate_limit_backoff(call: Callable[[], Any], *, max_attempts: int = 5) -> Any:
    """Retry an SDK call on transient errors with exponential-ish backoff.

    Catches Daytona rate-limit exceptions if importable, plus any error whose
    type or message matches Harbor's transient predicate when available.
    """
    rate_limit_types: Tuple[type, ...] = ()
    try:
        from daytona.common.errors import DaytonaRateLimitError  # type: ignore
        rate_limit_types = (DaytonaRateLimitError,)
    except Exception:
        pass

    is_transient_predicate: Optional[Callable[[Exception], bool]] = None
    try:
        from harbor.environments.daytona_utils import is_transient_daytona_error  # type: ignore
        is_transient_predicate = is_transient_daytona_error
    except Exception:
        is_transient_predicate = None

    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return call()
        except rate_limit_types as exc:  # type: ignore[misc]
            last_exc = exc
            delay = 30 * (attempt + 1)
            print(f"  rate-limited; sleeping {delay}s (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
            continue
        except Exception as exc:
            if is_transient_predicate is not None and is_transient_predicate(exc):
                last_exc = exc
                delay = 5 * (attempt + 1)
                print(f"  transient error: {exc}; sleeping {delay}s")
                time.sleep(delay)
                continue
            raise
    assert last_exc is not None
    raise last_exc


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

class _RegistryWriter:
    """Append-only JSONL registry, folded to latest-per-(hash, org) on read."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Return ``{(hash, org): latest_record}``."""
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if not self.path.exists():
            return out
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rec.get("hash", ""), rec.get("org", ""))
                if not key[0] or not key[1]:
                    continue
                out[key] = rec
        return out

    def append(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# -----------------------------------------------------------------------------
# Daytona client construction (factory pattern for testability)
# -----------------------------------------------------------------------------

def _default_daytona_factory(org: OrgConfig):
    """Create a real ``Daytona`` client for the given org."""
    from daytona import Daytona, DaytonaConfig  # type: ignore
    cfg_kwargs: Dict[str, Any] = {"api_key": org.api_key, "target": org.target}
    if org.api_url:
        cfg_kwargs["api_url"] = org.api_url
    return Daytona(DaytonaConfig(**cfg_kwargs))


# -----------------------------------------------------------------------------
# Core manager
# -----------------------------------------------------------------------------

class _SnapshotManager:
    """Per-launch worker that knows how to talk to one or more Daytona orgs."""

    def __init__(
        self,
        orgs: List[OrgConfig],
        *,
        registry_path: Optional[Path] = None,
        daytona_factory: Callable[[OrgConfig], Any] = _default_daytona_factory,
    ) -> None:
        if not orgs:
            raise ValueError("orgs must be non-empty")
        self.orgs = orgs
        self.registry = _RegistryWriter(_resolve_registry_path(registry_path))
        self._daytona_factory = daytona_factory
        self._clients: Dict[str, Any] = {}

    def _client(self, org: OrgConfig) -> Any:
        if org.name not in self._clients:
            self._clients[org.name] = self._daytona_factory(org)
        return self._clients[org.name]

    # --- read paths ------------------------------------------------------

    def _get_state(self, client: Any, name: str) -> Tuple[str, Optional[Any]]:
        """Return (normalized_state, raw_snapshot_or_None)."""
        try:
            from daytona.common.errors import DaytonaNotFoundError  # type: ignore
        except Exception:
            class DaytonaNotFoundError(Exception):  # type: ignore
                pass

        try:
            snap = _with_rate_limit_backoff(lambda: client.snapshot.get(name))
        except DaytonaNotFoundError:
            return ("MISSING", None)
        except Exception as exc:
            # Some clients raise generic exceptions on 404; sniff the message.
            msg = str(exc).lower()
            if "not found" in msg or "404" in msg:
                return ("MISSING", None)
            raise
        return (_normalized_state(getattr(snap, "state", None)), snap)

    def _wait_for_state(
        self,
        client: Any,
        name: str,
        *,
        target_state: str = "ACTIVE",
        timeout_s: float = 600.0,
        poll_s: float = 5.0,
    ) -> Tuple[str, Optional[Any]]:
        """Poll until the snapshot reaches target_state or timeout."""
        deadline = time.time() + timeout_s
        last_warn = 0.0
        while True:
            state, snap = self._get_state(client, name)
            if state == target_state:
                return (state, snap)
            if state == "ERROR":
                return (state, snap)
            if state == "MISSING":
                return (state, snap)
            # PENDING — keep polling
            now = time.time()
            if now >= deadline:
                print(f"  {name}: WARNING wait timed out after {timeout_s:.0f}s in state {state}")
                return (state, snap)
            if now - last_warn >= 420:  # ~7 min
                print(f"  {name}: still {state} after {int(now - (deadline - timeout_s))}s; "
                      f"consider increasing pending_wait_s")
                last_warn = now
            time.sleep(poll_s)

    # --- write paths -----------------------------------------------------

    def _ensure_one(
        self,
        org: OrgConfig,
        env_hash: str,
        env_dir: Path,
        *,
        target_region: str,
        build_timeout: float,
        pending_wait_s: float,
        registry_cache: Dict[Tuple[str, str], Dict[str, Any]],
        dry_run: bool,
    ) -> str:
        """Ensure exactly one snapshot exists on ``org``. Returns final state."""
        from daytona import CreateSnapshotParams  # type: ignore
        from daytona.common.image import Image  # type: ignore

        client = self._client(org)
        name = _snapshot_name(env_hash, target_region)

        # Registry fast-path: trust *but verify*.
        cached = registry_cache.get((env_hash, org.name))
        if cached and cached.get("state") == "ACTIVE":
            state, snap = self._get_state(client, name)
            if state == "ACTIVE":
                print(f"  [{org.name}] {name}: already ACTIVE (registry hit), skipping")
                return "ACTIVE"
            # Registry was stale; fall through and reconcile.

        state, snap = self._get_state(client, name)

        if state == "ACTIVE":
            print(f"  [{org.name}] {name}: already ACTIVE, skipping")
            self.registry.append({
                "hash": env_hash, "snapshot_name": name, "org": org.name,
                "state": "ACTIVE", "env_dir_path": str(env_dir),
            })
            return "ACTIVE"

        if state == "PENDING":
            print(f"  [{org.name}] {name}: PENDING, waiting up to {pending_wait_s:.0f}s")
            final_state, snap = self._wait_for_state(
                client, name, target_state="ACTIVE",
                timeout_s=pending_wait_s, poll_s=5.0,
            )
            if final_state == "ACTIVE":
                print(f"  [{org.name}] {name}: reached ACTIVE")
                self.registry.append({
                    "hash": env_hash, "snapshot_name": name, "org": org.name,
                    "state": "ACTIVE", "env_dir_path": str(env_dir),
                })
                return "ACTIVE"
            # Fall through: delete + recreate.
            state, snap = (final_state, snap)

        if state == "ERROR" or (state == "PENDING" and snap is not None):
            print(f"  [{org.name}] {name}: state={state}, deleting and rebuilding")
            try:
                _with_rate_limit_backoff(lambda: client.snapshot.delete(snap))
            except Exception as del_err:
                print(f"  [{org.name}] {name}: WARNING failed to delete: {del_err}")

        # Build (or rebuild).
        dockerfile_path = env_dir / "Dockerfile"
        if not dockerfile_path.exists():
            print(f"  [{org.name}] {name}: WARNING no Dockerfile at {dockerfile_path}, skipping")
            return "MISSING"

        if dry_run:
            print(f"  [{org.name}] {name}: [DRY-RUN] would build from {dockerfile_path}")
            return "MISSING"

        print(f"  [{org.name}] {name}: building from {dockerfile_path} ...")
        create_kwargs: Dict[str, Any] = {
            "name": name,
            "image": Image.from_dockerfile(str(dockerfile_path)),
        }
        if target_region:
            create_kwargs["region_id"] = target_region

        try:
            _with_rate_limit_backoff(lambda: client.snapshot.create(
                CreateSnapshotParams(**create_kwargs),
                on_logs=print,
                timeout=build_timeout,
            ))
        except Exception as exc:
            msg = str(exc).lower()
            # Concurrent-create idempotency: another launch beat us to it.
            if "already exists" in msg or "conflict" in msg or "duplicate" in msg:
                print(f"  [{org.name}] {name}: already exists (concurrent create), continuing")
            else:
                raise

        # Confirm final state.
        final_state, _ = self._wait_for_state(
            client, name, target_state="ACTIVE",
            timeout_s=pending_wait_s, poll_s=5.0,
        )
        self.registry.append({
            "hash": env_hash, "snapshot_name": name, "org": org.name,
            "state": final_state, "env_dir_path": str(env_dir),
        })
        if final_state != "ACTIVE":
            raise SnapshotBuildFailed(
                f"[{org.name}] {name}: never reached ACTIVE (final={final_state})"
            )
        print(f"  [{org.name}] {name}: built successfully")
        return "ACTIVE"

    # --- list / status ---------------------------------------------------

    def list_all(self, *, page_limit: int = 100) -> Dict[str, List[SnapshotInfo]]:
        out: Dict[str, List[SnapshotInfo]] = {}
        for org in self.orgs:
            client = self._client(org)
            items: List[SnapshotInfo] = []
            page = 1
            while True:
                try:
                    resp = _with_rate_limit_backoff(
                        lambda: client.snapshot.list(page=page, limit=page_limit)
                    )
                except TypeError:
                    # Some clients don't accept ``page``; fall back to a single call.
                    resp = _with_rate_limit_backoff(lambda: client.snapshot.list(limit=page_limit))
                snaps = getattr(resp, "items", None) or getattr(resp, "data", None) or []
                for snap in snaps:
                    name = getattr(snap, "name", "") or ""
                    state = _normalized_state(getattr(snap, "state", None))
                    # Decode hash from name: harbor__{hash}__... -> {hash}
                    h = ""
                    if name.startswith("harbor__"):
                        parts = name.split("__")
                        if len(parts) >= 3:
                            h = parts[1]
                    items.append(SnapshotInfo(name=name, hash=h, org=org.name, state=state))
                # Pagination heuristic
                total_pages = getattr(resp, "total_pages", None)
                if total_pages is None or page >= total_pages:
                    break
                page += 1
            out[org.name] = items
        return out


# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def _discover_hash_to_env_dir(resolved_data_paths: List[str]) -> Tuple[Dict[str, Path], Any]:
    """Run task discovery + analysis; returns (hash_to_env_dir, stats).

    Task enumeration uses harbor's authoritative ``get_task_configs`` path (via
    ``count_snapshots_from_tasks.load_tasks_from_local_dataset``) — the SAME
    routine the standalone snapshot-count tool uses and that the Harbor job
    itself uses at runtime to decide which tasks to run/snapshot. We deliberately
    do NOT use the naive ``discover_task_dirs`` filesystem walk here: it keys on a
    literal ``instruction.md`` and silently drops tasks with other valid layouts
    (e.g. multi-step tasks), so its count diverges from the real snapshot count.
    """
    try:
        from scripts.harbor.count_snapshots_from_tasks import (
            get_snapshot_env_dirs,
            load_tasks_from_local_dataset,
        )
        from harbor.utils.container_cache import analyze_task_dockerfiles
    except ImportError as exc:
        raise ImportError(
            "harbor/scripts.harbor imports failed; ensure harbor is installed "
            "(pip install -e /path/to/harbor) and the repo root is on PYTHONPATH"
        ) from exc

    task_dirs: List[Path] = []
    for data_path in resolved_data_paths:
        root = Path(data_path)
        if not root.is_dir():
            continue
        task_dirs.extend(load_tasks_from_local_dataset(dataset_path=root.expanduser().resolve()))
    if not task_dirs:
        return {}, None
    stats = analyze_task_dockerfiles(task_dirs)
    hash_to_env_dir = get_snapshot_env_dirs(task_dirs)
    return hash_to_env_dir, stats


def ensure_snapshots(
    resolved_data_paths: List[str],
    orgs: List[OrgConfig],
    *,
    max_new_snapshots: int = 10,
    max_org_snapshots: int = 60,
    target_region: str = "",
    build_timeout: float = 600.0,
    pending_wait_s: float = 600.0,
    registry_path: Optional[Path] = None,
    dry_run: bool = False,
    daytona_factory: Callable[[OrgConfig], Any] = _default_daytona_factory,
) -> SnapshotPlanResult:
    """Discover task envs, then ensure snapshots exist on every configured org.

    Behavior:
      - ACTIVE   -> skip
      - PENDING  -> wait up to ``pending_wait_s``
      - ERROR    -> delete + rebuild
      - MISSING  -> build
    Hard-fails (SnapshotCapExceeded) if any org's existing + needed exceeds
    ``max_org_snapshots``.

    Returns a SnapshotPlanResult with per-org StatusCounts and aggregate
    built/skipped counters.
    """
    print(f"\n=== Pre-building Daytona snapshots ({len(orgs)} org(s)) ===")
    hash_to_env_dir, stats = _discover_hash_to_env_dir(resolved_data_paths)
    if not hash_to_env_dir:
        print("No task directories or environments found; skipping.")
        return SnapshotPlanResult()
    print(f"Found {getattr(stats, 'total_tasks', '?')} task(s) with "
          f"{len(hash_to_env_dir)} unique environment(s)")

    if len(hash_to_env_dir) > max_new_snapshots:
        raise SnapshotCapExceeded(
            f"Dataset requires {len(hash_to_env_dir)} unique snapshots, "
            f"exceeding the safety limit of max_new_snapshots={max_new_snapshots}. "
            f"Pass max_new_snapshots=N to ensure_snapshots if intentional."
        )

    manager = _SnapshotManager(orgs, registry_path=registry_path, daytona_factory=daytona_factory)
    registry_cache = manager.registry.load()
    result = SnapshotPlanResult()

    # Cap check per org BEFORE any build (preserves prior hard-fail invariant).
    for org in orgs:
        client = manager._client(org)
        try:
            resp = _with_rate_limit_backoff(lambda: client.snapshot.list(limit=1))
            total_existing = int(getattr(resp, "total", 0))
        except Exception as exc:
            print(f"  [{org.name}] WARNING could not read org capacity: {exc}")
            total_existing = 0
        new_needed = len(hash_to_env_dir)
        if total_existing + new_needed > max_org_snapshots:
            raise SnapshotCapExceeded(
                f"Org '{org.name}' has {total_existing} snapshots; adding {new_needed} "
                f"would exceed the cap of {max_org_snapshots}. "
                "Delete unused snapshots (see cleanup_unused_snapshots) first."
            )

    # Build / ensure per (org, hash).
    for org in orgs:
        counts = StatusCounts()
        for env_hash, env_dir in hash_to_env_dir.items():
            try:
                final = manager._ensure_one(
                    org, env_hash, env_dir,
                    target_region=target_region,
                    build_timeout=build_timeout,
                    pending_wait_s=pending_wait_s,
                    registry_cache=registry_cache,
                    dry_run=dry_run,
                )
            except SnapshotBuildFailed as exc:
                result.errors.append(str(exc))
                counts.error += 1
                continue
            if final == "ACTIVE":
                # ensure_one logs "already ACTIVE" vs "built successfully";
                # we just bin by current state for the report.
                counts.active += 1
                if registry_cache.get((env_hash, org.name), {}).get("state") == "ACTIVE":
                    result.skipped += 1
                else:
                    result.built += 1
            elif final == "PENDING":
                counts.pending += 1
            elif final == "ERROR":
                counts.error += 1
            else:
                counts.missing += 1
        result.per_org[org.name] = counts

    print(f"\nSnapshot pre-build complete: {result.built} built, {result.skipped} already existed, "
          f"{len(result.errors)} error(s)")
    return result


def cleanup_unused_snapshots(
    needed_hashes: Set[str],
    orgs: List[OrgConfig],
    *,
    registry_path: Optional[Path] = None,
    dry_run: bool = False,
    daytona_factory: Callable[[OrgConfig], Any] = _default_daytona_factory,
) -> CleanupResult:
    """Delete any ``harbor__<hash>__...`` snapshot whose hash is not in needed_hashes.

    NEVER called automatically by ``ensure_snapshots``; manual use only.
    """
    print(f"\n=== Cleaning up unused Daytona snapshots ({len(orgs)} org(s)) ===")
    manager = _SnapshotManager(orgs, registry_path=registry_path, daytona_factory=daytona_factory)
    listing = manager.list_all()
    result = CleanupResult()
    for org_name, snaps in listing.items():
        deleted = 0
        for snap in snaps:
            if not snap.name.startswith("harbor__"):
                continue
            if snap.hash in needed_hashes:
                continue
            org = next(o for o in orgs if o.name == org_name)
            client = manager._client(org)
            if dry_run:
                print(f"  [{org_name}] {snap.name}: [DRY-RUN] would delete")
                deleted += 1
                continue
            try:
                # Re-fetch + delete (the listing payload may not be deleteable directly)
                obj = _with_rate_limit_backoff(lambda: client.snapshot.get(snap.name))
                _with_rate_limit_backoff(lambda: client.snapshot.delete(obj))
                print(f"  [{org_name}] {snap.name}: deleted")
                deleted += 1
            except Exception as exc:
                print(f"  [{org_name}] {snap.name}: WARNING delete failed: {exc}")
        result.per_org[org_name] = deleted
    print("\nCleanup complete:")
    for org_name, n in result.per_org.items():
        print(f"  [{org_name}]: {n} deleted")
    return result


def list_snapshots(
    orgs: List[OrgConfig],
    *,
    daytona_factory: Callable[[OrgConfig], Any] = _default_daytona_factory,
) -> Dict[str, List[SnapshotInfo]]:
    """Enumerate every snapshot on every configured org."""
    manager = _SnapshotManager(orgs, daytona_factory=daytona_factory)
    return manager.list_all()


def status_snapshots(
    resolved_data_paths: List[str],
    orgs: List[OrgConfig],
    *,
    target_region: str = "",
    daytona_factory: Callable[[OrgConfig], Any] = _default_daytona_factory,
) -> Dict[str, StatusCounts]:
    """For the needed hashes derived from ``resolved_data_paths``, report
    per-org ``StatusCounts``. Read-only; never creates snapshots."""
    hash_to_env_dir, _ = _discover_hash_to_env_dir(resolved_data_paths)
    needed = set(hash_to_env_dir.keys())

    manager = _SnapshotManager(orgs, daytona_factory=daytona_factory)
    out: Dict[str, StatusCounts] = {}
    for org in orgs:
        client = manager._client(org)
        counts = StatusCounts()
        for env_hash in needed:
            name = _snapshot_name(env_hash, target_region)
            state, _ = manager._get_state(client, name)
            if state == "ACTIVE":
                counts.active += 1
            elif state == "PENDING":
                counts.pending += 1
            elif state == "ERROR":
                counts.error += 1
            else:
                counts.missing += 1
        out[org.name] = counts
    return out


def load_orgs_from_env(names: List[str]) -> List[OrgConfig]:
    """Discover OrgConfig objects from ``DAYTONA_<NAME>_API_KEY`` env vars.

    Special-case ``"default"`` reads the bare ``DAYTONA_API_KEY``. Skips
    names whose env var is unset. Raises ValueError if zero orgs resolve.
    """
    orgs: List[OrgConfig] = []
    for name in names:
        if name == "default":
            env_var = "DAYTONA_API_KEY"
        else:
            env_var = f"DAYTONA_{name.upper()}_API_KEY"
        api_key = os.environ.get(env_var, "")
        if not api_key:
            continue
        api_url = os.environ.get("DAYTONA_API_URL") or None
        orgs.append(OrgConfig(name=name, api_key=api_key, api_url=api_url))
    if not orgs:
        raise ValueError(
            f"No Daytona keys found for orgs={names}; "
            f"expected env vars like DAYTONA_API_KEY or DAYTONA_<NAME>_API_KEY"
        )
    return orgs


# -----------------------------------------------------------------------------
# Optional CLI
# -----------------------------------------------------------------------------

def _parse_org_arg(spec: str) -> OrgConfig:
    """Parse a ``NAME=KEY`` org spec (used by the CLI)."""
    if "=" not in spec:
        raise ValueError(f"invalid --org spec {spec!r}; expected NAME=API_KEY")
    name, _, api_key = spec.partition("=")
    name = name.strip()
    api_key = api_key.strip()
    if not name or not api_key:
        raise ValueError(f"invalid --org spec {spec!r}; expected NAME=API_KEY")
    return OrgConfig(name=name, api_key=api_key)


def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m hpc.snapshot_manager")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_org = lambda sp: sp.add_argument(
        "--org", action="append", required=True,
        help="Daytona org spec NAME=API_KEY (can be repeated for multi-org)",
    )

    ens = sub.add_parser("ensure", help="Pre-build snapshots for a tasks dir")
    ens.add_argument("--tasks", required=True, help="Local tasks directory")
    common_org(ens)
    ens.add_argument("--target-region", default="")
    ens.add_argument("--max-new-snapshots", type=int, default=10)
    ens.add_argument("--max-org-snapshots", type=int, default=60)
    ens.add_argument("--pending-wait-s", type=float, default=600.0)
    ens.add_argument("--build-timeout", type=float, default=600.0)
    ens.add_argument("--dry-run", action="store_true")

    cln = sub.add_parser("cleanup", help="Delete snapshots not needed by --tasks")
    cln.add_argument("--tasks", required=True, help="Local tasks directory (defines needed set)")
    common_org(cln)
    cln.add_argument("--dry-run", action="store_true")

    lst = sub.add_parser("list", help="List all snapshots on each org")
    common_org(lst)

    sts = sub.add_parser("status", help="Per-needed-hash state report")
    sts.add_argument("--tasks", required=True)
    common_org(sts)
    sts.add_argument("--target-region", default="")

    args = p.parse_args()
    orgs = [_parse_org_arg(s) for s in args.org]

    if args.cmd == "ensure":
        r = ensure_snapshots(
            [args.tasks], orgs,
            max_new_snapshots=args.max_new_snapshots,
            max_org_snapshots=args.max_org_snapshots,
            target_region=args.target_region,
            build_timeout=args.build_timeout,
            pending_wait_s=args.pending_wait_s,
            dry_run=args.dry_run,
        )
        print(json.dumps(asdict(r), default=str, indent=2))
        return 0

    if args.cmd == "cleanup":
        hash_to_env_dir, _ = _discover_hash_to_env_dir([args.tasks])
        r = cleanup_unused_snapshots(set(hash_to_env_dir.keys()), orgs, dry_run=args.dry_run)
        print(json.dumps(asdict(r), default=str, indent=2))
        return 0

    if args.cmd == "list":
        r = list_snapshots(orgs)
        for org_name, items in r.items():
            print(f"\n=== {org_name} ({len(items)} snapshot(s)) ===")
            for s in items:
                print(f"  {s.state:8s} {s.name}")
        return 0

    if args.cmd == "status":
        r = status_snapshots([args.tasks], orgs, target_region=args.target_region)
        for org_name, counts in r.items():
            print(f"[{org_name}] active={counts.active} pending={counts.pending} "
                  f"error={counts.error} missing={counts.missing}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(_cli())
