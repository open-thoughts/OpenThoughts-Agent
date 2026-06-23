"""Local fetch daemon for OT-Agent iris jobs.

Polls the iris controller on a loop; for each registered job in a
terminal state, ``gcloud storage rsync``-es its GCS output prefix down
to ``~/.ot-agent/runs/<job-name>/`` and stamps the registry row.

The launcher (``hpc.iris_launch_utils.IrisLauncher.run``) writes one row
per submission via ``hpc.iris_job_registry.register_submission`` *before*
the daemon ever sees the job. That decoupling means the daemon can be
offline at submit time (or uninstalled entirely) without losing state —
a future ``run`` cycle picks it up, or the user runs
``python -m hpc.iris_fetch_daemon fetch <job-id>`` directly.

CLI::

    python -m hpc.iris_fetch_daemon run [--once] [--interval 60]
    python -m hpc.iris_fetch_daemon status
    python -m hpc.iris_fetch_daemon fetch <job-id>
    python -m hpc.iris_fetch_daemon install [--interval 60]
    python -m hpc.iris_fetch_daemon uninstall

Heartbeat: each poll cycle writes the current UTC timestamp to
``~/.ot-agent/state/daemon.heartbeat``. ``status`` shows how stale it is.

Hang watchdog: each poll, every actively-RUNNING job is checked for the
severed-Daytona dead-hang (RUNNING but harbor's result.json ``updated_at``
frozen past a threshold despite having served trials). Hung jobs are
``iris job kill``-ed and marked FAILED. Tuned via ``OT_AGENT_WATCHDOG``
(kill | log_only | off), ``OT_AGENT_WATCHDOG_HANG_SECONDS`` (default 7200),
``OT_AGENT_WATCHDOG_PREFIX`` (default empty = all jobs).

Design doc: ``notes/marin/flows/iris-outputs-redesign.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from hpc.iris_job_registry import (
    DB_PATH,
    JobRecord,
    STATUS_FAILED,
    STATUS_FETCH_FAILED,
    STATUS_FETCHED,
    STATUS_FETCHING,
    STATUS_RUNNING,
    STATUS_SUBMITTED,
    STATUS_SUCCEEDED,
    get,
    list_all,
    list_pending,
    update_status,
)
from hpc.local_paths import PATHS, ensure as ensure_local_paths


# ---------------------------------------------------------------------
# Tunables / external dependencies
# ---------------------------------------------------------------------

LABEL = "io.openthoughts.ot-agent-fetch"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"

HEARTBEAT_PATH = PATHS.state / "daemon.heartbeat"

# The `iris` CLI binary to shell to. Override via OT_AGENT_IRIS_CLI when
# the marin checkout isn't at the default location.
IRIS_CLI = os.environ.get(
    "OT_AGENT_IRIS_CLI",
    str(Path.home() / "Documents" / "marin" / ".venv" / "bin" / "iris"),
)

# `gcloud` binary; override via OT_AGENT_GCLOUD_CLI.
GCLOUD_CLI = os.environ.get(
    "OT_AGENT_GCLOUD_CLI",
    shutil.which("gcloud") or "/usr/local/bin/gcloud",
)

# Mapping from iris JOB_STATE_* enum string → coarse status the daemon cares about.
_IRIS_TERMINAL_SUCCESS = {"JOB_STATE_SUCCEEDED"}
_IRIS_TERMINAL_FAILURE = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_KILLED",
                          "JOB_STATE_TIMEOUT", "JOB_STATE_PREEMPTED"}
_IRIS_RUNNING = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SCHEDULED",
                 "JOB_STATE_QUEUED", "JOB_STATE_SUBMITTED", "JOB_STATE_ASSIGNED"}

# ---------------------------------------------------------------------
# Hang watchdog
# ---------------------------------------------------------------------
# Iris jobs occasionally dead-hang after a preempt-restart: iris keeps the
# job RUNNING, but harbor's connection to Daytona is silently severed and it
# stops making progress indefinitely (see memory
# `iris-datagen-severed-daytona-hang`). harbor stamps a wall-clock
# `updated_at` into its GCS result.json on every status flush; when the job
# hangs, that timestamp freezes. The watchdog kills a RUNNING job whose
# result.json shows it has served (`n_completed_trials > 0`) but whose
# `updated_at` has been stale longer than the threshold.
#
# The `n_completed_trials > 0` guard means a job still in cold compile (no
# result.json, or zero completed trials) is NEVER auto-killed — only jobs
# that were demonstrably alive and then went silent. Cold-compile stalls are
# the engine healthcheck's job, not the watchdog's.
WATCHDOG_MODE = os.environ.get("OT_AGENT_WATCHDOG", "kill")  # kill | log_only | off
WATCHDOG_HANG_SECONDS = int(os.environ.get("OT_AGENT_WATCHDOG_HANG_SECONDS", "7200"))
# Empty = police every registered RUNNING job; else only job_names with this prefix.
WATCHDOG_PREFIX = os.environ.get("OT_AGENT_WATCHDOG_PREFIX", "")

# --- iris local-tunnel port reservation -------------------------------------
# iris's SSH controller tunnel picks a LOCAL forward port via
# iris.cluster.providers.types.find_free_port(start=10000): it scans 10000
# upward and SKIPS any port whose /tmp/iris/port_<N> lockfile names a LIVE pid
# (os.kill(pid, 0) succeeds) WITHOUT binding the socket. The daemon shells out
# to `iris job logs`, so without intervention its tunnel grabs 10000 and holds
# it 24/7 — colliding with other apps that need 10000 (e.g. step-ca's OIDC
# listener during `step ssh certificate`, which binds 127.0.0.1:10000).
#
# Fix (no marin/iris source edit): at startup the daemon writes its own live pid
# into the lockfiles for RESERVED_LOCAL_PORTS, so every iris find_free_port scan
# on this host (the daemon's own `iris job logs` AND the monitor cron's
# `iris query`/`iris job logs`) skips them and lands on 10001+. The lockfile is
# advisory only — it does NOT bind the socket, so step-ca (or anything else) can
# still use the reserved port normally. Re-asserted each poll (self-healing).
IRIS_PORT_LOCK_DIR = Path(os.environ.get("IRIS_PORT_LOCK_DIR", "/tmp/iris"))
RESERVED_LOCAL_PORTS = [
    int(p) for p in os.environ.get("OT_AGENT_RESERVED_LOCAL_PORTS", "10000").split(",") if p.strip()
]


def reserve_iris_local_ports() -> None:
    """Steer iris's local-tunnel port scan away from RESERVED_LOCAL_PORTS.

    Writes ``/tmp/iris/port_<N> = os.getpid()`` (this daemon's live pid) for each
    reserved port. iris.find_free_port treats a lockfile naming a live pid as
    in-use and skips it, so iris tunnels on this host avoid the reserved ports
    and leave them free for other apps. Does NOT bind the socket. Best-effort.
    """
    try:
        IRIS_PORT_LOCK_DIR.mkdir(parents=True, exist_ok=True)
        for port in RESERVED_LOCAL_PORTS:
            (IRIS_PORT_LOCK_DIR / f"port_{port}").write_text(str(os.getpid()))
    except OSError as e:
        _log(f"could not reserve iris local ports {RESERVED_LOCAL_PORTS}: {e}", err=True)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str, *, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    print(f"[daemon {_now_iso()}] {msg}", file=stream, flush=True)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for dp, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                continue
    return total


def _strip_log_prefix(stdout: str) -> str:
    """Return the JSON-looking suffix of an iris CLI dump.

    The iris CLI prints I-level log lines to stdout before the JSON
    payload, which trips a vanilla ``json.loads``. We slice from the
    first ``[`` or ``{`` to the end of the buffer.
    """
    for i, ch in enumerate(stdout):
        if ch in "[{":
            return stdout[i:]
    return stdout


# ---------------------------------------------------------------------
# Iris controller interaction
# ---------------------------------------------------------------------

def _iris_job_list(cluster_config: str, user_prefix: str, *, timeout: int = 120) -> list[dict]:
    """Shell to ``iris --config <c> job list --prefix <p> --json``.

    Returns the parsed list, or [] on any subprocess / parse error
    (logged to stderr; daemon keeps polling on the next cycle).
    """
    cmd = [
        IRIS_CLI, "--config", cluster_config,
        "job", "list", "--prefix", user_prefix, "--json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        _log(f"iris list failed for {user_prefix}: {e}", err=True)
        return []
    if result.returncode != 0:
        _log(
            f"iris list exit={result.returncode} prefix={user_prefix} "
            f"stderr={result.stderr.strip()[:300]}",
            err=True,
        )
        return []
    try:
        return json.loads(_strip_log_prefix(result.stdout))
    except json.JSONDecodeError as e:
        _log(f"iris list output not JSON: {e}; first 200 chars: "
             f"{result.stdout[:200]!r}", err=True)
        return []


def _user_prefix_of(job_id: str) -> str:
    """Return ``/<user>/`` for an iris job id like ``/benjaminfeuer/foo-1``."""
    parts = job_id.lstrip("/").split("/", 1)
    return f"/{parts[0]}/" if parts else "/"


# ---------------------------------------------------------------------
# Fetch implementation
# ---------------------------------------------------------------------

# Filename for iris's centralized stdout/stderr capture, written into
# the same per-job dir as the GCS-fetched harbor artifacts. Picked a
# dotted prefix so a `ls` of the run dir leads with harbor's files.
IRIS_LOG_FILENAME = ".iris-job.log"


def _dump_iris_logs(record: JobRecord, local_dest: Path) -> Optional[int]:
    """Save iris's centralized stdout/stderr for the job to ``.iris-job.log``.

    Runs *regardless* of whether the GCS fetch succeeded — iris's
    finelog capture is the most reliable artifact (it survives worker
    eviction, container teardown, and any workload-side filesystem
    bugs). Returns the byte count on success, or None on any error
    (logged but never raised — the daemon's primary job is the GCS
    fetch, this is a best-effort companion step).
    """
    log_path = local_dest / IRIS_LOG_FILENAME
    cmd = [
        IRIS_CLI, "--config", record.cluster_config,
        "job", "logs", record.job_id, "--max-lines", "100000",
    ]
    try:
        with log_path.open("w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE,
                                    text=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        _log(f"iris-log dump failed for {record.job_id}: {e}", err=True)
        return None
    if result.returncode != 0:
        _log(
            f"iris-log dump rc={result.returncode} for {record.job_id}: "
            f"{(result.stderr or '').strip()[:200]}",
            err=True,
        )
        # Non-zero return; keep whatever bytes did stream through.
    try:
        size = log_path.stat().st_size
    except OSError:
        return None
    _log(f"iris-log saved {size} bytes -> {log_path}")
    return size


def fetch_record(record: JobRecord) -> bool:
    """Run ``gcloud storage rsync -r <gcs>/ <local>/`` for a single job.

    Always also captures iris's stdout/stderr via ``iris job logs`` into
    ``<local_dest>/.iris-job.log`` (best-effort, regardless of fetch
    outcome — iris's centralized capture is the most durable artifact
    we have).

    On success: marks status=fetched, stamps fetched_at + bytes_fetched.
    On failure: marks status=fetch_failed with error_msg.

    Idempotent — ``gcloud storage rsync`` only copies files whose
    sizes/mtimes differ. Safe to retry.
    """
    local_dest = Path(record.local_dest)
    ensure_local_paths(PATHS.runs)
    local_dest.mkdir(parents=True, exist_ok=True)

    # Iris log dump first so the file exists even if the GCS rsync hangs
    # or errors. _dump_iris_logs handles its own errors.
    _dump_iris_logs(record, local_dest)

    gcs_src = record.gcs_output_dir.rstrip("/") + "/"
    cmd = [GCLOUD_CLI, "storage", "rsync", "-r", gcs_src, str(local_dest)]

    _log(f"fetching {record.job_id}: {gcs_src} -> {local_dest}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        update_status(record.job_id, status=STATUS_FETCH_FAILED,
                      error_msg=f"gcloud rsync timed out / not found: {e}")
        _log(f"fetch failed for {record.job_id}: {e}", err=True)
        return False

    if result.returncode != 0:
        update_status(
            record.job_id, status=STATUS_FETCH_FAILED,
            error_msg=f"rc={result.returncode} stderr={result.stderr.strip()[:500]}",
        )
        _log(
            f"fetch failed for {record.job_id}: rc={result.returncode} "
            f"{result.stderr.strip()[:300]}",
            err=True,
        )
        return False

    size = _dir_size_bytes(local_dest)
    update_status(
        record.job_id, status=STATUS_FETCHED,
        fetched_at_iso=_now_iso(),
        bytes_fetched=size,
    )
    _log(f"fetched {record.job_id}: {size} bytes at {local_dest}")
    return True


# ---------------------------------------------------------------------
# Hang watchdog implementation
# ---------------------------------------------------------------------

def _parse_harbor_iso(ts: str) -> Optional[datetime]:
    """Parse harbor's result.json ``updated_at`` (ISO-8601 UTC, trailing Z)."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _is_hung(updated_at: Optional[datetime], n_completed: Optional[int],
             now: datetime, threshold_seconds: int) -> bool:
    """Pure hang predicate (no I/O — unit-tested directly).

    Hung iff harbor has served at least one trial (``n_completed > 0``) and
    its last status flush (``updated_at``) is older than ``threshold_seconds``.
    A job with no result.json / no completed trials (falsy ``updated_at`` or
    ``n_completed``) is never hung — that's a cold compile, owned by the
    engine healthcheck, not the watchdog.
    """
    if updated_at is None or not n_completed or n_completed <= 0:
        return False
    return (now - updated_at).total_seconds() > threshold_seconds


def _harbor_liveness(record: JobRecord) -> tuple[Optional[datetime], Optional[int]]:
    """Read ``(updated_at, n_completed_trials)`` from the job's result.json.

    result.json lives at ``<gcs_output_dir>/<job_name>/result.json`` and is
    rewritten by harbor on every status flush. Retries briefly on a
    truncated/racing read; returns ``(None, None)`` on any persistent failure
    (absent file, unreadable, parse error) so the caller treats the job as
    "can't tell → don't kill".
    """
    path = f"{record.gcs_output_dir.rstrip('/')}/{record.job_name}/result.json"
    for attempt in range(3):
        try:
            proc = subprocess.run(
                [GCLOUD_CLI, "storage", "cat", path],
                capture_output=True, text=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            _log(f"watchdog: result.json read failed for {record.job_id}: {e}", err=True)
            return None, None
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                data = None
            if data is not None:
                updated_at = _parse_harbor_iso(data.get("updated_at") or "")
                n_completed = data.get("stats", {}).get("n_completed_trials")
                return updated_at, n_completed
        if attempt < 2:
            time.sleep(1)
    # result.json absent (cold compile / never served) or persistently unreadable.
    return None, None


def _watchdog_kill(record: JobRecord) -> bool:
    """``iris job kill`` the hung job. Returns True on success."""
    cmd = [IRIS_CLI, "--config", record.cluster_config, "job", "kill", record.job_id]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        _log(f"watchdog: kill subprocess failed for {record.job_id}: {e}", err=True)
        return False
    if proc.returncode != 0:
        _log(f"watchdog: kill rc={proc.returncode} for {record.job_id}: "
             f"{(proc.stderr or '').strip()[:200]}", err=True)
        return False
    return True


def watchdog_check(record: JobRecord, now: datetime) -> bool:
    """Detect + act on a hung RUNNING job.

    Returns True iff the job was auto-killed (caller should stop treating it
    as RUNNING — the registry row is already marked FAILED). Honors
    ``OT_AGENT_WATCHDOG`` (kill | log_only | off) and ``OT_AGENT_WATCHDOG_PREFIX``.
    """
    if WATCHDOG_MODE == "off":
        return False
    if WATCHDOG_PREFIX and not record.job_name.startswith(WATCHDOG_PREFIX):
        return False

    updated_at, n_completed = _harbor_liveness(record)
    if not _is_hung(updated_at, n_completed, now, WATCHDOG_HANG_SECONDS):
        return False

    stale_h = (now - updated_at).total_seconds() / 3600.0
    detail = (f"harbor result.json stale {stale_h:.1f}h "
              f"(threshold {WATCHDOG_HANG_SECONDS / 3600:.1f}h), n_completed={n_completed}")

    if WATCHDOG_MODE == "log_only":
        _log(f"watchdog: WOULD KILL {record.job_id} — {detail}")
        return False

    _log(f"watchdog: KILL {record.job_id} — {detail}")
    if not _watchdog_kill(record):
        return False  # kill failed; leave RUNNING, retry next cycle
    update_status(
        record.job_id, status=STATUS_FAILED,
        last_polled_at_iso=_now_iso(),
        error_msg=f"watchdog: hung {stale_h:.1f}h, auto-killed",
    )
    _log(f"watchdog: killed {record.job_id}, marked FAILED")
    return True


# ---------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------

def poll_once() -> None:
    """One full reconciliation pass over the registry."""
    ensure_local_paths(PATHS.home, PATHS.state, PATHS.logs)
    HEARTBEAT_PATH.write_text(_now_iso())

    pending = list_pending()
    if not pending:
        return

    # Group by (cluster_config, user_prefix) so we do one iris-CLI call
    # per cluster rather than one per registered job. The user_prefix
    # narrows the controller-side scan; cluster_config picks the iris
    # endpoint.
    groups: dict[tuple[str, str], list[JobRecord]] = {}
    for r in pending:
        groups.setdefault((r.cluster_config, _user_prefix_of(r.job_id)), []).append(r)

    for (cluster_cfg, user_prefix), records in groups.items():
        jobs = _iris_job_list(cluster_cfg, user_prefix)
        if not jobs:
            # Either iris is unreachable or the user has no jobs under
            # this prefix; either way, retry next cycle. We still stamp
            # last_polled_at so status shows progress.
            for r in records:
                update_status(r.job_id, status=r.status, last_polled_at_iso=_now_iso())
            continue
        by_id = {j.get("job_id"): j for j in jobs}

        for r in records:
            j = by_id.get(r.job_id)
            if j is None:
                # Job submitted by this user but not in the recent list
                # window — could be filtered out by iris's default limit.
                # Mark polled and try again next cycle.
                update_status(r.job_id, status=r.status, last_polled_at_iso=_now_iso())
                continue

            state = j.get("state", "")
            exit_code = j.get("exit_code")
            preemption_count = j.get("preemption_count")
            now_iso = _now_iso()

            if state in _IRIS_RUNNING:
                # Hang watchdog: only actively-RUNNING jobs can be hung
                # (PENDING/QUEUED have no harbor result.json yet). If it
                # auto-kills, the row is already marked FAILED — skip the
                # RUNNING re-stamp so the job flows to fetch next cycle.
                if state == "JOB_STATE_RUNNING" and watchdog_check(
                    r, datetime.now(timezone.utc)
                ):
                    continue
                update_status(
                    r.job_id, status=STATUS_RUNNING,
                    last_polled_at_iso=now_iso,
                    iris_attempt_id=preemption_count,
                )
                continue

            terminal = state in _IRIS_TERMINAL_SUCCESS or state in _IRIS_TERMINAL_FAILURE
            if not terminal:
                # Unknown state — keep polling, don't lose the row.
                update_status(r.job_id, status=r.status, last_polled_at_iso=now_iso,
                              error_msg=f"unhandled iris state {state}")
                continue

            # Job is terminal. Don't re-fetch if already done.
            if r.status in (STATUS_FETCHED, STATUS_FETCH_FAILED, STATUS_FETCHING):
                continue

            terminal_status = (
                STATUS_SUCCEEDED if state in _IRIS_TERMINAL_SUCCESS else STATUS_FAILED
            )
            update_status(
                r.job_id, status=terminal_status,
                last_polled_at_iso=now_iso,
                exit_code=exit_code,
                iris_attempt_id=preemption_count,
            )
            update_status(r.job_id, status=STATUS_FETCHING)
            # Refresh the record for fetch_record (status / local_dest unchanged
            # but explicit).
            refreshed = get(r.job_id) or r
            fetch_record(refreshed)


def run_loop(interval: int, once: bool) -> int:
    """Main daemon entry. ``--once`` returns after a single pass."""
    ensure_local_paths(PATHS.home, PATHS.state, PATHS.logs)
    reserve_iris_local_ports()
    _log(f"started interval={interval}s once={once} db={DB_PATH} reserved_ports={RESERVED_LOCAL_PORTS}")

    stop_flag = {"set": False}

    def _on_signal(signum, _frame):
        stop_flag["set"] = True
        _log(f"caught signal {signum}, exiting after current poll")

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    while not stop_flag["set"]:
        reserve_iris_local_ports()  # self-heal: re-assert before each poll's iris calls
        try:
            poll_once()
        except Exception as e:
            _log(f"poll error: {type(e).__name__}: {e}", err=True)

        if once:
            return 0

        # Sleep in 1-second chunks so SIGTERM is responsive without
        # needing signal.pthread_sigmask gymnastics.
        for _ in range(interval):
            if stop_flag["set"]:
                break
            time.sleep(1)

    return 0


# ---------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    return run_loop(args.interval, args.once)


def _cmd_status(args: argparse.Namespace) -> int:
    """Show daemon liveness + recent jobs."""
    print(f"DB:        {DB_PATH}")
    print(f"Heartbeat: {HEARTBEAT_PATH}")

    if HEARTBEAT_PATH.exists():
        try:
            ts = HEARTBEAT_PATH.read_text().strip()
            age = time.time() - HEARTBEAT_PATH.stat().st_mtime
            health = "ALIVE" if age < 180 else f"STALE ({int(age)}s ago)"
            print(f"           {ts}  [{health}]")
        except OSError as e:
            print(f"           (read failed: {e})")
    else:
        print("           (no heartbeat — daemon has not run)")

    if PLIST_PATH.exists():
        print(f"Plist:     installed at {PLIST_PATH}")
    else:
        print("Plist:     NOT installed (use `install` to add launchd agent)")

    rows = list_all(limit=args.limit)
    if not rows:
        print("\nNo registered jobs.")
        return 0

    print(f"\nLast {len(rows)} job(s):")
    header = f"{'STATUS':<14} {'POLLED':<20} {'EXIT':>4}  {'BYTES':>10}  JOB"
    print(header)
    print("-" * len(header))
    for r in rows:
        polled = r.last_polled_at or "-"
        if polled and "T" in polled:
            polled = polled.split(".")[0]  # trim microseconds
        bytes_str = "-" if r.bytes_fetched is None else f"{r.bytes_fetched:>10}"
        exit_str = "-" if r.exit_code is None else f"{r.exit_code:>4}"
        print(f"{r.status:<14} {polled:<20} {exit_str}  {bytes_str:>10}  {r.job_id}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Manually fetch one job — bypass the poll loop."""
    record = get(args.job_id)
    if record is None:
        _log(f"job not in registry: {args.job_id}", err=True)
        _log("Submit via the launcher first, or re-register manually with "
             "hpc.iris_job_registry.register_submission()", err=True)
        return 1

    update_status(record.job_id, status=STATUS_FETCHING)
    refreshed = get(record.job_id) or record
    ok = fetch_record(refreshed)
    return 0 if ok else 1


# ---------------------------------------------------------------------
# launchd install / uninstall
# ---------------------------------------------------------------------

def _build_plist(*, interval: int) -> dict:
    """Construct the launchd agent plist dict for the current install.

    Captures the absolute path of the running Python interpreter so the
    daemon doesn't pick up a stale shell PATH at runtime. Also pins the
    OT-Agent repo as WorkingDirectory so ``hpc.*`` imports resolve.
    """
    python_exe = sys.executable
    repo_root = Path(__file__).resolve().parents[1]
    log_out = PATHS.logs / "daemon.out"
    log_err = PATHS.logs / "daemon.err"
    ensure_local_paths(PATHS.logs)

    # PATH inherited at run time excludes shell-rc additions; bake one
    # that finds iris + gcloud + python.
    path_entries = [
        str(Path(python_exe).parent),
        str(Path(IRIS_CLI).parent),
        str(Path(GCLOUD_CLI).parent),
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    path_ordered: list[str] = []
    for p in path_entries:
        if p not in seen:
            path_ordered.append(p)
            seen.add(p)

    return {
        "Label": LABEL,
        "ProgramArguments": [
            python_exe, "-m", "hpc.iris_fetch_daemon",
            "run", "--interval", str(interval),
        ],
        "WorkingDirectory": str(repo_root),
        "KeepAlive": True,
        "RunAtLoad": True,
        "ThrottleInterval": 30,
        "StandardOutPath": str(log_out),
        "StandardErrorPath": str(log_err),
        "EnvironmentVariables": {
            "PATH": ":".join(path_ordered),
            "OT_AGENT_IRIS_CLI": IRIS_CLI,
            "OT_AGENT_GCLOUD_CLI": GCLOUD_CLI,
            # Bake the watchdog config resolved at install time so the
            # launchd agent enforces the same policy as the foreground CLI.
            "OT_AGENT_WATCHDOG": WATCHDOG_MODE,
            "OT_AGENT_WATCHDOG_HANG_SECONDS": str(WATCHDOG_HANG_SECONDS),
            "OT_AGENT_WATCHDOG_PREFIX": WATCHDOG_PREFIX,
        },
    }


def _cmd_install(args: argparse.Namespace) -> int:
    plist = _build_plist(interval=args.interval)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLIST_PATH.open("wb") as f:
        plistlib.dump(plist, f)
    _log(f"wrote plist at {PLIST_PATH}")

    uid = os.getuid()
    target = f"gui/{uid}/{LABEL}"
    # Idempotent: bootout any previous install before bootstrap.
    subprocess.run(["launchctl", "bootout", target], capture_output=True)
    bootstrap = subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(PLIST_PATH)],
        capture_output=True, text=True,
    )
    if bootstrap.returncode != 0:
        _log(f"launchctl bootstrap failed: {bootstrap.stderr.strip()}", err=True)
        return 1
    subprocess.run(["launchctl", "enable", target], capture_output=True)
    kickstart = subprocess.run(
        ["launchctl", "kickstart", "-k", target], capture_output=True, text=True,
    )
    if kickstart.returncode != 0:
        _log(f"launchctl kickstart warning: {kickstart.stderr.strip()}", err=True)

    _log(f"installed daemon as {target}")
    _log(f"logs: {PATHS.logs / 'daemon.out'} / .err")
    _log("Run `python -m hpc.iris_fetch_daemon status` to verify.")
    return 0


def _cmd_uninstall(args: argparse.Namespace) -> int:
    uid = os.getuid()
    target = f"gui/{uid}/{LABEL}"
    bootout = subprocess.run(
        ["launchctl", "bootout", target], capture_output=True, text=True,
    )
    if bootout.returncode != 0 and "No such process" not in (bootout.stderr or ""):
        _log(f"launchctl bootout: {bootout.stderr.strip()}", err=True)

    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
        _log(f"removed {PLIST_PATH}")
    else:
        _log(f"plist not present at {PLIST_PATH}")

    _log("uninstalled")
    return 0


# ---------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m hpc.iris_fetch_daemon",
        description="Local daemon: polls iris, fetches completed jobs from GCS.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run the poll loop in the foreground.")
    pr.add_argument("--interval", type=int, default=60,
                    help="Seconds between polls (default 60).")
    pr.add_argument("--once", action="store_true",
                    help="Run a single pass and exit.")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("status", help="Show heartbeat + recent jobs.")
    ps.add_argument("--limit", type=int, default=10,
                    help="Number of recent jobs to display (default 10).")
    ps.set_defaults(func=_cmd_status)

    pf = sub.add_parser("fetch", help="Manually fetch outputs for one job.")
    pf.add_argument("job_id", help="Iris job id (e.g. /benjaminfeuer/eval-iris-...).")
    pf.set_defaults(func=_cmd_fetch)

    pi = sub.add_parser("install", help="Install the launchd user agent.")
    pi.add_argument("--interval", type=int, default=60,
                    help="Poll interval baked into the plist (default 60s).")
    pi.set_defaults(func=_cmd_install)

    pu = sub.add_parser("uninstall", help="Remove the launchd user agent.")
    pu.set_defaults(func=_cmd_uninstall)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
