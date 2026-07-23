#!/usr/bin/env python3
"""Shared Iris lifecycle polling and command-line state watcher.

Poll the *authoritative job lifecycle state* of an iris job on an interval, emit
a line on every state transition, and exit with a clear terminal verdict the
moment the job leaves RUNNING (succeeded / failed / killed / worker_failed /
unschedulable) **OR disappears from the cluster entirely**.

Polls the authoritative iris job state, not log content: the iris controller
retains terminal job records after the pods are reaped, so even a job that has
fully vanished from k8s still reports its terminal state via ``iris job
summary``.

Authoritative state source
---------------------------
``iris --cluster=<C> job summary <job_id> --json`` (the richest single-job call:
``state`` + ``error`` + ``exit_code`` + per-task states + ``finished_at``). It is
backed by the controller's ``GetJobStatus`` / ``ListTasks`` RPCs and works for
running *and* completed/terminal jobs. If that call fails transiently we fall
back to the lighter ``iris query "SELECT state FROM jobs WHERE job_id=..."``
which returns the numeric state. As a final cross-check (and to catch the
"disappeared from cluster / 0 pods" case explicitly) we can count live pods via
``kubectl``.

iris JobState enum (lib/iris/src/iris/rpc/job.proto):
    0 UNSPECIFIED  1 PENDING  2 BUILDING  3 RUNNING
    4 SUCCEEDED    5 FAILED   6 KILLED    7 WORKER_FAILED   8 UNSCHEDULABLE

Usage
-----
    # one-shot: print the current authoritative state and exit
    python scripts/iris/iris_ops.py /benjaminfeuer/<job> --once

    # watch on a 60s interval until the job leaves RUNNING (then exit)
    python scripts/iris/iris_ops.py /benjaminfeuer/<job> --interval 60

Exit codes: 0 succeeded · 1 failed/killed/worker_failed/unschedulable · 2 the
job is absent from the controller AND has 0 pods (disappeared) · 3 watch error.

Importable: ``get_job_state(job_id, cluster)`` returns a ``JobStateSnapshot``;
``watch(job_id, ...)`` runs the poll loop and returns the terminal snapshot, so a
supervising agent can use this as the watch primitive instead of grepping logs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# The operational helpers deliberately validate identifiers with the current
# Iris implementation from Marin main, not a copy vendored into this repo.
MARIN_MAIN_ROOT = Path(os.environ.get("MARIN_MAIN_ROOT", "/Users/benjaminfeuer/Documents/marin"))
MARIN_IRIS_SOURCE = MARIN_MAIN_ROOT / "lib" / "iris" / "src"
if MARIN_IRIS_SOURCE.exists() and str(MARIN_IRIS_SOURCE) not in sys.path:
    sys.path.insert(0, str(MARIN_IRIS_SOURCE))
try:
    from iris.cluster.types import JobName  # noqa: E402
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Marin Iris is unavailable: install the locked dev dependency or provide "
        f"MARIN_MAIN_ROOT (looked for {MARIN_IRIS_SOURCE})."
    ) from exc

# --- iris invocation ------------------------------------------------------
# The CoreWeave GPU (k8s) backend needs a `kubernetes` install that the bare
# marin `.venv` lacks; the otagent env has it AND ships the iris CLI, so use
# that binary by default. Override with $IRIS_BIN if it ever moves.
IRIS_BIN = os.environ.get(
    "IRIS_BIN", "/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris"
)
DEFAULT_CLUSTER = "cw-us-east-02a"  # the GPU RL cluster; use "marin" for TPU jobs
DEFAULT_BUNDLE_ROOT = Path(
    "/Users/benjaminfeuer/Documents/experiments/active/iris-job-bundles"
)

# JobState int -> friendly name (lib/iris/src/iris/rpc/job.proto).
STATE_NAMES = {
    0: "unspecified",
    1: "pending",
    2: "building",
    3: "running",
    4: "succeeded",
    5: "failed",
    6: "killed",
    7: "worker_failed",
    8: "unschedulable",
}
NAME_TO_INT = {v: k for k, v in STATE_NAMES.items()}

RUNNING_STATES = {"pending", "building", "running", "unspecified"}
TERMINAL_STATES = {"succeeded", "failed", "killed", "worker_failed", "unschedulable"}

# Retry/backoff for the Iris CLI call: a transient tunnel/RPC blip should not
# a transient tunnel/RPC blip should not be read as a terminal transition).
IRIS_ATTEMPTS = 3
IRIS_BACKOFFS = (2, 5)
DNS_ATTEMPTS = 4
DNS_INITIAL_BACKOFF = 2
TRANSIENT_DNS_MARKERS = (
    "dns error",
    "no records found for query",
    "temporary failure in name resolution",
    "failed to lookup address information",
    "name or service not known",
    "nodename nor servname provided",
)


@dataclass(frozen=True)
class JobBundle:
    """Stable local evidence directory for one Iris job on one cluster.

    The full Iris id is preserved as path components rather than flattened, so
    every watcher and analyzer addresses the same evidence unambiguously:
    ``<root>/jobs/<cluster>/<user>/<job>/``.
    """

    root: Path
    cluster: str
    job_id: str

    @property
    def directory(self) -> Path:
        return self.root.joinpath("jobs", self.cluster, *job_id_parts(self.job_id))

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"


def job_id_parts(job_id: str) -> tuple[str, ...]:
    """Validate an Iris id and return its safe local path components."""
    name = JobName.from_string(job_id)
    if not name.is_root:
        raise ValueError(f"Expected root Iris job id '/<user>/<job>', got {job_id!r}.")
    return (name.user, name.name)


def job_bundle(bundle_root: Path, cluster: str, job_id: str) -> JobBundle:
    """Return the canonical local bundle for ``cluster`` and ``job_id``."""
    if not cluster or "/" in cluster:
        raise ValueError(f"Invalid Iris cluster name {cluster!r}.")
    return JobBundle(bundle_root, cluster, job_id)


def load_bundle_manifest(bundle: JobBundle) -> dict[str, Any]:
    """Load an existing bundle manifest, returning an empty value if absent."""
    if not bundle.manifest_path.exists():
        return {}
    try:
        value = json.loads(bundle.manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid bundle manifest at {bundle.manifest_path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Bundle manifest at {bundle.manifest_path} must contain a JSON object.")
    return value


def write_bundle_manifest(bundle: JobBundle, updates: dict[str, Any]) -> None:
    """Merge ``updates`` into the durable manifest for a local evidence bundle."""
    bundle.directory.mkdir(parents=True, exist_ok=True)
    manifest = load_bundle_manifest(bundle)
    manifest.update(updates)
    manifest.update(
        {
            "bundle_format": 1,
            "cluster": bundle.cluster,
            "job_id": bundle.job_id,
            "bundle_directory": str(bundle.directory),
        }
    )
    bundle.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")


def is_transient_dns_failure(result: subprocess.CompletedProcess[str]) -> bool:
    """Return whether an Iris command failed only because name resolution failed."""
    if result.returncode == 0:
        return False
    message = f"{result.stderr}\n{result.stdout}".lower()
    return any(marker in message for marker in TRANSIENT_DNS_MARKERS)


def run_iris_command(
    arguments: list[str],
    *,
    cluster: str,
    iris_bin: str = IRIS_BIN,
    environment: dict[str, str] | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    """Run an Iris CLI command with bounded retries for transient DNS failures."""
    for attempt in range(DNS_ATTEMPTS):
        result = subprocess.run(
            [iris_bin, f"--cluster={cluster}", *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
        if not is_transient_dns_failure(result) or attempt == DNS_ATTEMPTS - 1:
            return result
        time.sleep(DNS_INITIAL_BACKOFF * 2**attempt)
    raise AssertionError("unreachable")


@dataclass
class JobStateSnapshot:
    """One observation of a job's authoritative lifecycle state."""

    job_id: str
    state: str  # friendly name, or "absent" when the controller has no record
    state_int: int | None
    error: str = ""
    exit_code: int | None = None
    failure_count: int | None = None
    preemption_count: int | None = None
    task_count: int | None = None
    completed_count: int | None = None
    task_state_counts: dict[str, int] = field(default_factory=dict)
    finished_at_ms: int | None = None
    source: str = ""  # "summary" | "query" | "absent"
    pods_alive: int | None = None  # kubectl cross-check, if run

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES or self.state == "absent"

    @property
    def is_running(self) -> bool:
        return self.state in RUNNING_STATES

    def verdict_line(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        parts = [f"[{ts}] {self.job_id} state={self.state}"]
        if self.state_int is not None:
            parts.append(f"({self.state_int})")
        if self.source:
            parts.append(f"src={self.source}")
        if self.pods_alive is not None:
            parts.append(f"pods={self.pods_alive}")
        if self.completed_count is not None and self.task_count is not None:
            parts.append(f"tasks={self.completed_count}/{self.task_count}")
        if self.exit_code is not None:
            parts.append(f"exit={self.exit_code}")
        if self.preemption_count:
            parts.append(f"preempts={self.preemption_count}")
        if self.error:
            parts.append(f"error={self.error!r}")
        return " ".join(parts)


# ---------- iris CLI helpers (authoritative state) ----------


def _run_iris(args: list[str], cluster: str, timeout: int = 180) -> subprocess.CompletedProcess:
    cmd = [IRIS_BIN, f"--cluster={cluster}", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def get_job_summary(job_id: str, cluster: str) -> JobStateSnapshot | None:
    """Authoritative primary: ``iris job summary <job_id>`` (text).

    NOTE: the ``--json`` option was removed from ``iris job summary`` upstream
    (marin ``main``), so we parse the text summary instead. The text header is
    rich enough for our fields, e.g.::

        State: running  exit=0  failures=0  preemptions=0
        Tasks: 0/16 completed  running=16

    Returns a snapshot, or None on a (retryable) failure. A job the controller
    has never heard of yields an error we surface as None so the caller can fall
    back to the SQL/pod path and decide "absent".
    """
    import re

    last_err = ""
    for attempt in range(IRIS_ATTEMPTS):
        try:
            proc = _run_iris(["job", "summary", job_id], cluster)
        except subprocess.TimeoutExpired:
            last_err = "timeout"
            if attempt < len(IRIS_BACKOFFS):
                time.sleep(IRIS_BACKOFFS[attempt])
            continue
        out = proc.stdout or ""
        m_state = re.search(r"State:\s*(\S+)", out)
        if proc.returncode == 0 and m_state:
            state = m_state.group(1).lower()

            def _int(pat: str) -> int | None:
                m = re.search(pat, out)
                if not m or m.group(1) == "-":
                    return None
                try:
                    return int(m.group(1))
                except ValueError:
                    return None

            # "Tasks: <done>/<total> completed  <k=v> <k=v> ..." (trailing k=v are per-state counts)
            task_count = completed_count = None
            task_state_counts: dict[str, int] = {}
            m_tasks = re.search(r"Tasks:\s*(\d+)\s*/\s*(\d+)\s*completed(.*)", out)
            if m_tasks:
                completed_count = int(m_tasks.group(1))
                task_count = int(m_tasks.group(2))
                for k, v in re.findall(r"(\w+)=(\d+)", m_tasks.group(3)):
                    task_state_counts[k] = int(v)
            return JobStateSnapshot(
                job_id=job_id,
                state=state,
                state_int=NAME_TO_INT.get(state),
                error="",
                exit_code=_int(r"exit=(-?\d+|-)"),
                failure_count=_int(r"failures=(\d+)"),
                preemption_count=_int(r"preemptions=(\d+)"),
                task_count=task_count,
                completed_count=completed_count,
                task_state_counts=task_state_counts,
                source="summary",
            )
        last_err = (proc.stderr or proc.stdout)[-400:]
        if attempt < len(IRIS_BACKOFFS):
            time.sleep(IRIS_BACKOFFS[attempt])
    print(f"  [summary] failed after {IRIS_ATTEMPTS} attempts: {last_err}", file=sys.stderr)
    return None


def get_job_state_via_query(job_id: str, cluster: str) -> JobStateSnapshot | None:
    """Fallback: ``iris query "SELECT state FROM jobs WHERE job_id=..."``.

    Lighter than ``summary`` (numeric state only). Returns None on failure or an
    empty result (no such job → caller treats as candidate for "absent").
    """
    sql = f"SELECT job_id, state FROM jobs WHERE job_id='{job_id}'"
    try:
        proc = _run_iris(["query", sql, "-f", "csv"], cluster)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    rows = [ln for ln in proc.stdout.strip().splitlines() if ln and "," in ln]
    # first line is the header "job_id,state"
    data_rows = [r for r in rows if not r.startswith("job_id,")]
    if not data_rows:
        return None
    try:
        _jid, state_s = data_rows[0].rsplit(",", 1)
        state_int = int(state_s)
    except (ValueError, IndexError):
        return None
    return JobStateSnapshot(
        job_id=job_id,
        state=STATE_NAMES.get(state_int, f"state_{state_int}"),
        state_int=state_int,
        source="query",
    )


# ---------- kubectl cross-check (the disappeared / 0-pods case) ----------


def count_live_pods(job_id: str) -> int | None:
    """Count cluster pods whose name carries the job's short name.

    Requires KUBECONFIG to point at the cluster (e.g. ~/.kube/coreweave-iris-gpu).
    Returns the pod count, or None if kubectl is unavailable / errors. iris job
    pods are named after the job's short id (last path segment); a vanished job
    has 0.
    """
    short = job_id.rstrip("/").rsplit("/", 1)[-1]
    try:
        proc = subprocess.run(
            ["kubectl", "get", "pods", "-A", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    return sum(1 for ln in proc.stdout.splitlines() if short in ln)


# ---------- the authoritative single observation ----------


def get_job_state(job_id: str, cluster: str = DEFAULT_CLUSTER, check_pods: bool = False) -> JobStateSnapshot:
    """Return the current authoritative state of ``job_id``.

    Order: ``job summary --json`` (primary) → ``query`` (fallback) → if both say
    "no such job" AND (when ``check_pods``) the cluster has 0 matching pods, the
    job has disappeared → state="absent" (a TERMINAL signal). If the controller
    is simply unreachable we raise, so a transient outage is not misread as a
    terminal transition.
    """
    snap = get_job_summary(job_id, cluster)
    if snap is None:
        snap = get_job_state_via_query(job_id, cluster)

    pods = count_live_pods(job_id) if check_pods else None

    if snap is None:
        # Controller has no record (or both calls failed). Distinguish
        # "disappeared" from "controller unreachable" via the pod count.
        if pods == 0:
            return JobStateSnapshot(
                job_id=job_id, state="absent", state_int=None,
                source="absent", pods_alive=0,
                error="no controller record AND 0 pods on cluster (disappeared)",
            )
        raise RuntimeError(
            f"could not read state for {job_id}: iris summary+query both failed "
            f"and pod count is {pods} (controller unreachable, or job not yet placed)"
        )

    snap.pods_alive = pods
    return snap


# ---------- the watch loop ----------


def watch(
    job_id: str,
    cluster: str = DEFAULT_CLUSTER,
    interval: int = 60,
    check_pods: bool = True,
    max_polls: int | None = None,
) -> JobStateSnapshot:
    """Poll authoritative state until the job leaves RUNNING; return the terminal
    snapshot. Emits a line on EVERY state transition (and the first observation).
    """
    print(
        f"[watch] {job_id} on cluster={cluster} every {interval}s "
        f"(authoritative iris job-state poll; check_pods={check_pods})",
        file=sys.stderr,
    )
    prev_state: str | None = None
    polls = 0
    while True:
        polls += 1
        try:
            snap = get_job_state(job_id, cluster, check_pods=check_pods)
        except RuntimeError as e:
            # Transient: report, keep watching (do NOT treat as terminal).
            print(f"  [watch] transient read error: {e}", file=sys.stderr)
            snap = None
        if snap is not None:
            if snap.state != prev_state:
                print(snap.verdict_line(), flush=True)
                prev_state = snap.state
            if snap.is_terminal:
                print(f"[watch] TERMINAL: {snap.state}", file=sys.stderr)
                return snap
        if max_polls is not None and polls >= max_polls:
            print(f"[watch] reached max_polls={max_polls}; stopping", file=sys.stderr)
            return snap if snap is not None else JobStateSnapshot(
                job_id=job_id, state="unspecified", state_int=0, source="timeout"
            )
        time.sleep(interval)


_EXIT_FOR_STATE = {
    "succeeded": 0,
    "failed": 1,
    "killed": 1,
    "worker_failed": 1,
    "unschedulable": 1,
    "absent": 2,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_id", help="iris job id, e.g. /benjaminfeuer/rl-131k-cpdcp2r3")
    ap.add_argument("--cluster", default=DEFAULT_CLUSTER, help=f"iris cluster (default: {DEFAULT_CLUSTER}; use 'marin' for TPU)")
    ap.add_argument("--interval", type=int, default=60, help="poll interval seconds (default 60)")
    ap.add_argument("--once", action="store_true", help="print current state once and exit")
    ap.add_argument("--no-pods", action="store_true", help="skip the kubectl pod cross-check")
    ap.add_argument("--max-polls", type=int, default=None, help="stop after N polls even if still running")
    ap.add_argument("--json", action="store_true", help="emit the final snapshot as JSON")
    args = ap.parse_args()

    check_pods = not args.no_pods
    try:
        if args.once:
            snap = get_job_state(args.job_id, args.cluster, check_pods=check_pods)
            print(snap.verdict_line(), flush=True)
        else:
            snap = watch(args.job_id, args.cluster, interval=args.interval, check_pods=check_pods, max_polls=args.max_polls)
    except RuntimeError as e:
        print(f"[watch] ERROR: {e}", file=sys.stderr)
        return 3

    if args.json:
        from dataclasses import asdict
        print(json.dumps(asdict(snap), indent=2, default=str))

    return _EXIT_FOR_STATE.get(snap.state, 3 if not snap.is_terminal else 0)


if __name__ == "__main__":
    sys.exit(main())
