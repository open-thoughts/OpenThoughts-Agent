#!/usr/bin/env python3
"""Sync and summarize every active CoreWeave Iris RL job for one user.

The monitor is deliberately read-only.  Each Iris job gets one stable local
directory, keyed by its full Iris job id, so repeated sweeps and Iris task
retries refresh the same artifacts instead of making timestamped copies.  It
captures the complete finelog plus complete pod/Ray/vLLM logs, then mirrors
the 500 most recently modified Harbor ``trace_jobs`` across the active RL fleet
by default. The recent trace selection is based on object-store ``LastModified``
metadata, never trace names. Use ``--trace-sync-limit 0`` for a deliberately
full trace sync. The sync still skips non-log objects larger than the configured
size bound; that rule avoids repeatedly downloading giant rollout payloads while
preserving any diagnostic log regardless of size.

By default the scope is the current lab user's active RL jobs on both
CoreWeave GPU clusters.  Use ``--all-users`` only when cross-user monitoring is
intended.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.iris.coreweave_ops import (  # noqa: E402
    CLUSTERS as COREWEAVE_CLUSTERS,
    NAMESPACE,
    iter_objects,
    kubectl_base,
    object_store_client,
    ray_log_inventory,
    resolve_runtime_python,
    safe_relative_path,
    save_ray_logs,
    split_s3_uri,
)
from scripts.iris.iris_ops import (  # noqa: E402
    DEFAULT_BUNDLE_ROOT,
    MonitorError,
    StyledCell,
    box_table,
    filter_records,
    format_duration,
    job_bundle,
    parse_regex_filters,
    run_iris_command,
    write_bundle_manifest,
    write_error_report,
)


DEFAULT_USER = "benjaminfeuer"
DEFAULT_MAX_NON_LOG_BYTES = 100 * 1024 * 1024
DEFAULT_TRACE_SYNC_LIMIT = 500
ACTIVE_STATES = {1: "pending", 2: "building", 3: "running"}
STATE_NAMES = {
    **ACTIVE_STATES,
    4: "succeeded",
    5: "failed",
    6: "killed",
    7: "worker_failed",
    8: "unschedulable",
}
RL_ENTRYPOINT_MARKERS = ("start_rl_iris_controller.py", "cloud.iris.run_rl")
TRIALS_URI_PATTERN = re.compile(
    r"(?:terminal_bench_config\.trials_dir=|--trials-dir(?:=|\s+))"
    r"(?P<uri>s3://[^\s'\"\\]+)"
)
TRAIN_DATA_PATTERN = re.compile(
    r"--train[_-]data(?:=|\s+)(?:'(?P<single>\[[^']+\])'|\"(?P<double>\[[^\"]+\])\"|(?P<bare>\[[^\s]+\]))"
)
PROGRESS_PATTERN = re.compile(r"Training Step Progress:\s*(\d+)\s*/\s*(\d+)")
MIRROR_PATTERN = re.compile(r"WANDB_MIRROR kind=train step=(\d+) metrics=(\{.*\})")
ERROR_PATTERNS = (
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"(?:RayTaskError|ActorDiedError|WorkerCrashedError)"),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"Train loop failed", re.IGNORECASE),
)
LOG_SUFFIXES = (".log", ".out", ".err", ".jsonl", ".txt")
MISSING_OBJECT_ERROR_CODES = {"404", "NoSuchKey", "NoSuchObject", "NotFound"}


@dataclass(frozen=True)
class Cluster:
    name: str
    kubeconfig: Path
    context: str | None


@dataclass(frozen=True)
class RlJob:
    cluster: Cluster
    job_id: str
    state: str
    submitted_at_ms: int
    entrypoint: str
    dataset: str = "—"
    finished_at_ms: int | None = None

    @property
    def short_name(self) -> str:
        return self.job_id.rstrip("/").rsplit("/", 1)[-1]

    @property
    def is_terminal(self) -> bool:
        return self.state in {"succeeded", "failed"}


@dataclass(frozen=True)
class ArtifactResult:
    finelog: str
    pod_logs: str
    ray_logs: str
    traces: str
    trace_started: int | None
    trace_completed: int | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class TraceJobObjects:
    """One remote trace directory and its latest object modification time."""

    name: str
    last_modified: datetime
    objects: tuple[dict[str, Any], ...]
    completed: bool


@dataclass(frozen=True)
class TraceInventory:
    """Remote trace objects for one RL job, collected before fleet selection."""

    job: RlJob
    bucket: str
    root_prefix: str
    client: Any
    traces: tuple[TraceJobObjects, ...]
    available: int
    completed: int


@dataclass
class ProgressReporter:
    """Emit phase-level timing to stderr without contaminating the status table."""

    enabled: bool = True
    started_at: float = field(default_factory=time.monotonic)

    def phase(self, message: str) -> None:
        if not self.enabled:
            return
        elapsed_seconds = int(time.monotonic() - self.started_at)
        elapsed = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
        print(f"[rl-watch +{elapsed}] {message}", file=sys.stderr, flush=True)


CLUSTERS = tuple(
    Cluster(name, config.kubeconfig, config.context)
    for name, config in COREWEAVE_CLUSTERS.items()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Root for canonical local Iris evidence bundles and RL reports.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Only include jobs submitted within this many hours; 0 means all history (default: 24).",
    )
    parser.add_argument("--user", default=DEFAULT_USER, help=f"Iris user to monitor (default: {DEFAULT_USER}).")
    parser.add_argument("--all-users", action="store_true", help="Discover active RL jobs for every user.")
    parser.add_argument(
        "--max-non-log-bytes",
        type=int,
        default=DEFAULT_MAX_NON_LOG_BYTES,
        help="Skip non-log trace objects larger than this many bytes (default: 100 MiB; 0 disables).",
    )
    parser.add_argument(
        "--trace-sync-limit",
        type=int,
        default=DEFAULT_TRACE_SYNC_LIMIT,
        help=(
            "Sync only this many most recently modified trace directories across all discovered RL jobs "
            "(default: 500; 0 syncs every remote trace)."
        ),
    )
    parser.add_argument("--no-sync", action="store_true", help="Report lifecycle state without collecting artifacts.")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=REGEX",
        help=(
            "Keep jobs matching every case-insensitive regex filter. Available keys: "
            "cluster, job, name, dataset, type, state, submitted, duration."
        ),
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Suppress stderr phase/timing updates while artifacts are collected.",
    )
    return parser.parse_args()


def run_iris(cluster: Cluster, arguments: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["KUBECONFIG"] = str(cluster.kubeconfig)
    return run_iris_command(
        arguments,
        cluster=cluster.name,
        iris_bin="/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris",
        environment=environment,
        timeout=timeout,
    )


def entrypoint_text(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw))
    except json.JSONDecodeError:
        return raw


def command_strings(entrypoint: str) -> list[str]:
    """Return all string leaves from an Iris entrypoint JSON payload."""
    try:
        value = json.loads(entrypoint)
    except json.JSONDecodeError:
        return [entrypoint]

    strings: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return strings


def dataset_from_entrypoint(entrypoint: str) -> str:
    """Extract and deduplicate the submitted dataset list without guessing from a config."""
    datasets: list[str] = []
    for command in command_strings(entrypoint):
        for match in TRAIN_DATA_PATTERN.finditer(command):
            raw_dataset_list = next(value for value in match.groupdict().values() if value is not None)
            try:
                values = json.loads(raw_dataset_list)
            except json.JSONDecodeError:
                continue
            if isinstance(values, list):
                datasets.extend(str(value) for value in values)
    return ", ".join(dict.fromkeys(datasets)) or "—"


def csv_rows(output: str) -> list[dict[str, str]]:
    """Parse Iris CSV after its informational controller/tunnel preamble."""
    lines = output.splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith("job_id,")), None)
    if header_index is None:
        raise ValueError("Iris query returned no CSV job_id header")
    return list(csv.DictReader(lines[header_index:]))


def discover_rl_jobs(
    cluster: Cluster,
    user: str | None,
    *,
    submitted_since_ms: int | None = None,
) -> tuple[list[RlJob], list[str]]:
    where_user = "" if user is None else f" AND j.job_id LIKE '/{user}/%'"
    where_submission = "" if submitted_since_ms is None else f" AND j.submitted_at_ms >= {submitted_since_ms}"
    sql = (
        "SELECT j.job_id, j.state, j.submitted_at_ms, j.finished_at_ms, jc.entrypoint_json "
        "FROM jobs j JOIN job_config jc ON j.job_id=jc.job_id "
        "WHERE ("
        f"j.state IN ({','.join(str(state) for state in sorted(ACTIVE_STATES))}) "
        "OR j.state IN (4,5)"
        f"){where_user}{where_submission} "
        "ORDER BY j.submitted_at_ms DESC"
    )
    result = run_iris(cluster, ["query", sql, "-f", "csv"])
    if result.returncode:
        message = (result.stderr or result.stdout).strip().replace("\n", " ")
        return [], [f"{cluster.name}: discovery failed: {message[-240:]}"]

    jobs: list[RlJob] = []
    try:
        rows = csv_rows(result.stdout)
    except ValueError as error:
        return [], [f"{cluster.name}: discovery failed: {error}"]
    for row in rows:
        entrypoint = entrypoint_text(row.get("entrypoint_json", ""))
        if not any(marker in entrypoint for marker in RL_ENTRYPOINT_MARKERS):
            continue
        try:
            state_code = int(row["state"])
            submitted_at_ms = int(row["submitted_at_ms"])
        except (KeyError, ValueError):
            continue
        jobs.append(
            RlJob(
                cluster=cluster,
                job_id=row["job_id"],
                state=STATE_NAMES.get(state_code, f"state-{state_code}"),
                submitted_at_ms=submitted_at_ms,
                entrypoint=entrypoint,
                dataset=dataset_from_entrypoint(entrypoint),
                finished_at_ms=int(row["finished_at_ms"]) if row.get("finished_at_ms") else None,
            )
        )
    return jobs, []


def job_directory(bundle_root: Path, job: RlJob) -> Path:
    """Return the shared canonical evidence directory for this Iris job."""
    return job_bundle(bundle_root, job.cluster.name, job.job_id).directory


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def fetch_finelog(job: RlJob, destination: Path) -> tuple[str, str | None]:
    result = run_iris(
        job.cluster,
        ["job", "logs", job.job_id, "--max-lines", "10000000", "--no-tail"],
        timeout=900,
    )
    stderr_path = destination / "finelog.stderr"
    stderr_path.write_text(result.stderr)
    if result.returncode:
        message = (result.stderr or result.stdout).strip().replace("\n", " ")
        return "unavailable", f"finelog: {message[-180:]}"
    (destination / "finelog.log").write_text(result.stdout)
    return f"{len(result.stdout.splitlines()):,} lines", None


def job_pods(job: RlJob) -> list[tuple[str, str]]:
    base = kubectl_base(
        COREWEAVE_CLUSTERS[job.cluster.name], SimpleNamespace(kubeconfig=None, kube_context=None)
    )
    result = subprocess.run(
        [*base, "-n", NAMESPACE, "get", "pods", "-o", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-240:])
    needle = job.short_name.lower()
    return sorted(
        (
            item["metadata"]["name"],
            item.get("status", {}).get("phase", "Unknown"),
        )
        for item in json.loads(result.stdout).get("items", [])
        if needle in item.get("metadata", {}).get("name", "").lower()
    )


def fetch_complete_pod_log(base: list[str], pod: str, destination: Path) -> None:
    result = subprocess.run(
        [*base, "-n", NAMESPACE, "logs", pod, "-c", "task", "--tail=-1"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    destination.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-240:])


def fetch_complete_ray_logs(base: list[str], pod: str, destination: Path) -> int:
    runtime_python = resolve_runtime_python(base, pod, "task")
    inventory = ray_log_inventory(base, pod, "task", patterns=None, python_executable=runtime_python)
    if not inventory:
        return 0
    try:
        saved, skipped = save_ray_logs(
            base,
            pod,
            "task",
            inventory,
            sys.maxsize,
            destination,
            incremental=True,
            python_executable=runtime_python,
        )
    except RuntimeError:
        # Ray rotates/removes worker logs while a live pod is writing. Rebuild
        # the inventory once so a stale path cannot abort the whole fleet scan.
        inventory = ray_log_inventory(base, pod, "task", patterns=None, python_executable=runtime_python)
        if not inventory:
            return 0
        saved, skipped = save_ray_logs(
            base,
            pod,
            "task",
            inventory,
            sys.maxsize,
            destination,
            incremental=True,
            python_executable=runtime_python,
        )
    if skipped:
        raise AssertionError("A maximum-size sync should not skip Ray/vLLM logs.")
    return len(saved)


def sync_pod_and_ray_logs(
    job: RlJob,
    destination: Path,
    *,
    progress: ProgressReporter | None = None,
) -> tuple[str, str, list[str]]:
    """Capture all current pod stdout plus all Ray/vLLM logs without a size cap."""
    errors: list[str] = []
    try:
        pods = job_pods(job)
    except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return "unavailable", "unavailable", [f"pod discovery: {error}"]
    if not pods:
        return "no pod yet", "no pod yet", []
    running_pods = [pod for pod, phase in pods if phase == "Running"]
    if not running_pods:
        phases = ", ".join(sorted(phase for _, phase in pods))
        return f"{len(pods)} pod(s): {phases}", "awaiting host", []

    base = kubectl_base(
        COREWEAVE_CLUSTERS[job.cluster.name], SimpleNamespace(kubeconfig=None, kube_context=None)
    )
    pod_dir = destination / "pod_logs"
    ray_dir = destination / "ray_vllm_logs"
    pod_dir.mkdir(exist_ok=True)
    ray_dir.mkdir(exist_ok=True)
    ray_files = 0
    for index, pod in enumerate(running_pods, start=1):
        try:
            if progress:
                progress.phase(
                    f"pod stdout {index}/{len(running_pods)} {job.short_name}/{pod}"
                )
            fetch_complete_pod_log(base, pod, pod_dir / f"{pod}.log")
        except (RuntimeError, subprocess.SubprocessError) as error:
            errors.append(f"{pod} stdout: {error}")
        try:
            if progress:
                progress.phase(
                    f"Ray/vLLM logs {index}/{len(running_pods)} {job.short_name}/{pod}"
                )
            pod_ray_dir = ray_dir / pod
            pod_ray_dir.mkdir(exist_ok=True)
            ray_files += fetch_complete_ray_logs(base, pod, pod_ray_dir)
        except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            errors.append(f"{pod} Ray/vLLM: {error}")
    return f"{len(pods)} pod(s), {len(running_pods)} Running", f"{ray_files:,} files", errors


def trials_uri(job: RlJob) -> str:
    match = TRIALS_URI_PATTERN.search(job.entrypoint)
    if match:
        return match.group("uri").rstrip("/" )
    # This is exactly the launcher's --trials-dir auto convention.  Keep the
    # fallback local and visible rather than reading a possibly different YAML.
    return f"s3://marin-us-east-02a/iris/{job.short_name}/trace_jobs"


def is_log_object(relative_path: str) -> bool:
    path = relative_path.lower()
    filename = path.rsplit("/", 1)[-1]
    return (
        any(suffix in filename for suffix in LOG_SUFFIXES)
        or "/logs/" in path
        or path.startswith("logs/")
    )


def is_missing_object_error(error: ClientError) -> bool:
    """Return whether a listed object disappeared before it could be downloaded."""
    error_details = error.response.get("Error", {})
    return str(error_details.get("Code")) in MISSING_OBJECT_ERROR_CODES


def recent_trace_jobs(
    objects: list[dict[str, Any]], root_prefix: str, trace_sync_limit: int
) -> tuple[list[TraceJobObjects], int, int]:
    """Return trace directories ordered by latest remote object modification.

    Object-store ``LastModified`` is the only ordering source. A trace is
    complete when any object in its directory is ``result.json``; those counts
    cover the complete remote prefix even when a recent subset is downloaded.
    """
    traces: dict[str, list[dict[str, Any]]] = {}
    for item in objects:
        relative = item["Key"].removeprefix(root_prefix)
        if relative:
            traces.setdefault(relative.split("/", 1)[0], []).append(item)

    trace_jobs: list[TraceJobObjects] = []
    completed = 0
    for name, trace_objects in traces.items():
        latest_modified: datetime | None = None
        completed_trace = False
        for item in trace_objects:
            modified = item.get("LastModified")
            if not isinstance(modified, datetime):
                raise ValueError(f"Object {item['Key']!r} is missing LastModified metadata")
            if latest_modified is None or modified > latest_modified:
                latest_modified = modified
            relative = item["Key"].removeprefix(root_prefix)
            completed_trace = completed_trace or relative.endswith("/result.json")
        assert latest_modified is not None
        completed += completed_trace
        trace_jobs.append(TraceJobObjects(name, latest_modified, tuple(trace_objects), completed_trace))

    trace_jobs.sort(key=lambda trace: (trace.last_modified, trace.name), reverse=True)
    selected = trace_jobs if trace_sync_limit == 0 else trace_jobs[:trace_sync_limit]
    return selected, len(trace_jobs), completed


def trace_selection_manifest(
    selected: list[TraceJobObjects],
    inventory: TraceInventory,
    trace_sync_limit: int,
    fleet_available: int,
    fleet_selected: int,
) -> dict[str, Any]:
    """Describe this job's share of a bounded, fleet-wide trace selection."""
    return {
        "selection": "latest_object_store_last_modified_across_active_rl_jobs",
        "trace_sync_limit": trace_sync_limit,
        "available_traces": inventory.available,
        "selected_traces": len(selected),
        "omitted_traces": inventory.available - len(selected),
        "fleet_available_traces": fleet_available,
        "fleet_selected_traces": fleet_selected,
        "selected": [
            {"name": trace.name, "last_modified": trace.last_modified.isoformat(), "completed": trace.completed}
            for trace in selected
        ],
    }


def collect_trace_inventory(job: RlJob) -> TraceInventory:
    """List all remote objects for one job before selecting a fleet-wide subset."""
    uri = trials_uri(job)
    bucket, prefix = split_s3_uri(uri)
    base = kubectl_base(
        COREWEAVE_CLUSTERS[job.cluster.name], SimpleNamespace(kubeconfig=None, kube_context=None)
    )
    client = object_store_client(base, COREWEAVE_CLUSTERS[job.cluster.name])
    root_prefix = f"{prefix.rstrip('/')}/"
    objects = iter_objects(client, bucket, root_prefix)
    traces, available, completed = recent_trace_jobs(objects, root_prefix, trace_sync_limit=0)
    return TraceInventory(job, bucket, root_prefix, client, tuple(traces), available, completed)


def select_recent_fleet_traces(
    inventories: list[TraceInventory], trace_sync_limit: int
) -> dict[tuple[str, str], list[TraceJobObjects]]:
    """Select the newest trace directories globally, using S3 modification metadata."""
    candidates = [(inventory, trace) for inventory in inventories for trace in inventory.traces]
    candidates.sort(
        key=lambda candidate: (
            candidate[1].last_modified,
            candidate[0].job.cluster.name,
            candidate[0].job.job_id,
            candidate[1].name,
        ),
        reverse=True,
    )
    if trace_sync_limit:
        candidates = candidates[:trace_sync_limit]
    selected: dict[tuple[str, str], list[TraceJobObjects]] = {}
    for inventory, trace in candidates:
        key = (inventory.job.cluster.name, inventory.job.job_id)
        selected.setdefault(key, []).append(trace)
    return selected


def sync_trace_inventory(
    inventory: TraceInventory,
    destination: Path,
    selected: list[TraceJobObjects],
    max_non_log_bytes: int,
    trace_sync_limit: int,
    fleet_available: int,
    fleet_selected: int,
    progress: ProgressReporter | None = None,
) -> tuple[str, int, int, str | None]:
    """Mirror the globally selected traces for one job without deleting prior evidence."""
    destination.mkdir(exist_ok=True)
    write_json(
        destination / "sync_selection.json",
        trace_selection_manifest(selected, inventory, trace_sync_limit, fleet_available, fleet_selected),
    )
    copied = skipped = 0
    skipped_objects: list[dict[str, Any]] = []
    candidate_objects = sum(len(trace.objects) for trace in selected)
    candidate_bytes = sum(int(item["Size"]) for trace in selected for item in trace.objects)
    if progress:
        progress.phase(
            f"trace transfer {inventory.job.cluster.name}/{inventory.job.short_name}: "
            f"{len(selected):,} traces, {candidate_objects:,} objects, "
            f"{candidate_bytes / 1_048_576:.1f} MiB candidate payload"
        )
    inspected = 0
    try:
        for trace in selected:
            for item in trace.objects:
                inspected += 1
                relative = item["Key"].removeprefix(inventory.root_prefix)
                size = int(item["Size"])
                if max_non_log_bytes and size > max_non_log_bytes and not is_log_object(relative):
                    skipped += 1
                    skipped_objects.append({"key": relative, "size": size, "reason": "non_log_size_limit"})
                    if progress and (inspected == candidate_objects or inspected % 25 == 0):
                        progress.phase(
                            f"trace transfer {inventory.job.short_name}: "
                            f"{inspected:,}/{candidate_objects:,} objects inspected; "
                            f"{copied:,} downloaded, {skipped:,} size-skipped"
                        )
                    continue
                local_path = destination / safe_relative_path(item["Key"], inventory.root_prefix)
                if local_path.exists() and local_path.stat().st_size == size:
                    if progress and (inspected == candidate_objects or inspected % 25 == 0):
                        progress.phase(
                            f"trace transfer {inventory.job.short_name}: "
                            f"{inspected:,}/{candidate_objects:,} objects inspected; "
                            f"{copied:,} downloaded, {skipped:,} size-skipped"
                        )
                    continue
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    inventory.client.download_file(inventory.bucket, item["Key"], str(local_path))
                except ClientError as error:
                    if not is_missing_object_error(error):
                        raise
                    skipped += 1
                    skipped_objects.append(
                        {"key": relative, "size": size, "reason": "missing_after_listing"}
                    )
                    continue
                copied += 1
                if progress and (inspected == candidate_objects or inspected % 25 == 0):
                    progress.phase(
                        f"trace transfer {inventory.job.short_name}: "
                        f"{inspected:,}/{candidate_objects:,} objects inspected; "
                        f"{copied:,} downloaded, {skipped:,} size-skipped"
                    )
    except Exception as error:  # object stores may race a currently-uploading trace
        write_json(destination / "skipped_objects.json", skipped_objects)
        return (
            f"partial: fleet {len(selected):,}/{inventory.available:,} selected "
            f"({fleet_selected:,}/{fleet_available:,} total); {copied:,} copied, {skipped:,} skipped",
            inventory.available,
            inventory.completed,
            str(error)[-240:],
        )
    write_json(destination / "skipped_objects.json", skipped_objects)
    scope = "all" if trace_sync_limit == 0 else "newest"
    return (
        f"fleet {scope} {len(selected):,}/{inventory.available:,} selected "
        f"({fleet_selected:,}/{fleet_available:,} total); {copied:,} copied, {skipped:,} skipped",
        inventory.available,
        inventory.completed,
        None,
    )


def parse_metrics(finelog: Path) -> tuple[int | None, int | None, dict[str, Any], str | None]:
    if not finelog.exists():
        return None, None, {}, None
    try:
        text = finelog.read_text(errors="replace")
    except OSError as error:
        return None, None, {}, f"could not read finelog: {error}"
    progress = PROGRESS_PATTERN.findall(text)
    step = int(progress[-1][0]) if progress else None
    total = int(progress[-1][1]) if progress else None
    for line in reversed(text.splitlines()):
        match = MIRROR_PATTERN.search(line)
        if not match:
            continue
        try:
            metrics = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        return int(match.group(1)), total, metrics, None
    return step, total, {}, None


def metric(metrics: dict[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def display_metric(value: Any | None, precision: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def terminal_signal(finelog: Path) -> str | None:
    if not finelog.exists():
        return None
    try:
        tail = finelog.read_text(errors="replace")[-2_000_000:]
    except OSError:
        return None
    for pattern in ERROR_PATTERNS:
        match = pattern.search(tail)
        if match:
            return match.group(0)
    return None


def sync_warning(errors: tuple[str, ...]) -> str | None:
    """Render a stable table cell for artifact-sync errors, never raw proxy bodies."""
    if not errors:
        return None
    first_error = errors[0]
    if "Ray/vLLM" in first_error:
        return "Ray/vLLM log sync unavailable; local diagnostic saved"
    return first_error[-90:]


def _monitor_error(scope: str, operation: str, error: object) -> MonitorError:
    message = str(error).strip() or type(error).__name__
    return MonitorError(scope, operation, message)


def _state_cell(state: str) -> StyledCell:
    if state in {"running", "succeeded"}:
        tone = "success"
    elif state in {"pending", "building"}:
        tone = "warning"
    else:
        tone = "error"
    return StyledCell(state, tone)


def job_filter_values(job: RlJob, *, now_ms: int) -> dict[str, str]:
    """Return the pre-sync RL job fields available to ``--filter``."""
    return {
        "cluster": job.cluster.name,
        "job": job.job_id,
        "name": job.short_name,
        "dataset": job.dataset,
        "type": "RL",
        "state": job.state,
        "submitted": datetime.fromtimestamp(job.submitted_at_ms / 1000, UTC).strftime("%m-%d %H:%M"),
        "duration": format_duration(job.submitted_at_ms, job.finished_at_ms, now_ms=now_ms),
    }


def report_row(job: RlJob, artifacts: ArtifactResult, directory: Path) -> list[object]:
    """Build one status row; monitor failures belong in the separate error report."""
    step, total, metrics, _parse_error = parse_metrics(directory / "finelog.log")
    step_display = "—" if step is None else f"{step}/{total if total is not None else '—'}"
    reward = metric(metrics, "reward/avg_raw_reward", "loss/avg_final_rewards")
    policy_loss = metric(metrics, "policy/policy_loss", "policy_loss")
    grad_norm = metric(metrics, "policy/raw_grad_norm", "raw_grad_norm")
    entropy = metric(metrics, "policy/policy_entropy", "policy_entropy")
    log_ratio = metric(metrics, "tis/log_ratio_abs_mean", "log_ratio_abs_mean")
    signal = terminal_signal(directory / "finelog.log")
    trend = f"entropy={display_metric(entropy)}; TIS log-ratio={display_metric(log_ratio)}"
    if signal:
        trend = "workload error detected; see error report"
    elif step is None:
        trend += "; bring-up/buffer (metrics not emitted)"
    return [
        f"{job.cluster.name}/{job.short_name}",
        job.dataset,
        _state_cell(job.state),
        step_display,
        display_metric(reward),
        display_metric(policy_loss),
        display_metric(grad_norm),
        artifacts.traces,
        StyledCell(trend, "error" if signal else "muted"),
    ]


def sync_job(
    job: RlJob,
    bundle_root: Path,
    *,
    no_sync: bool,
    terminal_only: bool = False,
    progress: ProgressReporter | None = None,
) -> tuple[ArtifactResult, Path]:
    """Sync non-trace evidence for one job before the fleet trace selection."""
    bundle = job_bundle(bundle_root, job.cluster.name, job.job_id)
    directory = bundle.directory
    directory.mkdir(parents=True, exist_ok=True)
    if no_sync:
        return ArtifactResult("not requested", "not requested", "not requested", "not requested", None, None, ()), directory
    errors: list[str] = []
    if progress:
        progress.phase(f"finelog sync {job.cluster.name}/{job.short_name}")
    finelog, error = fetch_finelog(job, directory)
    if error:
        errors.append(error)
    if terminal_only:
        return ArtifactResult(
            finelog,
            "not requested (terminal)",
            "not requested (terminal)",
            "not requested (terminal)",
            None,
            None,
            tuple(errors),
        ), directory
    if progress:
        progress.phase(f"pod + Ray/vLLM log sync {job.cluster.name}/{job.short_name}")
    pod_logs, ray_logs, pod_errors = sync_pod_and_ray_logs(job, directory, progress=progress)
    errors.extend(pod_errors)
    return ArtifactResult(finelog, pod_logs, ray_logs, "pending fleet selection", None, None, tuple(errors)), directory


def sync_fleet_trace_jobs(
    job_directories: list[tuple[RlJob, Path]],
    max_non_log_bytes: int,
    trace_sync_limit: int,
    progress: ProgressReporter | None = None,
) -> dict[tuple[str, str], tuple[str, int | None, int | None, str | None]]:
    """Select at most one global trace budget, then sync each job's selected share."""
    statuses: dict[tuple[str, str], tuple[str, int | None, int | None, str | None]] = {}
    inventories: list[TraceInventory] = []
    for index, (job, _) in enumerate(job_directories, start=1):
        key = (job.cluster.name, job.job_id)
        try:
            if progress:
                progress.phase(
                    f"trace inventory {index}/{len(job_directories)} "
                    f"{job.cluster.name}/{job.short_name}"
                )
            inventories.append(collect_trace_inventory(job))
        except Exception as error:
            # Object-store failures must degrade one row, not abort the fleet-wide report.
            statuses[key] = ("unavailable", None, None, str(error)[-240:])

    selected = select_recent_fleet_traces(inventories, trace_sync_limit)
    fleet_available = sum(inventory.available for inventory in inventories)
    fleet_selected = sum(len(traces) for traces in selected.values())
    if progress:
        progress.phase(
            f"trace selection: {fleet_selected:,}/{fleet_available:,} newest trace jobs across "
            f"{len(inventories):,} RL jobs"
        )
    directories = {(job.cluster.name, job.job_id): directory for job, directory in job_directories}
    for inventory in inventories:
        key = (inventory.job.cluster.name, inventory.job.job_id)
        statuses[key] = sync_trace_inventory(
            inventory,
            directories[key] / "trace_jobs",
            selected.get(key, []),
            max_non_log_bytes,
            trace_sync_limit,
            fleet_available,
            fleet_selected,
            progress,
        )
    return statuses


def write_job_manifest(
    job: RlJob,
    bundle_root: Path,
    directory: Path,
    artifacts: ArtifactResult,
    max_non_log_bytes: int,
    trace_sync_limit: int,
) -> None:
    bundle = job_bundle(bundle_root, job.cluster.name, job.job_id)
    write_bundle_manifest(
        bundle,
        {
            "kind": "rl",
            "job": asdict(job),
            "job_directory": str(directory),
            "trials_uri": trials_uri(job),
            "synced_at": datetime.now(UTC).isoformat(),
            "max_non_log_bytes": max_non_log_bytes,
            "trace_sync_limit": trace_sync_limit,
            "artifacts": asdict(artifacts),
        },
    )


def main() -> int:
    args = parse_args()
    if args.max_non_log_bytes < 0:
        raise ValueError("--max-non-log-bytes must be non-negative")
    if args.trace_sync_limit < 0:
        raise ValueError("--trace-sync-limit must be non-negative")
    if args.hours < 0:
        raise ValueError("--hours must be non-negative")
    if not args.all_users and not re.fullmatch(r"[A-Za-z0-9_-]+", args.user):
        raise ValueError("--user may contain only letters, numbers, _ and -")
    filters = parse_regex_filters(
        args.filter,
        {"cluster", "job", "name", "dataset", "type", "state", "submitted", "duration"},
    )
    args.bundle_root.mkdir(parents=True, exist_ok=True)
    report_directory = args.bundle_root / "reports" / "rl"
    report_directory.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(enabled=not args.quiet_progress)
    checked_at = datetime.now(UTC)
    now_ms = int(checked_at.timestamp() * 1000)

    jobs: list[RlJob] = []
    errors: list[MonitorError] = []
    scope_user = None if args.all_users else args.user
    submitted_since_ms = None if args.hours == 0 else now_ms - int(args.hours * 3_600_000)
    for cluster in CLUSTERS:
        progress.phase(f"discovering RL jobs on {cluster.name}")
        try:
            found, discovery_errors = discover_rl_jobs(
                cluster,
                scope_user,
                submitted_since_ms=submitted_since_ms,
            )
        except Exception as error:
            found, discovery_errors = [], [str(error)]
        active_count = sum(not job.is_terminal for job in found)
        terminal_count = len(found) - active_count
        window_label = "all history" if args.hours == 0 else f"submitted in last {args.hours:g}h"
        progress.phase(
            f"discovery {cluster.name}: {active_count:,} active, {terminal_count:,} "
            f"succeeded/failed; {window_label}"
        )
        jobs.extend(found)
        errors.extend(
            _monitor_error(cluster.name, "job discovery", error)
            for error in discovery_errors
        )

    jobs = filter_records(jobs, filters, lambda job: job_filter_values(job, now_ms=now_ms))

    synced_jobs: list[tuple[RlJob, ArtifactResult, Path]] = []
    ordered_jobs = sorted(jobs, key=lambda item: (item.cluster.name, item.submitted_at_ms, item.job_id))
    for index, job in enumerate(ordered_jobs, start=1):
        try:
            progress.phase(f"job evidence {index}/{len(ordered_jobs)} {job.cluster.name}/{job.short_name}")
            artifacts, directory = sync_job(
                job,
                args.bundle_root,
                no_sync=args.no_sync,
                terminal_only=job.is_terminal,
                progress=progress,
            )
        except Exception as error:
            directory = job_directory(args.bundle_root, job)
            artifacts = ArtifactResult(
                "unavailable",
                "unavailable",
                "unavailable",
                "unavailable",
                None,
                None,
                (f"{type(error).__name__}: {error}",),
            )
        synced_jobs.append((job, artifacts, directory))

    active_job_directories = [
        (job, directory) for job, _, directory in synced_jobs if not job.is_terminal
    ]
    if not args.no_sync and active_job_directories:
        progress.phase("starting fleet-wide trace inventory and transfer")
        try:
            trace_statuses = sync_fleet_trace_jobs(
                active_job_directories,
                args.max_non_log_bytes,
                args.trace_sync_limit,
                progress,
            )
        except Exception as error:
            trace_statuses = {
                (job.cluster.name, job.job_id): (
                    "unavailable",
                    None,
                    None,
                    f"{type(error).__name__}: {error}",
                )
                for job, _directory in active_job_directories
            }
        synchronized: list[tuple[RlJob, ArtifactResult, Path]] = []
        for job, artifacts, directory in synced_jobs:
            if job.is_terminal:
                synchronized.append((job, artifacts, directory))
                continue
            traces, started, completed, trace_error = trace_statuses.get(
                (job.cluster.name, job.job_id),
                ("unavailable", None, None, "fleet trace sync returned no status"),
            )
            errors_for_job = artifacts.errors + ((f"trace sync: {trace_error}",) if trace_error else ())
            synchronized.append(
                (
                    job,
                    replace(
                        artifacts,
                        traces=traces,
                        trace_started=started,
                        trace_completed=completed,
                        errors=errors_for_job,
                    ),
                    directory,
                )
            )
        synced_jobs = synchronized
    elif not args.no_sync:
        progress.phase("no active RL jobs; trace inventory and transfer skipped")

    rows: list[list[object]] = []
    job_report: dict[str, Any] = {}
    for job, artifacts, directory in synced_jobs:
        scope = f"{job.cluster.name}/{job.job_id}"
        errors.extend(
            _monitor_error(scope, "artifact sync", error)
            for error in artifacts.errors
        )
        _step, _total, _metrics, parse_error = parse_metrics(directory / "finelog.log")
        if parse_error:
            errors.append(_monitor_error(scope, "Finelog parse", parse_error))
        signal = terminal_signal(directory / "finelog.log")
        if signal:
            errors.append(_monitor_error(scope, "workload signal", signal))
        try:
            write_job_manifest(
                job,
                args.bundle_root,
                directory,
                artifacts,
                args.max_non_log_bytes,
                args.trace_sync_limit,
            )
        except Exception as error:
            errors.append(_monitor_error(scope, "manifest write", error))
        try:
            rows.append(report_row(job, artifacts, directory))
        except Exception as error:
            errors.append(_monitor_error(scope, "row rendering", error))
            rows.append(
                [
                    f"{job.cluster.name}/{job.short_name}",
                    job.dataset,
                    _state_cell(job.state),
                    "—",
                    "—",
                    "—",
                    "—",
                    "unavailable",
                    StyledCell("status unavailable; see error report", "error"),
                ]
            )
        job_report[job.job_id] = {"cluster": job.cluster.name, "directory": str(directory), "artifacts": asdict(artifacts)}

    headers = ["Job", "Dataset", "State", "Step", "Reward", "Policy Loss", "Grad Norm", "Traces", "Trend"]
    table = (
        box_table(headers, rows)
        if rows
        else "No active or succeeded/failed CoreWeave Iris RL jobs in the selected window."
    )
    terminal_table = box_table(headers, rows, color=sys.stdout.isatty()) if rows else table
    filter_suffix = f"; filters={','.join(args.filter)}" if args.filter else ""
    window = "all" if args.hours == 0 else f"{args.hours:g}h"
    timestamp = checked_at.strftime("%Y%m%dT%H%M%SZ")
    error_report_path = write_error_report(
        report_directory,
        timestamp,
        "Iris CoreWeave RL monitor errors",
        checked_at,
        errors,
    )
    error_summary = f"Monitor errors: {len(errors)}; details: {error_report_path}"
    heading = f"# Iris CoreWeave RL status — {checked_at.isoformat()}; submitted={window}{filter_suffix}"
    report = f"{heading}\n\n{table}\n\n{error_summary}\n"
    report_path = report_directory / f"{timestamp}.md"
    report_path.write_text(report)
    (report_directory / "latest.md").write_text(report)
    write_json(
        report_directory / "latest.json",
        {
            "checked_at": checked_at.isoformat(),
            "jobs": job_report,
            "report": str(report_path),
            "error_count": len(errors),
            "error_report": str(error_report_path),
        },
    )
    progress.phase("report written; printing status table")
    print(f"{heading}\n\n{terminal_table}\n\n{error_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
