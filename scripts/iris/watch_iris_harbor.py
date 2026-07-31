#!/usr/bin/env python3
"""Sweep every active Iris Harbor datagen *and eval* job across Iris clusters.

With no arguments this discovers the current user's Harbor launch commands:
``run_tracegen.py`` for datagen and ``eval.local.run_eval`` for evals.  It reads
each job's recorded Harbor output location when available, counts direct trial
``result.json`` objects, and writes a durable box-table report.  Older eval
launches that only used pod-local ``trace_jobs`` remain visible as ``log-only``
rows instead of being silently omitted.  The monitor is read-only: it never
stops or relaunches a job.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.iris.coreweave_ops import (  # noqa: E402
    CLUSTERS as COREWEAVE_CLUSTERS,
    DEFAULT_MAX_VLLM_LOG_BYTES,
    command as run_kubectl_command,
    find_pod,
    iter_objects,
    kubectl_base,
    object_store_client,
    ray_log_inventory,
    save_ray_logs,
    split_s3_uri,
)
from scripts.iris.iris_ops import (  # noqa: E402
    DEFAULT_BUNDLE_ROOT,
    job_bundle,
    run_iris_command,
    write_bundle_manifest,
)


USER = "benjaminfeuer"
DEFAULT_STALL_MINUTES = 120
MEAN_PARSE_TAIL_BYTES = 8 * 1024 * 1024
STATE_NAMES = {
    1: "pending",
    2: "building",
    3: "running",
    4: "succeeded",
    5: "failed",
    6: "killed",
    7: "worker_failed",
    8: "unschedulable",
}
TERMINAL_STATES = {"succeeded", "failed", "killed", "worker_failed", "unschedulable"}
JOBS_DIR_PATTERN = re.compile(
    r"(?:--harbor_extra_arg=)?--jobs-dir(?:=|\s+)(s3://[^\s'\"\\]+|gs://[^\s'\"\\]+)"
)
JOB_NAME_PATTERN = re.compile(r"--job_name(?:=|\s+)([A-Za-z0-9._-]+)")
TASKS_INPUT_PATTERN = re.compile(r"--tasks_input_path(?:=|\s+)([^\s'\"\\]+)")
DATASET_PATH_PATTERN = re.compile(r"--dataset_path(?:=|\s+)([^\s'\"\\]+)")
DATASET_PATTERN = re.compile(r"--dataset(?:=|\s+)([^\s'\"\\]+)")
MEAN_PATTERN = re.compile(r"\d+/\d+ Mean: ([-0-9.]+)")
EVAL_LIVE_TRIAL_PATTERN = re.compile(
    r"\b(?P<trial>[A-Za-z0-9][A-Za-z0-9._-]*__[A-Za-z0-9]+): "
    r"(?:starting environment|running agent|running verifier)"
)


@dataclass(frozen=True)
class Cluster:
    name: str
    iris_bin: Path
    environment: dict[str, str | None]


@dataclass(frozen=True)
class HarborJob:
    cluster: Cluster
    job_id: str
    state: str
    task_state: str | None
    submitted_at_ms: int
    kind: str
    jobs_dir: str | None
    harbor_job_name: str | None
    dataset: str


@dataclass(frozen=True)
class Progress:
    completed: int | None
    total: int | None
    completion_source: str
    error: str | None = None
    mean_reward: float | None = None
    error_counts: dict[str, int] = field(default_factory=dict)
    exception_file_count: int | None = None


CLUSTERS = (
    Cluster(
        name="cw-rno2a",
        iris_bin=Path("/Users/benjaminfeuer/Documents/marin/.venv/bin/iris"),
        environment={"KUBECONFIG": None},
    ),
    Cluster(
        name="cw-us-east-02a",
        iris_bin=Path("/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris"),
        environment={"KUBECONFIG": "/Users/benjaminfeuer/.kube/coreweave-iris-gpu"},
    ),
    Cluster(
        name="marin",
        iris_bin=Path("/Users/benjaminfeuer/Documents/marin/.venv/bin/iris"),
        environment={"OAUTHLIB_RELAX_TOKEN_SCOPE": "1"},
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Root for canonical local Iris evidence bundles and Harbor reports.",
    )
    parser.add_argument("--stalled-after-minutes", type=int, default=DEFAULT_STALL_MINUTES)
    parser.add_argument("--job", help="Restrict the live report to one exact Iris job id.")
    parser.add_argument("--notify", action="store_true", help="Send a macOS notification on health changes.")
    return parser.parse_args()


def command_environment(cluster: Cluster) -> dict[str, str]:
    environment = os.environ.copy()
    for name, value in cluster.environment.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def run_iris(cluster: Cluster, arguments: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return run_iris_command(
        arguments,
        cluster=cluster.name,
        iris_bin=str(cluster.iris_bin),
        environment=command_environment(cluster),
        timeout=timeout,
    )


def entrypoint_text(raw: str) -> str:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(decoded)


def dataset_from_command(command: str) -> str:
    """Return a visible dataset identity for either Harbor job kind."""
    for pattern in (TASKS_INPUT_PATTERN, DATASET_PATH_PATTERN, DATASET_PATTERN):
        match = pattern.search(command)
        if match:
            return match.group(1)
    return "?"


def harbor_job_kind(command: str) -> str | None:
    """Classify a baked command without guessing from its Iris job name."""
    if "run_tracegen.py" in command:
        return "datagen"
    if "eval.local.run_eval" in command or "eval/local/run_eval.py" in command:
        return "eval"
    return None


def harbor_job_from_row(cluster: Cluster, row: dict[str, str]) -> HarborJob | None:
    """Parse one controller row into a supported Harbor job, if applicable.

    The identity fields are deliberately optional: pre-launcher evals used the
    Harbor YAML default ``trace_jobs`` directory and have no controller-visible
    durable path.  They still need a lifecycle/finelog row in the watcher.
    """
    command = entrypoint_text(row.get("entrypoint_json", ""))
    kind = harbor_job_kind(command)
    if kind is None:
        return None
    jobs_dir_match = JOBS_DIR_PATTERN.search(command)
    job_name_matches = JOB_NAME_PATTERN.findall(command)
    task_state_value = row.get("task_state")
    return HarborJob(
        cluster=cluster,
        job_id=row["job_id"],
        state=STATE_NAMES.get(int(row["state"]), f"state-{row['state']}"),
        task_state=(
            STATE_NAMES.get(int(task_state_value), f"state-{task_state_value}")
            if task_state_value
            else None
        ),
        submitted_at_ms=int(row["submitted_at_ms"]),
        kind=kind,
        jobs_dir=jobs_dir_match.group(1).rstrip("/") if jobs_dir_match else None,
        harbor_job_name=job_name_matches[-1] if job_name_matches else None,
        dataset=dataset_from_command(command),
    )


def discover_harbor_jobs(cluster: Cluster) -> tuple[list[HarborJob], list[str]]:
    sql = (
        "SELECT j.job_id, j.state, j.submitted_at_ms, jc.entrypoint_json, "
        "CASE "
        "WHEN EXISTS (SELECT 1 FROM tasks t WHERE t.job_id=j.job_id AND t.state=3) THEN 3 "
        "WHEN EXISTS (SELECT 1 FROM tasks t WHERE t.job_id=j.job_id AND t.state=2) THEN 2 "
        "WHEN EXISTS (SELECT 1 FROM tasks t WHERE t.job_id=j.job_id AND t.state=1) THEN 1 "
        "ELSE NULL END AS task_state "
        "FROM jobs j JOIN job_config jc ON j.job_id=jc.job_id "
        f"WHERE j.state IN (1,2,3) AND j.job_id LIKE '/{USER}/%' "
        "ORDER BY j.submitted_at_ms DESC"
    )
    result = run_iris(cluster, ["query", sql, "-f", "csv"])
    if result.returncode:
        message = (result.stderr or result.stdout).strip().replace("\n", " ")
        return [], [f"{cluster.name}: controller query failed: {message[-240:]}"]

    jobs: list[HarborJob] = []
    errors: list[str] = []
    for row in csv.DictReader(result.stdout.splitlines()):
        job = harbor_job_from_row(cluster, row)
        if job is not None:
            jobs.append(job)
    return jobs, errors


def display_state(job: HarborJob) -> str:
    """Return the worker-visible state instead of only the root job state."""
    if job.state in {"pending", "building"}:
        return "awaiting placement"
    if job.state == "running" and job.task_state in {"pending", "building"}:
        return "awaiting placement"
    return job.state


def finelog_path(job: HarborJob, bundle_root: Path) -> Path:
    return job_bundle(bundle_root, job.cluster.name, job.job_id).directory / "finelog.log"


def fetch_finelog(
    job: HarborJob,
    bundle_root: Path,
    since_ms: int,
) -> tuple[Path | None, str | None, int | None]:
    """Synchronize Finelog history, appending only the interval since the prior tick."""
    destination = finelog_path(job, bundle_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = run_iris(
        job.cluster,
        [
            "job",
            "logs",
            job.job_id,
            "--since-ms",
            str(since_ms),
            "--max-lines",
            "500000",
            "--no-tail",
        ],
        timeout=600,
    )
    if result.returncode:
        message = (result.stderr or result.stdout).strip().replace("\n", " ")
        return None, f"Finelog sync failed: {message[-180:]}", None
    write_mode = "a" if destination.exists() and since_ms > job.submitted_at_ms else "w"
    with destination.open(write_mode) as output:
        output.write(result.stdout)
    return destination, None, int(datetime.now(UTC).timestamp() * 1000)


def fetch_ray_vllm_logs(job: HarborJob, bundle_root: Path) -> tuple[str, str | None]:
    """Collect bounded live Ray worker and vLLM logs for a CoreWeave Harbor pod."""
    # An eval may use an external endpoint and has no local Ray/vLLM workload to
    # inspect. Its Finelog is still collected above; avoid treating an absent
    # worker pod as an error signal.
    if job.kind == "eval":
        return "not applicable (eval)", None
    if display_state(job) == "awaiting placement":
        return "awaiting placement", None
    if job.cluster.name not in COREWEAVE_CLUSTERS:
        return "not applicable", None
    cluster_config = COREWEAVE_CLUSTERS[job.cluster.name]
    base = kubectl_base(cluster_config, SimpleNamespace(kubeconfig=None, kube_context=None))
    destination = job_bundle(bundle_root, job.cluster.name, job.job_id).directory / "ray-vllm"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "logs").mkdir(exist_ok=True)
    try:
        pod = find_pod(base, SimpleNamespace(job=job.job_id, pod=None))
        task_log = run_kubectl_command(
            [*base, "-n", "iris", "logs", pod, "-c", "task", "--tail", "5000"],
            timeout=120,
        )
        (destination / "pod-task-tail.log").write_text(task_log)
        inventory = ray_log_inventory(base, pod, "task")
        saved, skipped = save_ray_logs(
            base,
            pod,
            "task",
            inventory,
            DEFAULT_MAX_VLLM_LOG_BYTES,
            destination / "logs",
            incremental=True,
        )
    except Exception as error:
        return "unavailable", f"Ray/vLLM sync failed: {error}"
    manifest = {
        "synced_at": datetime.now(UTC).isoformat(),
        "job": job.job_id,
        "pod": pod,
        "saved": saved,
        "skipped": skipped,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return f"{len(saved)} saved, {len(skipped)} skipped", None


def read_gcs_progress(job: HarborJob, artifact_dir: Path) -> Progress:
    assert job.jobs_dir is not None and job.harbor_job_name is not None
    root = f"{job.jobs_dir}/{job.harbor_job_name}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    aggregate = subprocess.run(
        ["gcloud", "storage", "cat", f"{root}/result.json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    total: int | None = None
    if aggregate.returncode == 0:
        try:
            aggregate_path = artifact_dir / "result.json"
            aggregate_path.write_text(aggregate.stdout)
            total = int(json.loads(aggregate_path.read_text()).get("n_total_trials") or 0) or None
        except json.JSONDecodeError:
            pass

    if aggregate.returncode:
        message = (aggregate.stderr or aggregate.stdout).strip().replace("\n", " ")
        return Progress(None, total, "Harbor aggregate", f"GCS aggregate fetch failed: {message[-180:]}")
    aggregate_data = json.loads((artifact_dir / "result.json").read_text())
    progress = progress_from_harbor_aggregate(aggregate_data, "Harbor aggregate")
    (artifact_dir / "completion-source.txt").write_text(
        "completed comes from Harbor stats.n_completed_trials in the local aggregate result.json\n"
    )
    return Progress(
        progress.completed,
        total,
        progress.completion_source,
        mean_reward=progress.mean_reward,
        error_counts=progress.error_counts,
    )


def coreweave_client(cluster: Cluster) -> Any:
    config = COREWEAVE_CLUSTERS[cluster.name]
    base = kubectl_base(config, SimpleNamespace(kubeconfig=None, kube_context=None))
    return object_store_client(base, config)


def s3_trial_artifacts(client: Any, bucket: str, prefix: str) -> tuple[list[str], int]:
    """Return direct completed-trial names and standalone exception-file count."""
    root = f"{prefix.rstrip('/')}/"
    completed: set[str] = set()
    exception_files = 0
    for item in iter_objects(client, bucket, root):
        relative = item["Key"].removeprefix(root)
        if relative.count("/") != 1:
            continue
        if relative.endswith("/result.json"):
            completed.add(relative.split("/", 1)[0])
        elif relative.endswith("/exception.txt"):
            exception_files += 1
    return sorted(completed), exception_files


def progress_from_harbor_aggregate(aggregate: dict[str, Any], source: str) -> Progress:
    """Extract count, mean reward, and typed Harbor errors from an aggregate result."""
    stats = aggregate.get("stats", {})
    error_counts: dict[str, int] = {}
    weighted_mean = 0.0
    mean_weight = 0
    for evaluation in stats.get("evals", {}).values():
        exception_stats = evaluation.get("exception_stats", {})
        for error_name, trial_ids in exception_stats.items():
            error_counts[error_name] = error_counts.get(error_name, 0) + len(trial_ids)
        trial_count = int(evaluation.get("n_trials") or 0)
        for metric in evaluation.get("metrics", []):
            mean = metric.get("mean")
            if isinstance(mean, (int, float)) and trial_count:
                weighted_mean += float(mean) * trial_count
                mean_weight += trial_count
                break
    errored_trials = int(stats.get("n_errored_trials") or 0)
    typed_errors = sum(error_counts.values())
    if errored_trials > typed_errors:
        error_counts["other Harbor errors"] = errored_trials - typed_errors
    return Progress(
        int(stats.get("n_completed_trials") or 0),
        int(aggregate.get("n_total_trials") or 0) or None,
        source,
        mean_reward=weighted_mean / mean_weight if mean_weight else None,
        error_counts=error_counts,
    )


def read_s3_progress(job: HarborJob, client: Any, artifact_dir: Path) -> Progress:
    assert job.jobs_dir is not None and job.harbor_job_name is not None
    bucket, prefix = split_s3_uri(f"{job.jobs_dir}/{job.harbor_job_name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    completed_trial_names, exception_file_count = s3_trial_artifacts(client, bucket, prefix)
    (artifact_dir / "trial-result-keys.txt").write_text("\n".join(completed_trial_names) + "\n")
    completed = len(completed_trial_names)
    try:
        response = client.get_object(Bucket=bucket, Key=f"{prefix}/result.json")
        aggregate_path = artifact_dir / "result.json"
        aggregate_path.write_bytes(response["Body"].read())
        aggregate = json.loads(aggregate_path.read_text())
    except Exception as error:
        return Progress(
            completed,
            None,
            "direct result.json",
            f"S3 aggregate result unreadable: {error}",
            error_counts={"exception.txt": exception_file_count} if exception_file_count else {},
            exception_file_count=exception_file_count,
        )
    aggregate_progress = progress_from_harbor_aggregate(aggregate, "direct result.json")
    error_counts = aggregate_progress.error_counts
    if not error_counts and exception_file_count:
        error_counts = {"exception.txt": exception_file_count}
    return Progress(
        completed,
        aggregate_progress.total,
        aggregate_progress.completion_source,
        mean_reward=aggregate_progress.mean_reward,
        error_counts=error_counts,
        exception_file_count=exception_file_count,
    )


def read_pod_local_eval_progress(job: HarborJob, bundle_root: Path) -> Progress:
    """Read the current Harbor aggregate from a legacy eval's live worker pod.

    Older Iris eval launches left ``trace_jobs`` under ``/app`` rather than a
    durable ``--jobs-dir``.  The running pod is still authoritative while the
    job is live, so use its newest Harbor run directory and preserve the exact
    aggregate in the shared evidence bundle.  This is intentionally eval-only:
    datagen must have a durable output identity.
    """
    if job.kind != "eval" or job.cluster.name not in COREWEAVE_CLUSTERS:
        raise LookupError("pod-local Harbor aggregates are only available for CoreWeave evals")
    cluster_config = COREWEAVE_CLUSTERS[job.cluster.name]
    base = kubectl_base(cluster_config, SimpleNamespace(kubeconfig=None, kube_context=None))
    pod = find_pod(base, SimpleNamespace(job=job.job_id, pod=None))
    result_text = run_kubectl_command(
        [
            *base,
            "-n",
            "iris",
            "exec",
            pod,
            "-c",
            "task",
            "--",
            "sh",
            "-lc",
            "latest=$(ls -dt /app/trace_jobs/*/ 2>/dev/null | head -n 1); "
            "test -n \"$latest\"; cat \"${latest}result.json\"",
        ],
        timeout=120,
    )
    aggregate = json.loads(result_text)
    artifact_dir = job_bundle(bundle_root, job.cluster.name, job.job_id).directory / "harbor"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "result.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (artifact_dir / "completion-source.txt").write_text(
        f"pod-local Harbor aggregate from {pod} at {datetime.now(UTC).isoformat()}\n"
    )
    return progress_from_harbor_aggregate(aggregate, "pod-local Harbor aggregate")


def mean_reward(local_log: Path | None) -> str:
    if local_log is None:
        return "—"
    with local_log.open("rb") as log_file:
        log_file.seek(max(0, local_log.stat().st_size - MEAN_PARSE_TAIL_BYTES))
        recent_log = log_file.read().decode(errors="replace")
    matches = MEAN_PATTERN.findall(recent_log)
    return matches[-1] if matches else "—"


def finelog_activity(local_log: Path | None) -> str:
    """Describe live eval work when an older launch has only pod-local traces."""
    if local_log is None:
        return "finelog unavailable"
    with local_log.open("rb") as log_file:
        log_file.seek(max(0, local_log.stat().st_size - MEAN_PARSE_TAIL_BYTES))
        recent_log = log_file.read().decode(errors="replace")
    active_trials = {match.group("trial") for match in EVAL_LIVE_TRIAL_PATTERN.finditer(recent_log)}
    if active_trials:
        return f"finelog ({len(active_trials)} recent trial ID{'s' if len(active_trials) != 1 else ''})"
    return "finelog (no current trial line)"


def format_error_counts(progress: Progress) -> str:
    """Render compact typed error counts, with exception.txt as a fallback source."""
    if not progress.error_counts:
        return "—"
    ranked = sorted(progress.error_counts.items(), key=lambda item: (-item[1], item[0]))
    displayed = ranked[:3]
    detail = ", ".join(f"{name} {count}" for name, count in displayed)
    if len(ranked) > len(displayed):
        detail += f", +{len(ranked) - len(displayed)} types"
    return f"{sum(progress.error_counts.values())}: {detail}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def health_label(
    job: HarborJob,
    progress: Progress,
    previous: dict[str, Any],
    checked_at: datetime,
    stalled_after_minutes: int,
) -> tuple[str, str]:
    prior = previous.get("jobs", {}).get(job.job_id, {})
    if display_state(job) == "awaiting placement":
        return "awaiting placement", prior.get("last_advanced_at", checked_at.isoformat())
    if progress.error:
        return "output-unavailable", checked_at.isoformat()
    if job.state in TERMINAL_STATES:
        return f"terminal ({job.state})", prior.get("last_advanced_at", checked_at.isoformat())
    if progress.completed is None:
        return "awaiting output", prior.get("last_advanced_at", checked_at.isoformat())
    if prior.get("completed") is None or progress.completed > prior["completed"]:
        return ("advancing" if prior else "baseline"), checked_at.isoformat()
    last_advanced_at = prior.get("last_advanced_at", checked_at.isoformat())
    idle_minutes = (checked_at - datetime.fromisoformat(last_advanced_at)).total_seconds() / 60
    if job.state == "running" and idle_minutes >= stalled_after_minutes:
        return f"stalled ({idle_minutes:.0f}m)", last_advanced_at
    return f"no new result ({idle_minutes:.0f}m)", last_advanced_at


def box_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def border(left: str, middle: str, right: str, fill: str) -> str:
        return left + middle.join(fill * (width + 2) for width in widths) + right

    def line(values: list[str]) -> str:
        return "│" + "│".join(f" {value.ljust(width)} " for value, width in zip(values, widths)) + "│"

    return "\n".join(
        [
            border("┌", "┬", "┐", "─"),
            line(headers),
            border("├", "┼", "┤", "─"),
            *(line(row) for row in rows),
            border("└", "┴", "┘", "─"),
        ]
    )


def notify(message: str) -> None:
    subprocess.run(
        ["osascript", "-e", f'display notification "{message.replace(chr(34), chr(92) + chr(34))}" with title "Iris Harbor monitor"'],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    args = parse_args()
    if args.stalled_after_minutes <= 0:
        raise ValueError("--stalled-after-minutes must be positive")
    args.bundle_root.mkdir(parents=True, exist_ok=True)
    report_directory = args.bundle_root / "reports" / "harbor"
    report_directory.mkdir(parents=True, exist_ok=True)
    latest_path = report_directory / "latest.json"
    previous = load_json(latest_path)
    checked_at = datetime.now(UTC)

    jobs: list[HarborJob] = []
    errors: list[str] = []
    for cluster in CLUSTERS:
        found, cluster_errors = discover_harbor_jobs(cluster)
        jobs.extend(found)
        errors.extend(cluster_errors)
    if args.job:
        jobs = [job for job in jobs if job.job_id == args.job]
        if not jobs:
            errors.append(f"No active Harbor job with id {args.job!r} was discovered.")

    s3_clients: dict[str, Any] = {}
    local_logs: dict[str, tuple[Path | None, str | None, int | None]] = {}
    for job in jobs:
        prior = previous.get("jobs", {}).get(job.job_id, {})
        prior_sync = prior.get("finelog_synced_at_ms")
        destination = finelog_path(job, args.bundle_root)
        if prior_sync is None and destination.exists() and previous.get("checked_at"):
            prior_sync = int(datetime.fromisoformat(previous["checked_at"]).timestamp() * 1000)
        local_logs[job.job_id] = fetch_finelog(
            job,
            args.bundle_root,
            int(prior_sync) if prior_sync is not None else job.submitted_at_ms,
        )
    ray_vllm_logs = {job.job_id: fetch_ray_vllm_logs(job, args.bundle_root) for job in jobs}
    rows: list[list[str]] = []
    current_jobs: dict[str, Any] = {}
    for job in sorted(jobs, key=lambda item: (item.cluster.name, item.job_id)):
        state = display_state(job)
        try:
            artifact_dir = job_bundle(args.bundle_root, job.cluster.name, job.job_id).directory / "harbor"
            if job.jobs_dir is None or job.harbor_job_name is None:
                if job.kind == "eval":
                    progress = read_pod_local_eval_progress(job, args.bundle_root)
                else:
                    local_log, _log_error, _synced_at = local_logs[job.job_id]
                    progress = Progress(None, None, finelog_activity(local_log))
            elif job.jobs_dir.startswith("s3://"):
                client = s3_clients.setdefault(job.cluster.name, coreweave_client(job.cluster))
                progress = read_s3_progress(job, client, artifact_dir)
            else:
                progress = read_gcs_progress(job, artifact_dir)
        except Exception as error:
            progress = Progress(None, None, "unavailable", f"progress read failed: {error}")
        health, last_advanced_at = health_label(
            job, progress, previous, checked_at, args.stalled_after_minutes
        )
        completed = "?" if progress.completed is None else f"{progress.completed:,}"
        total = "?" if progress.total is None else f"{progress.total:,}"
        remaining = "?" if progress.completed is None or progress.total is None else f"{max(0, progress.total - progress.completed):,}"
        local_log, log_error, finelog_synced_at_ms = local_logs[job.job_id]
        ray_vllm_status, ray_vllm_error = ray_vllm_logs[job.job_id]
        mean = f"{progress.mean_reward:.3f}" if progress.mean_reward is not None else mean_reward(local_log)
        error_counts = format_error_counts(progress)
        trend = health if progress.error is None or state == "awaiting placement" else f"{health}: {progress.error[-100:]}"
        log_status = "synced" if local_log is not None else f"error: {log_error[-70:] if log_error else 'unknown'}"
        ray_vllm_status = ray_vllm_status if ray_vllm_error is None else f"error: {ray_vllm_error[-70:]}"
        rows.append([job.cluster.name, job.kind, job.job_id.rsplit("/", 1)[-1], job.dataset, state, f"{completed}/{total}", remaining, progress.completion_source, mean, error_counts, log_status, ray_vllm_status, trend])
        current_jobs[job.job_id] = {
            "cluster": job.cluster.name,
            "job_kind": job.kind,
            "state": state,
            "controller_state": job.state,
            "task_state": job.task_state,
            "completed": progress.completed,
            "total": progress.total,
            "mean_reward": progress.mean_reward,
            "error_counts": progress.error_counts,
            "exception_file_count": progress.exception_file_count,
            "last_advanced_at": last_advanced_at,
            "health": health,
            "jobs_dir": job.jobs_dir,
            "harbor_job_name": job.harbor_job_name,
            "dataset": job.dataset,
            "finelog_synced_at_ms": finelog_synced_at_ms,
            "ray_vllm_status": ray_vllm_status,
            "bundle_directory": str(job_bundle(args.bundle_root, job.cluster.name, job.job_id).directory),
        }
        write_bundle_manifest(
            job_bundle(args.bundle_root, job.cluster.name, job.job_id),
            {
                "kind": "harbor",
                "job_kind": job.kind,
                "dataset": job.dataset,
                "harbor_job_name": job.harbor_job_name,
                "jobs_dir": job.jobs_dir,
                "state": state,
                "controller_state": job.state,
                "task_state": job.task_state,
                "submitted_at_ms": job.submitted_at_ms,
                "last_synced_at": checked_at.isoformat(),
                "progress": {
                    "completed": progress.completed,
                    "total": progress.total,
                    "mean_reward": progress.mean_reward,
                    "error_counts": progress.error_counts,
                    "exception_file_count": progress.exception_file_count,
                },
            },
        )

    if rows:
        table = box_table(["Cluster", "Kind", "Harbor run", "Dataset", "State", "Trials", "Remaining", "Count source", "Mean", "Errors", "Finelog", "Ray/vLLM", "Trend"], rows)
    else:
        table = "No active Iris Harbor datagen or eval jobs discovered."
    report = f"# Iris Harbor datagen / eval status — {checked_at.isoformat()}\n\n{table}\n"
    if errors:
        report += "\n## Monitor errors\n\n" + "\n".join(f"- {error}" for error in errors) + "\n"
    timestamp = checked_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = report_directory / f"{timestamp}.md"
    report_path.write_text(report)
    (report_directory / "latest.md").write_text(report)
    current = {"checked_at": checked_at.isoformat(), "jobs": current_jobs, "report": str(report_path)}
    latest_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    if args.notify:
        changed = [job_id for job_id, data in current_jobs.items() if previous.get("jobs", {}).get(job_id, {}).get("health") != data["health"]]
        if changed:
            notify(f"{len(changed)} Harbor health change(s); report saved to {report_directory / 'latest.md'}")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
