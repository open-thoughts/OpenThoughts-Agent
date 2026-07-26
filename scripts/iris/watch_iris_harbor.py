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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from google.cloud import storage as gcs_storage


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
    MonitorError,
    StyledCell,
    box_table,
    filter_records,
    format_duration,
    job_bundle,
    parse_regex_filters,
    run_iris_command,
    write_error_report,
    write_bundle_manifest,
)


USER = "benjaminfeuer"
DEFAULT_STALL_MINUTES = 120
TRACE_TREND_HOURS = 2
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
MODEL_PATTERN = re.compile(r"--model(?:=|\s+)([^\s'\"\\]+)")
CONCURRENCY_PATTERN = re.compile(r"--n_concurrent(?:=|\s+)(\d+)")
GPU_COUNT_PATTERN = re.compile(r"--gpus(?:=|\s+)(\d+)")
GPU_SHAPE_PATTERN = re.compile(r"--gpu(?:=|\s+)H100x(\d+)", re.IGNORECASE)
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
    submitted_at_ms: int
    kind: str
    jobs_dir: str | None
    harbor_job_name: str | None
    dataset: str
    model: str | None = None
    n_concurrent: int | None = None
    gpu_count: int | None = None


@dataclass(frozen=True)
class Progress:
    completed: int | None
    total: int | None
    completion_source: str
    error: str | None = None
    mean_reward: float | None = None
    error_counts: dict[str, int] = field(default_factory=dict)
    exception_file_count: int | None = None
    recent_completed: int | None = None
    recent_errored: int | None = None
    recent_window_error: str | None = None


@dataclass(frozen=True)
class TrialArtifacts:
    completed_names: list[str]
    exception_file_count: int
    recent_completed: int
    recent_errored: int


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
    parser.add_argument(
        "--stalled-after-minutes", type=int, default=DEFAULT_STALL_MINUTES
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Only include jobs submitted within this many hours; 0 means all history (default: 24).",
    )
    parser.add_argument(
        "--job", help="Restrict the live report to one exact Iris job id."
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=REGEX",
        help=(
            "Keep jobs matching every case-insensitive regex filter. Available keys: "
            "cluster, job, name, dataset, kind, state, submitted, duration."
        ),
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send a macOS notification on health changes.",
    )
    return parser.parse_args()


def command_environment(cluster: Cluster) -> dict[str, str]:
    environment = os.environ.copy()
    for name, value in cluster.environment.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def run_iris(
    cluster: Cluster, arguments: list[str], *, timeout: int = 180
) -> subprocess.CompletedProcess[str]:
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
    model_match = MODEL_PATTERN.search(command)
    concurrency_match = CONCURRENCY_PATTERN.search(command)
    gpu_match = GPU_COUNT_PATTERN.search(command) or GPU_SHAPE_PATTERN.search(command)
    return HarborJob(
        cluster=cluster,
        job_id=row["job_id"],
        state=STATE_NAMES.get(int(row["state"]), f"state-{row['state']}"),
        submitted_at_ms=int(row["submitted_at_ms"]),
        kind=kind,
        jobs_dir=jobs_dir_match.group(1).rstrip("/") if jobs_dir_match else None,
        harbor_job_name=job_name_matches[-1] if job_name_matches else None,
        dataset=dataset_from_command(command),
        model=model_match.group(1) if model_match else None,
        n_concurrent=int(concurrency_match.group(1)) if concurrency_match else None,
        gpu_count=int(gpu_match.group(1)) if gpu_match else None,
    )


def discover_harbor_jobs(
    cluster: Cluster, *, submitted_since_ms: int | None = None
) -> tuple[list[HarborJob], list[str]]:
    submitted_clause = (
        ""
        if submitted_since_ms is None
        else f" AND j.submitted_at_ms >= {submitted_since_ms}"
    )
    sql = (
        "SELECT j.job_id, j.state, j.submitted_at_ms, jc.entrypoint_json "
        "FROM jobs j JOIN job_config jc ON j.job_id=jc.job_id "
        f"WHERE j.state IN (1,2,3) AND j.job_id LIKE '/{USER}/%'{submitted_clause} "
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


def finelog_path(job: HarborJob, bundle_root: Path) -> Path:
    return (
        job_bundle(bundle_root, job.cluster.name, job.job_id).directory / "finelog.log"
    )


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
    if job.cluster.name not in COREWEAVE_CLUSTERS:
        return "not applicable", None
    cluster_config = COREWEAVE_CLUSTERS[job.cluster.name]
    base = kubectl_base(
        cluster_config, SimpleNamespace(kubeconfig=None, kube_context=None)
    )
    destination = (
        job_bundle(bundle_root, job.cluster.name, job.job_id).directory / "ray-vllm"
    )
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
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return f"{len(saved)} saved, {len(skipped)} skipped", None


def _object_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _trial_artifacts(
    objects: list[tuple[str, datetime | None]], root: str, cutoff: datetime
) -> TrialArtifacts:
    prefix = f"{root.rstrip('/')}/"
    completed: set[str] = set()
    recent_completed: set[str] = set()
    errored: set[str] = set()
    exception_file_count = 0
    for key, modified_at in objects:
        relative = key.removeprefix(prefix)
        if relative == key or relative.count("/") != 1:
            continue
        trial_name, filename = relative.split("/", 1)
        if filename == "result.json":
            completed.add(trial_name)
            if modified_at is not None and modified_at >= cutoff:
                recent_completed.add(trial_name)
        elif filename == "exception.txt":
            errored.add(trial_name)
            exception_file_count += 1
    return TrialArtifacts(
        sorted(completed),
        exception_file_count,
        len(recent_completed),
        len(recent_completed & errored),
    )


def gcs_trial_artifacts(client: Any, root: str, cutoff: datetime) -> TrialArtifacts:
    location = root.removeprefix("gs://")
    bucket, object_prefix = location.split("/", 1)
    object_prefix = object_prefix.rstrip("/")
    objects: list[tuple[str, datetime | None]] = []
    for filename in ("result.json", "exception.txt"):
        blobs = client.list_blobs(
            bucket,
            prefix=f"{object_prefix}/",
            match_glob=f"{object_prefix}/*/{filename}",
            fields="items(name,timeCreated),nextPageToken",
        )
        objects.extend((blob.name, _object_time(blob.time_created)) for blob in blobs)
    return _trial_artifacts(objects, object_prefix, cutoff)


def read_gcs_progress(
    job: HarborJob, client: Any, artifact_dir: Path, cutoff: datetime
) -> Progress:
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
            total = (
                int(json.loads(aggregate_path.read_text()).get("n_total_trials") or 0)
                or None
            )
        except json.JSONDecodeError:
            pass

    if aggregate.returncode:
        message = (aggregate.stderr or aggregate.stdout).strip().replace("\n", " ")
        return Progress(
            None,
            total,
            "Harbor aggregate",
            f"GCS aggregate fetch failed: {message[-180:]}",
        )
    aggregate_data = json.loads((artifact_dir / "result.json").read_text())
    progress = progress_from_harbor_aggregate(aggregate_data, "Harbor aggregate")
    artifacts: TrialArtifacts | None = None
    window_error: str | None = None
    try:
        artifacts = gcs_trial_artifacts(client, root, cutoff)
    except Exception as error:
        window_error = str(error)
    (artifact_dir / "completion-source.txt").write_text(
        "completed comes from Harbor stats.n_completed_trials in the local aggregate result.json\n"
    )
    return Progress(
        progress.completed,
        total,
        progress.completion_source,
        mean_reward=progress.mean_reward,
        error_counts=progress.error_counts,
        exception_file_count=artifacts.exception_file_count if artifacts else None,
        recent_completed=artifacts.recent_completed if artifacts else None,
        recent_errored=artifacts.recent_errored if artifacts else None,
        recent_window_error=window_error,
    )


def coreweave_client(cluster: Cluster) -> Any:
    config = COREWEAVE_CLUSTERS[cluster.name]
    base = kubectl_base(config, SimpleNamespace(kubeconfig=None, kube_context=None))
    return object_store_client(base, config)


def s3_trial_artifacts(
    client: Any, bucket: str, prefix: str, cutoff: datetime
) -> TrialArtifacts:
    """Return direct trial artifacts, including the exact recent error intersection."""
    root = f"{prefix.rstrip('/')}/"
    objects: list[tuple[str, datetime | None]] = []
    for item in iter_objects(client, bucket, root):
        objects.append((item["Key"], _object_time(item.get("LastModified"))))
    return _trial_artifacts(objects, prefix.rstrip("/"), cutoff)


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


def read_s3_progress(
    job: HarborJob, client: Any, artifact_dir: Path, cutoff: datetime
) -> Progress:
    assert job.jobs_dir is not None and job.harbor_job_name is not None
    bucket, prefix = split_s3_uri(f"{job.jobs_dir}/{job.harbor_job_name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts = s3_trial_artifacts(client, bucket, prefix, cutoff)
    completed_trial_names = artifacts.completed_names
    exception_file_count = artifacts.exception_file_count
    (artifact_dir / "trial-result-keys.txt").write_text(
        "\n".join(completed_trial_names) + "\n"
    )
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
            error_counts={"exception.txt": exception_file_count}
            if exception_file_count
            else {},
            exception_file_count=exception_file_count,
            recent_completed=artifacts.recent_completed,
            recent_errored=artifacts.recent_errored,
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
        recent_completed=artifacts.recent_completed,
        recent_errored=artifacts.recent_errored,
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
        raise LookupError(
            "pod-local Harbor aggregates are only available for CoreWeave evals"
        )
    cluster_config = COREWEAVE_CLUSTERS[job.cluster.name]
    base = kubectl_base(
        cluster_config, SimpleNamespace(kubeconfig=None, kube_context=None)
    )
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
            'test -n "$latest"; cat "${latest}result.json"',
        ],
        timeout=120,
    )
    aggregate = json.loads(result_text)
    artifact_dir = (
        job_bundle(bundle_root, job.cluster.name, job.job_id).directory / "harbor"
    )
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
    active_trials = {
        match.group("trial") for match in EVAL_LIVE_TRIAL_PATTERN.finditer(recent_log)
    }
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


# A single H100x8 node sustains about 55 completed GLM-5.2 Terminus traces in
# two hours at concurrency four.  This is deliberately a capacity reference,
# not a generic per-dataset speed limit: long-horizon tasks and smaller models
# need their own measured profiles before the watcher applies a floor to them.
GLM52_REFERENCE_TRACES_2H = 55
GLM52_REFERENCE_CONCURRENCY = 4
GLM52_REFERENCE_H100S = 8
TOLERABLE_ERROR_RATE = 0.25
FAILING_ERROR_RATE = 0.75


def capacity_floor_2h(job: HarborJob) -> int | None:
    """Return a conservative measured throughput floor for a known model/node shape.

    The floor scales the GLM-5.2 single-H100x8 reference by the requested
    Harbor concurrency and available H100 FLOPS.  Returning ``None`` keeps
    unfamiliar models on the existing error-only classification rather than
    inventing a misleading universal throughput target.
    """
    identity = " ".join(
        value for value in (job.job_id, job.harbor_job_name, job.model) if value
    ).lower()
    if "glm52" not in identity and "glm-5.2" not in identity:
        return None
    # The 355B MoE saturates one H100x8 node at roughly four agent slots.
    # Higher configured Harbor concurrency adds queued work, not usable model
    # FLOPS, so it must not inflate the health floor.
    concurrency = min(
        job.n_concurrent or GLM52_REFERENCE_CONCURRENCY, GLM52_REFERENCE_CONCURRENCY
    )
    h100s = job.gpu_count or GLM52_REFERENCE_H100S
    reference = GLM52_REFERENCE_TRACES_2H * concurrency / GLM52_REFERENCE_CONCURRENCY
    expected = reference * h100s / GLM52_REFERENCE_H100S
    # Use 75% of observed steady-state capacity: it catches a real slowdown
    # without incorrectly flagging normal task-length variance as degraded.
    return max(1, round(expected * 0.75))


def recent_window_is_healthy(job: HarborJob, progress: Progress) -> bool:
    if progress.recent_completed is None or progress.recent_errored is None:
        return False
    if progress.recent_completed == 0:
        return False
    floor = capacity_floor_2h(job)
    if floor is None:
        return False
    error_rate = progress.recent_errored / progress.recent_completed
    return error_rate <= TOLERABLE_ERROR_RATE and progress.recent_completed >= floor


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def health_label(
    job: HarborJob,
    progress: Progress,
    previous: dict[str, Any],
    checked_at: datetime,
    stalled_after_minutes: int,
) -> tuple[str, str]:
    prior = previous.get("jobs", {}).get(job.job_id, {})
    if progress.error:
        return "output-unavailable", checked_at.isoformat()
    if job.state in TERMINAL_STATES:
        return f"terminal ({job.state})", prior.get(
            "last_advanced_at", checked_at.isoformat()
        )
    if progress.completed is None:
        return "awaiting output", prior.get("last_advanced_at", checked_at.isoformat())
    if progress.recent_completed is not None and progress.recent_errored is not None:
        recent_completed = progress.recent_completed
        recent_errored = progress.recent_errored
        if recent_completed:
            detail = f"+{recent_completed}/2h; {recent_errored} errors"
            floor = capacity_floor_2h(job)
            error_rate = recent_errored / recent_completed
            if error_rate >= FAILING_ERROR_RATE:
                return f"failing ({detail})", checked_at.isoformat()
            if recent_errored == 0:
                return f"advancing ({detail})", checked_at.isoformat()
            if recent_window_is_healthy(job, progress):
                floor_detail = f"; FLOPS floor {floor}/2h" if floor is not None else ""
                return f"healthy ({detail}{floor_detail})", checked_at.isoformat()
            return f"degraded ({detail})", checked_at.isoformat()
        job_age = checked_at - datetime.fromtimestamp(job.submitted_at_ms / 1000, UTC)
        if job.state == "running" and job_age >= timedelta(hours=TRACE_TREND_HOURS):
            return "stalled (+0 traces/2h)", prior.get(
                "last_advanced_at", checked_at.isoformat()
            )
        return "warming up (+0 traces/2h)", prior.get(
            "last_advanced_at", checked_at.isoformat()
        )
    if prior.get("completed") is None or progress.completed > prior["completed"]:
        return ("advancing" if prior else "baseline"), checked_at.isoformat()
    last_advanced_at = prior.get("last_advanced_at", checked_at.isoformat())
    idle_minutes = (
        checked_at - datetime.fromisoformat(last_advanced_at)
    ).total_seconds() / 60
    if job.state == "running" and idle_minutes >= stalled_after_minutes:
        return f"stalled ({idle_minutes:.0f}m)", last_advanced_at
    return f"no new result ({idle_minutes:.0f}m)", last_advanced_at


def job_filter_values(job: HarborJob, *, now_ms: int) -> dict[str, str]:
    """Return the pre-sync Harbor job fields available to ``--filter``."""
    return {
        "cluster": job.cluster.name,
        "job": job.job_id,
        "name": job.job_id.rsplit("/", 1)[-1],
        "dataset": job.dataset,
        "kind": job.kind,
        "state": job.state,
        "submitted": datetime.fromtimestamp(job.submitted_at_ms / 1000, UTC).strftime(
            "%m-%d %H:%M"
        ),
        "duration": format_duration(job.submitted_at_ms, now_ms=now_ms),
    }


def notify(message: str) -> None:
    subprocess.run(
        [
            "osascript",
            "-e",
            f'display notification "{message.replace(chr(34), chr(92) + chr(34))}" with title "Iris Harbor monitor"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _monitor_error(scope: str, operation: str, error: object) -> MonitorError:
    message = str(error).strip() or type(error).__name__
    return MonitorError(scope, operation, message)


def _state_cell(state: str) -> StyledCell:
    if state in {"running", "succeeded"}:
        tone = "success"
    elif state in {"pending", "building", "unspecified"}:
        tone = "warning"
    else:
        tone = "error"
    return StyledCell(state, tone)


def _health_cell(health: str) -> StyledCell:
    if health in {"advancing", "baseline", "healthy"} or health.startswith(
        ("advancing (", "healthy (")
    ):
        tone = "success"
    elif health.startswith(
        (
            "stalled",
            "failing",
            "output-unavailable",
            "terminal (failed",
            "terminal (worker_failed",
        )
    ):
        tone = "error"
    else:
        tone = "warning"
    return StyledCell(health, tone)


def recent_trend_cell(job: HarborJob, progress: Progress) -> StyledCell:
    if progress.recent_completed is None or progress.recent_errored is None:
        return StyledCell("unavailable", "muted")
    completed = progress.recent_completed
    errored = progress.recent_errored
    if completed == 0:
        return StyledCell("+0 traces; 0 errors", "warning")
    rate = errored / completed
    text = f"+{completed:,} traces; {errored:,} errors ({rate:.0%})"
    if errored == 0 or recent_window_is_healthy(job, progress):
        tone = "success"
    elif errored / completed >= FAILING_ERROR_RATE:
        tone = "error"
    else:
        tone = "warning"
    return StyledCell(text, tone)


def report_row(
    job: HarborJob,
    progress: Progress,
    health: str,
    local_log: Path | None,
    ray_vllm_status: str,
) -> list[object]:
    """Build one bounded status row without embedding monitor exception text."""
    completed = "?" if progress.completed is None else f"{progress.completed:,}"
    total = "?" if progress.total is None else f"{progress.total:,}"
    try:
        mean = (
            f"{progress.mean_reward:.3f}"
            if progress.mean_reward is not None
            else mean_reward(local_log)
        )
    except OSError:
        mean = "—"
    trial_errors = format_error_counts(progress)
    evidence = f"Finelog {'synced' if local_log is not None else 'unavailable'}; Ray/vLLM {ray_vllm_status}"
    return [
        f"{job.cluster.name}/{job.kind}",
        job.job_id.rsplit("/", 1)[-1],
        job.dataset,
        _state_cell(job.state),
        f"{completed}/{total}",
        mean,
        StyledCell(trial_errors, "warning" if trial_errors != "—" else "muted"),
        recent_trend_cell(job, progress),
        StyledCell(evidence, "warning" if "unavailable" in evidence else "muted"),
        _health_cell(health),
    ]


def main() -> int:
    args = parse_args()
    if args.stalled_after_minutes <= 0:
        raise ValueError("--stalled-after-minutes must be positive")
    if args.hours < 0:
        raise ValueError("--hours must be non-negative")
    filters = parse_regex_filters(
        args.filter,
        {"cluster", "job", "name", "dataset", "kind", "state", "submitted", "duration"},
    )
    args.bundle_root.mkdir(parents=True, exist_ok=True)
    report_directory = args.bundle_root / "reports" / "harbor"
    report_directory.mkdir(parents=True, exist_ok=True)
    latest_path = report_directory / "latest.json"
    previous = load_json(latest_path)
    checked_at = datetime.now(UTC)
    trace_cutoff = checked_at - timedelta(hours=TRACE_TREND_HOURS)
    now_ms = int(checked_at.timestamp() * 1000)
    submitted_since_ms = (
        None if args.hours == 0 else now_ms - int(args.hours * 3_600_000)
    )

    jobs: list[HarborJob] = []
    errors: list[MonitorError] = []
    for cluster in CLUSTERS:
        try:
            found, cluster_errors = discover_harbor_jobs(
                cluster, submitted_since_ms=submitted_since_ms
            )
        except Exception as error:
            found, cluster_errors = [], [str(error)]
        jobs.extend(found)
        errors.extend(
            _monitor_error(cluster.name, "job discovery", error)
            for error in cluster_errors
        )
    if args.job:
        jobs = [job for job in jobs if job.job_id == args.job]
        if not jobs:
            errors.append(
                _monitor_error(
                    args.job,
                    "job selection",
                    "No matching active Harbor job was discovered.",
                )
            )
    jobs = filter_records(
        jobs, filters, lambda job: job_filter_values(job, now_ms=now_ms)
    )

    s3_clients: dict[str, Any] = {}
    gcs_client: Any | None = None
    local_logs: dict[tuple[str, str], tuple[Path | None, str | None, int | None]] = {}
    for job in jobs:
        key = (job.cluster.name, job.job_id)
        try:
            prior = previous.get("jobs", {}).get(job.job_id, {})
            prior_sync = prior.get("finelog_synced_at_ms")
            destination = finelog_path(job, args.bundle_root)
            if (
                prior_sync is None
                and destination.exists()
                and previous.get("checked_at")
            ):
                prior_sync = int(
                    datetime.fromisoformat(previous["checked_at"]).timestamp() * 1000
                )
            result = fetch_finelog(
                job,
                args.bundle_root,
                int(prior_sync) if prior_sync is not None else job.submitted_at_ms,
            )
        except Exception as error:
            result = (None, str(error), None)
        local_logs[key] = result
        if result[1]:
            errors.append(
                _monitor_error(
                    f"{job.cluster.name}/{job.job_id}", "Finelog sync", result[1]
                )
            )
    ray_vllm_logs: dict[tuple[str, str], tuple[str, str | None]] = {}
    for job in jobs:
        key = (job.cluster.name, job.job_id)
        try:
            result = fetch_ray_vllm_logs(job, args.bundle_root)
        except Exception as error:
            result = ("unavailable", str(error))
        ray_vllm_logs[key] = result
        if result[1]:
            errors.append(
                _monitor_error(
                    f"{job.cluster.name}/{job.job_id}", "Ray/vLLM sync", result[1]
                )
            )
    rows: list[list[object]] = []
    current_jobs: dict[str, Any] = {}
    for job in sorted(jobs, key=lambda item: (item.cluster.name, item.job_id)):
        key = (job.cluster.name, job.job_id)
        try:
            artifact_dir = (
                job_bundle(args.bundle_root, job.cluster.name, job.job_id).directory
                / "harbor"
            )
            if job.jobs_dir is None or job.harbor_job_name is None:
                if job.kind == "eval":
                    progress = read_pod_local_eval_progress(job, args.bundle_root)
                else:
                    local_log, _log_error, _synced_at = local_logs[key]
                    progress = Progress(None, None, finelog_activity(local_log))
            elif job.jobs_dir.startswith("s3://"):
                client = s3_clients.setdefault(
                    job.cluster.name, coreweave_client(job.cluster)
                )
                progress = read_s3_progress(job, client, artifact_dir, trace_cutoff)
            else:
                if gcs_client is None:
                    gcs_client = gcs_storage.Client()
                progress = read_gcs_progress(
                    job, gcs_client, artifact_dir, trace_cutoff
                )
        except Exception as error:
            progress = Progress(
                None, None, "unavailable", f"progress read failed: {error}"
            )
        if progress.error:
            errors.append(
                _monitor_error(
                    f"{job.cluster.name}/{job.job_id}", "progress read", progress.error
                )
            )
        if progress.recent_window_error:
            errors.append(
                _monitor_error(
                    f"{job.cluster.name}/{job.job_id}",
                    "two-hour trace trend",
                    progress.recent_window_error,
                )
            )
        try:
            health, last_advanced_at = health_label(
                job, progress, previous, checked_at, args.stalled_after_minutes
            )
        except Exception as error:
            health, last_advanced_at = "output-unavailable", checked_at.isoformat()
            errors.append(
                _monitor_error(
                    f"{job.cluster.name}/{job.job_id}", "health calculation", error
                )
            )
        local_log, _log_error, finelog_synced_at_ms = local_logs[key]
        ray_vllm_status, ray_vllm_error = ray_vllm_logs[key]
        if ray_vllm_error:
            ray_vllm_status = "unavailable"
        try:
            rows.append(report_row(job, progress, health, local_log, ray_vllm_status))
        except Exception as error:
            errors.append(
                _monitor_error(
                    f"{job.cluster.name}/{job.job_id}", "row rendering", error
                )
            )
            rows.append(
                [
                    f"{job.cluster.name}/{job.kind}",
                    job.job_id.rsplit("/", 1)[-1],
                    job.dataset,
                    _state_cell(job.state),
                    "?/?",
                    "—",
                    StyledCell("—", "muted"),
                    StyledCell("unavailable", "muted"),
                    StyledCell("unavailable", "error"),
                    StyledCell("status unavailable; see error report", "error"),
                ]
            )
        current_jobs[job.job_id] = {
            "cluster": job.cluster.name,
            "job_kind": job.kind,
            "completed": progress.completed,
            "total": progress.total,
            "mean_reward": progress.mean_reward,
            "error_counts": progress.error_counts,
            "exception_file_count": progress.exception_file_count,
            "recent_completed_2h": progress.recent_completed,
            "recent_errored_2h": progress.recent_errored,
            "last_advanced_at": last_advanced_at,
            "health": health,
            "jobs_dir": job.jobs_dir,
            "harbor_job_name": job.harbor_job_name,
            "dataset": job.dataset,
            "finelog_synced_at_ms": finelog_synced_at_ms,
            "ray_vllm_status": ray_vllm_status,
            "bundle_directory": str(
                job_bundle(args.bundle_root, job.cluster.name, job.job_id).directory
            ),
        }
        try:
            write_bundle_manifest(
                job_bundle(args.bundle_root, job.cluster.name, job.job_id),
                {
                    "kind": "harbor",
                    "job_kind": job.kind,
                    "dataset": job.dataset,
                    "harbor_job_name": job.harbor_job_name,
                    "jobs_dir": job.jobs_dir,
                    "state": job.state,
                    "submitted_at_ms": job.submitted_at_ms,
                    "last_synced_at": checked_at.isoformat(),
                    "progress": {
                        "completed": progress.completed,
                        "total": progress.total,
                        "mean_reward": progress.mean_reward,
                        "error_counts": progress.error_counts,
                        "exception_file_count": progress.exception_file_count,
                        "recent_completed_2h": progress.recent_completed,
                        "recent_errored_2h": progress.recent_errored,
                    },
                },
            )
        except Exception as error:
            errors.append(
                _monitor_error(
                    f"{job.cluster.name}/{job.job_id}", "manifest write", error
                )
            )

    headers = [
        "Target",
        "Harbor run",
        "Dataset",
        "State",
        "Trials",
        "Mean",
        "Trial errors",
        "Last 2h",
        "Evidence",
        "Health",
    ]
    if rows:
        table = box_table(headers, rows)
        terminal_table = box_table(headers, rows, color=sys.stdout.isatty())
    else:
        table = "No active Iris Harbor datagen or eval jobs discovered."
        terminal_table = table
    filter_suffix = f"; filters={','.join(args.filter)}" if args.filter else ""
    window = "all" if args.hours == 0 else f"{args.hours:g}h"
    timestamp = checked_at.strftime("%Y%m%dT%H%M%SZ")
    error_report_path = write_error_report(
        report_directory,
        timestamp,
        "Iris Harbor monitor errors",
        checked_at,
        errors,
    )
    error_summary = f"Monitor errors: {len(errors)}; details: {error_report_path}"
    heading = f"# Iris Harbor datagen / eval status — {checked_at.isoformat()}; submitted={window}{filter_suffix}"
    report = f"{heading}\n\n{table}\n\n{error_summary}\n"
    report_path = report_directory / f"{timestamp}.md"
    report_path.write_text(report)
    (report_directory / "latest.md").write_text(report)
    current = {
        "checked_at": checked_at.isoformat(),
        "jobs": current_jobs,
        "report": str(report_path),
        "error_count": len(errors),
        "error_report": str(error_report_path),
    }
    latest_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    if args.notify:
        changed = [
            job_id
            for job_id, data in current_jobs.items()
            if previous.get("jobs", {}).get(job_id, {}).get("health") != data["health"]
        ]
        if changed:
            notify(
                f"{len(changed)} Harbor health change(s); report saved to {report_directory / 'latest.md'}"
            )
    print(f"{heading}\n\n{terminal_table}\n\n{error_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
