from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from scripts.iris import coreweave_ops
from scripts.iris.iris_ops import (
    MonitorError,
    StyledCell,
    box_table,
    filter_records,
    format_duration,
    job_bundle,
    job_id_parts,
    load_bundle_manifest,
    parse_regex_filters,
    render_error_report,
    strip_ansi,
    write_error_report,
    write_bundle_manifest,
)
from scripts.iris import watch_coreweave_rl


def test_job_bundle_uses_cluster_and_full_iris_identity(tmp_path):
    bundle = job_bundle(tmp_path, "cw-rno2a", "/benjaminfeuer/glm52-r10")

    assert (
        bundle.directory
        == tmp_path / "jobs" / "cw-rno2a" / "benjaminfeuer" / "glm52-r10"
    )

    write_bundle_manifest(bundle, {"kind": "harbor", "progress": {"completed": 4}})

    assert (
        json.loads(bundle.manifest_path.read_text())["job_id"]
        == "/benjaminfeuer/glm52-r10"
    )
    assert load_bundle_manifest(bundle)["progress"] == {"completed": 4}


def test_shared_regex_filters_duration_and_table_renderer():
    records = [
        {"cluster": "cw-rno2a", "state": "running", "name": "glm52"},
        {"cluster": "marin", "state": "running", "name": "qwen"},
        {"cluster": "cw-rno2a", "state": "failed", "name": "glm52"},
    ]
    filters = parse_regex_filters(
        ["cluster=^cw-", "name=glm", "state=running"], {"cluster", "name", "state"}
    )

    assert filter_records(records, filters, lambda record: record) == [records[0]]
    assert format_duration(60_000, 7_320_000) == "2h 1m"
    assert "│ one │ two │" in box_table(["A", "B"], [["one", "two"]])


def test_box_table_wraps_to_width_and_sanitizes_multiline_cells():
    table = box_table(
        ["Job", "Status"],
        [["a-very-long-job-name-that-must-wrap", "line one\nline two"]],
        max_width=32,
    )

    assert max(len(line) for line in table.splitlines()) <= 32
    assert "line one" in table
    assert "line two" in table


def test_box_table_color_preserves_plain_layout_and_can_be_stripped():
    rows = [[StyledCell("running", "success"), StyledCell("warning", "warning")]]

    plain = box_table(["State", "Health"], rows, max_width=40)
    colored = box_table(["State", "Health"], rows, max_width=40, color=True)

    assert "\x1b[" not in plain
    assert "\x1b[" in colored
    assert strip_ansi(colored) == plain


def test_monitor_error_report_is_separate_stable_and_single_line(tmp_path):
    errors = [
        MonitorError("cw-rno2a/job-a", "Finelog sync", "proxy failed\nTraceback: details"),
        MonitorError("marin", "discovery", "controller unavailable"),
    ]

    report = render_error_report(
        "Iris Harbor monitor errors",
        datetime(2026, 7, 25, 12, tzinfo=UTC),
        errors,
    )
    path = write_error_report(
        tmp_path,
        "20260725T120000Z",
        "Iris Harbor monitor errors",
        datetime(2026, 7, 25, 12, tzinfo=UTC),
        errors,
    )

    assert "## cw-rno2a/job-a" in report
    assert "- **Finelog sync:** proxy failed Traceback: details" in report
    assert path == tmp_path / "20260725T120000Z.errors.md"
    assert path.read_text() == report
    assert (tmp_path / "latest-errors.md").read_text() == report


def test_shared_regex_filter_rejects_unknown_fields_and_invalid_regexes():
    with pytest.raises(ValueError, match="Unknown filter field"):
        parse_regex_filters(["missing=value"], {"state"})
    with pytest.raises(ValueError, match="Invalid regex"):
        parse_regex_filters(["state=["], {"state"})


@pytest.mark.parametrize(
    "job_id", ["glm52-r10", "/benjaminfeuer", "/benjaminfeuer/../bad"]
)
def test_job_id_parts_rejects_noncanonical_or_unsafe_ids(job_id):
    with pytest.raises(ValueError):
        job_id_parts(job_id)


def test_find_pod_uses_untruncated_iris_job_label(monkeypatch):
    job_id = "/benjaminfeuer/grug-agentic-eval-v2-65k-harbor394c-r3"
    monkeypatch.setattr(
        coreweave_ops,
        "command",
        lambda _args: json.dumps(
            {
                "items": [
                    {
                        "metadata": {
                            "name": "iris-benjaminfeuer-grug-agentic-eval-v2-65k-harbor39-c309bb6c-0",
                            "labels": {
                                "iris.job_id": "benjaminfeuer.grug-agentic-eval-v2-65k-harbor394c-r3"
                            },
                        },
                        "status": {"phase": "Running"},
                    }
                ]
            }
        ),
    )

    assert coreweave_ops.find_pod(
        ["kubectl"], SimpleNamespace(job=job_id, pod=None)
    ) == ("iris-benjaminfeuer-grug-agentic-eval-v2-65k-harbor39-c309bb6c-0")


def test_save_ray_logs_reports_empty_tar_stream_as_sync_error(monkeypatch, tmp_path):
    class EmptyTarProcess:
        stdout = BytesIO()
        stderr = BytesIO(b"tar: session log vanished")

        def wait(self) -> int:
            return 1

    monkeypatch.setattr(
        coreweave_ops.subprocess, "Popen", lambda *_args, **_kwargs: EmptyTarProcess()
    )

    with pytest.raises(RuntimeError, match="Could not archive Ray/vLLM logs"):
        coreweave_ops.save_ray_logs(
            ["kubectl"],
            "pod",
            "task",
            [{"path": "worker-1.out", "size": 10}],
            100,
            tmp_path,
        )
    assert (
        tmp_path / "ray-vllm-sync-error.txt"
    ).read_text() == "tar: session log vanished"


def test_save_ray_logs_retries_transient_kubectl_html(monkeypatch, tmp_path):
    archive_bytes = BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo("worker-1.out")
        payload = b"ray log\n"
        member.size = len(payload)
        archive.addfile(member, BytesIO(payload))

    class TarProcess:
        def __init__(self, stdout: bytes, stderr: bytes, return_code: int):
            self.stdout = BytesIO(stdout)
            self.stderr = BytesIO(stderr)
            self.return_code = return_code

        def wait(self) -> int:
            return self.return_code

    processes = iter(
        [
            TarProcess(b"", b"<!doctype html><html>temporary proxy error", 1),
            TarProcess(archive_bytes.getvalue(), b"", 0),
        ]
    )
    delays: list[int] = []
    monkeypatch.setattr(
        coreweave_ops.subprocess, "Popen", lambda *_args, **_kwargs: next(processes)
    )
    monkeypatch.setattr(coreweave_ops.time, "sleep", delays.append)

    saved, skipped = coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        [{"path": "worker-1.out", "size": 10}],
        100,
        tmp_path,
    )

    assert saved == [{"path": "worker-1.out", "size": 10}]
    assert skipped == []
    assert delays == [coreweave_ops.DNS_INITIAL_BACKOFF]
    assert (tmp_path / "worker-1.out").read_bytes() == b"ray log\n"


def _ray_delta_archive(entries: list[tuple[str, int, bytes]]) -> bytes:
    archive_bytes = BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for name, offset, payload in entries:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.pax_headers = {"otagent.offset": str(offset)}
            archive.addfile(member, BytesIO(payload))
    return archive_bytes.getvalue()


def test_save_ray_logs_incrementally_appends_and_replaces_rotated_logs(
    monkeypatch, tmp_path
):
    class CapturingInput(BytesIO):
        def close(self) -> None:
            self.closed_by_sync = True

    class TarProcess:
        def __init__(self, archive: bytes):
            self.stdin = CapturingInput()
            self.stdout = BytesIO(archive)
            self.stderr = BytesIO()

        def wait(self) -> int:
            return 0

    commands: list[list[str]] = []
    inputs: list[CapturingInput] = []
    archives = iter(
        [
            _ray_delta_archive([("worker-1.out", 0, b"abc")]),
            _ray_delta_archive([("worker-1.out", 3, b"def")]),
            _ray_delta_archive([("worker-1.out", 0, b"xy")]),
        ]
    )

    def fake_popen(command, **_kwargs):
        commands.append(command)
        process = TarProcess(next(archives))
        inputs.append(process.stdin)
        return process

    monkeypatch.setattr(coreweave_ops.subprocess, "Popen", fake_popen)
    initial = [{"path": "worker-1.out", "size": 3, "inode": 41}]
    saved, skipped = coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        initial,
        100,
        tmp_path,
        incremental=True,
        python_executable="python",
    )
    assert (saved, skipped) == (initial, [])
    assert (tmp_path / "worker-1.out").read_bytes() == b"abc"

    grown = [{"path": "worker-1.out", "size": 6, "inode": 41}]
    coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        grown,
        100,
        tmp_path,
        incremental=True,
        python_executable="python",
    )
    assert (tmp_path / "worker-1.out").read_bytes() == b"abcdef"
    assert b'"offset": 3' in inputs[1].getvalue()
    assert getattr(inputs[1], "closed_by_sync", False)
    assert "-i" in commands[1]

    # An unchanged file does not open another kubectl exec stream.
    coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        grown,
        100,
        tmp_path,
        incremental=True,
        python_executable="python",
    )
    assert len(commands) == 2

    rotated = [{"path": "worker-1.out", "size": 2, "inode": 99}]
    coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        rotated,
        100,
        tmp_path,
        incremental=True,
        python_executable="python",
    )
    assert (tmp_path / "worker-1.out").read_bytes() == b"xy"
    assert b'"offset": 0' in inputs[2].getvalue()


def test_coreweave_command_retries_transient_kubectl_html(monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess(
                [], 1, stderr="<!doctype html><html>temporary proxy error"
            ),
            subprocess.CompletedProcess([], 0, stdout="ok\n"),
        ]
    )
    delays: list[int] = []
    monkeypatch.setattr(
        coreweave_ops.subprocess, "run", lambda *_args, **_kwargs: next(results)
    )
    monkeypatch.setattr(coreweave_ops.time, "sleep", delays.append)

    assert coreweave_ops.command(["kubectl", "get", "pods"]) == "ok\n"
    assert delays == [coreweave_ops.DNS_INITIAL_BACKOFF]


def test_coreweave_command_retries_generic_kubectl_exec_transport_failure(monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess(
                [], 1, stderr="command terminated with exit code 1"
            ),
            subprocess.CompletedProcess([], 0, stdout="ok\n"),
        ]
    )
    delays: list[int] = []
    monkeypatch.setattr(
        coreweave_ops.subprocess, "run", lambda *_args, **_kwargs: next(results)
    )
    monkeypatch.setattr(coreweave_ops.time, "sleep", delays.append)

    assert coreweave_ops.command(["kubectl", "exec", "pod", "--", "true"]) == "ok\n"
    assert delays == [coreweave_ops.DNS_INITIAL_BACKOFF]


def test_ray_log_inventory_uses_explicit_python_for_rl_images(monkeypatch):
    monkeypatch.setattr(
        coreweave_ops,
        "resolve_container_python",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("generic discovery should not run")
        ),
    )
    commands: list[list[str]] = []

    def fake_command(arguments, **_kwargs):
        commands.append(arguments)
        return "[]"

    monkeypatch.setattr(coreweave_ops, "command", fake_command)

    assert (
        coreweave_ops.ray_log_inventory(
            ["kubectl"],
            "pod",
            "task",
            python_executable="/opt/openthoughts/envs/rl/bin/python",
        )
        == []
    )
    assert "/opt/openthoughts/envs/rl/bin/python" in commands[0]
    assert "st_ino" in commands[0][-1]


def test_rl_sync_warning_never_renders_proxy_html():
    warning = watch_coreweave_rl.sync_warning(
        ("pod Ray/vLLM: Could not archive logs (<!doctype html><html>proxy body)",)
    )

    assert warning == "Ray/vLLM log sync unavailable; local diagnostic saved"


def test_rl_report_row_keeps_artifact_exceptions_out_of_trend(tmp_path):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.RlJob(
        cluster,
        "/user/rl-job",
        "running",
        0,
        "",
        dataset="DCAgent/tasks",
    )
    artifacts = watch_coreweave_rl.ArtifactResult(
        "unavailable",
        "unavailable",
        "unavailable",
        "unavailable",
        None,
        None,
        ("Ray/vLLM: raw proxy exception body",),
    )

    row = watch_coreweave_rl.report_row(job, artifacts, tmp_path)

    assert "raw proxy exception body" not in repr(row)
    assert len(row) == 9
    assert row[2].value == "running"


def test_rl_report_row_replaces_traceback_signal_with_error_report_pointer(tmp_path):
    (tmp_path / "finelog.log").write_text("Traceback (most recent call last)\nraw details\n")
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.RlJob(cluster, "/user/rl-job", "failed", 0, "")
    artifacts = watch_coreweave_rl.ArtifactResult(
        "synced", "synced", "synced", "synced", None, None, ()
    )

    row = watch_coreweave_rl.report_row(job, artifacts, tmp_path)

    assert "Traceback" not in repr(row)
    assert row[-1].value == "workload error detected; see error report"


def test_rl_main_degrades_unexpected_job_sync_failure_into_error_report(
    monkeypatch, tmp_path, capsys
):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.RlJob(
        cluster,
        "/user/rl-job",
        "running",
        1,
        "",
        dataset="DCAgent/tasks",
    )
    monkeypatch.setattr(
        watch_coreweave_rl,
        "parse_args",
        lambda: SimpleNamespace(
            max_non_log_bytes=0,
            trace_sync_limit=0,
            hours=24.0,
            all_users=False,
            user="user",
            filter=[],
            bundle_root=tmp_path,
            quiet_progress=True,
            no_sync=True,
        ),
    )
    monkeypatch.setattr(watch_coreweave_rl, "CLUSTERS", (cluster,))
    monkeypatch.setattr(
        watch_coreweave_rl,
        "discover_rl_jobs",
        lambda *_args, **_kwargs: ([job], []),
    )
    monkeypatch.setattr(
        watch_coreweave_rl,
        "sync_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LookupError("unexpected raw sync exception")
        ),
    )
    monkeypatch.setattr(watch_coreweave_rl, "write_job_manifest", lambda *_args, **_kwargs: None)

    assert watch_coreweave_rl.main() == 0

    stdout = capsys.readouterr().out
    assert "unexpected raw sync exception" not in stdout
    errors = (tmp_path / "reports/rl/latest-errors.md").read_text()
    assert "unexpected raw sync exception" in errors


def test_rl_progress_reporter_writes_phase_and_elapsed_time(monkeypatch, capsys):
    monkeypatch.setattr(watch_coreweave_rl.time, "monotonic", lambda: 165.0)

    reporter = watch_coreweave_rl.ProgressReporter(started_at=100.0)
    reporter.phase("trace inventory 1/2")

    assert capsys.readouterr().err == "[rl-watch +01:05] trace inventory 1/2\n"


def test_rl_discovery_skips_iris_preamble_and_includes_recent_terminal_jobs(
    monkeypatch,
):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    output = """I20260722 controller tunnel ready
job_id,state,submitted_at_ms,finished_at_ms,entrypoint_json
/benjaminfeuer/rl-live,3,1000,,"start_rl_iris_controller.py --train_data '[\\"live\\"]'"
/benjaminfeuer/rl-failed,5,900,1500,"start_rl_iris_controller.py --train_data '[\\"failed\\"]'"
"""
    queries: list[str] = []

    def fake_run_iris(_cluster, arguments, **_kwargs):
        queries.append(arguments[1])
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr(watch_coreweave_rl, "run_iris", fake_run_iris)

    jobs, errors = watch_coreweave_rl.discover_rl_jobs(
        cluster,
        "benjaminfeuer",
        submitted_since_ms=1200,
    )

    assert errors == []
    assert [job.short_name for job in jobs] == ["rl-live", "rl-failed"]
    assert [job.is_terminal for job in jobs] == [False, True]
    assert "OR j.state IN (4,5)" in queries[0]
    assert "j.submitted_at_ms >= 1200" in queries[0]


def test_recent_trace_jobs_uses_remote_last_modified_and_preserves_remote_counts():
    root = "iris/rl/trace_jobs/"
    first = datetime(2026, 7, 22, 10, tzinfo=UTC)
    objects = [
        {"Key": f"{root}z-old/result.json", "Size": 1, "LastModified": first},
        {
            "Key": f"{root}a-new/agent/trajectory.json",
            "Size": 1,
            "LastModified": first + timedelta(hours=3),
        },
        {
            "Key": f"{root}a-new/result.json",
            "Size": 1,
            "LastModified": first + timedelta(hours=2),
        },
        {
            "Key": f"{root}m-middle/result.json",
            "Size": 1,
            "LastModified": first + timedelta(hours=2),
        },
    ]

    selected, available, completed = watch_coreweave_rl.recent_trace_jobs(
        objects, root, trace_sync_limit=2
    )

    assert [trace.name for trace in selected] == ["a-new", "m-middle"]
    assert available == 3
    assert completed == 3
    full, _, _ = watch_coreweave_rl.recent_trace_jobs(objects, root, trace_sync_limit=0)
    assert [trace.name for trace in full] == ["a-new", "m-middle", "z-old"]

    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    first_job = watch_coreweave_rl.RlJob(cluster, "/user/first", "running", 0, "")
    second_job = watch_coreweave_rl.RlJob(cluster, "/user/second", "running", 0, "")
    first_inventory = watch_coreweave_rl.TraceInventory(
        first_job, "bucket", root, object(), tuple(full), 3, 3
    )
    second_inventory = watch_coreweave_rl.TraceInventory(
        second_job,
        "bucket",
        root,
        object(),
        (
            watch_coreweave_rl.TraceJobObjects(
                "other", first + timedelta(hours=4), (), True
            ),
        ),
        1,
        1,
    )

    fleet_selected = watch_coreweave_rl.select_recent_fleet_traces(
        [first_inventory, second_inventory], trace_sync_limit=2
    )

    assert [trace.name for trace in fleet_selected[("cw-rno2a", "/user/first")]] == [
        "a-new"
    ]
    assert [trace.name for trace in fleet_selected[("cw-rno2a", "/user/second")]] == [
        "other"
    ]
    manifest = watch_coreweave_rl.trace_selection_manifest(
        fleet_selected[("cw-rno2a", "/user/first")],
        first_inventory,
        2,
        fleet_available=4,
        fleet_selected=2,
    )
    assert (
        manifest["selection"]
        == "latest_object_store_last_modified_across_active_rl_jobs"
    )
    assert manifest["omitted_traces"] == 2
    assert manifest["fleet_selected_traces"] == 2

    full_fleet = watch_coreweave_rl.select_recent_fleet_traces(
        [first_inventory, second_inventory], trace_sync_limit=0
    )
    assert sum(len(traces) for traces in full_fleet.values()) == 4


def test_recent_trace_jobs_requires_remote_last_modified_metadata():
    with pytest.raises(ValueError, match="LastModified"):
        watch_coreweave_rl.recent_trace_jobs(
            [{"Key": "iris/rl/trace_jobs/trace/result.json", "Size": 1}],
            "iris/rl/trace_jobs/",
            trace_sync_limit=500,
        )


def test_trace_sync_skips_object_that_disappears_after_listing(tmp_path):
    class MissingObjectClient:
        def download_file(self, _bucket, _key, _destination):
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )

    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.RlJob(cluster, "/user/rl-job", "running", 0, "")
    root = "iris/rl-job/trace_jobs/"
    trace = watch_coreweave_rl.TraceJobObjects(
        "trial-1",
        datetime(2026, 7, 22, 10, tzinfo=UTC),
        (
            {
                "Key": f"{root}trial-1/result.json",
                "Size": 1,
                "LastModified": datetime(2026, 7, 22, 10, tzinfo=UTC),
            },
        ),
        True,
    )
    inventory = watch_coreweave_rl.TraceInventory(
        job, "bucket", root, MissingObjectClient(), (trace,), available=1, completed=1
    )

    status, available, completed, error = watch_coreweave_rl.sync_trace_inventory(
        inventory,
        tmp_path / "trace_jobs",
        [trace],
        max_non_log_bytes=0,
        trace_sync_limit=500,
        fleet_available=1,
        fleet_selected=1,
    )

    assert status.endswith("0 copied, 1 skipped")
    assert (available, completed, error) == (1, 1, None)
    assert json.loads(
        (tmp_path / "trace_jobs" / "skipped_objects.json").read_text()
    ) == [
        {
            "key": "trial-1/result.json",
            "reason": "missing_after_listing",
            "size": 1,
        }
    ]
